"""
SimulatorAdapter — differentiable I/O adapter for EXTERNAL simulation engines.

The product thesis: don't rebuild physics in JAX — wrap production codes
(VASP, LAMMPS, SIESTA, Quantum Espresso, CP2K, ORCA, ...) so that any chain of
real simulators becomes differentiable end-to-end. Every quantum/MD engine
already computes the quantity its own gradient needs: forces (= −∂E/∂R) come
free with the SCF or the pair-potential evaluation. This module packages that
fact as a JAX `custom_vjp`, so `jax.grad` flows through an external subprocess
as if it were native JAX.

Design (one improvement over the older XTBLayer pattern in layers/electronic.py):
the forward callback fetches energy AND forces in a single engine call and
stores the forces as the VJP residual. An expensive engine (a VASP SCF, a
LAMMPS minimize) therefore runs ONCE per forward+backward pass — the older
pattern re-ran the SCF in the backward pass.

Two entry points:

  SimulatorAdapter(eval_fn, n_atoms)
      Fully generic. `eval_fn(positions_np) -> (energy, forces)` may do
      anything — subprocess a licensed binary, call a REST endpoint, read a
      queue. Positions in Å, energy in eV, forces in eV/Å (the Coexist unit
      contract); adapt units inside your eval_fn.

  ASEAdapter(atomic_numbers, calculator_factory, cell=None, pbc=False)
      Wraps any ASE calculator — one class covers every engine ASE speaks to:
      EMT (zero-dependency test engine), LAMMPS (`ase.calculators.lammpslib` /
      `lammpsrun`), VASP (`ase.calculators.vasp`), SIESTA, Quantum Espresso,
      GPAW, CP2K, ORCA, Psi4, ... Swapping engines is one constructor argument.

Both compose with everything else in Coexist:

    adapter = ASEAdapter([29]*13, lambda: EMT())          # Cu13 cluster, EMT
    E = adapter(R)                                        # JAX scalar, eV
    F = -jax.grad(adapter)(R)                             # engine's own forces
    # ... feed E into the CG→PF bridge → CH → FEM → jax.grad end-to-end

Limitations (v1, stated honestly):
  * Gradients w.r.t. atomic POSITIONS only. Cell/stress differentiation and
    engine-parameter gradients (e.g. ∂E/∂(pair-style coefficients)) are the
    natural v2 — LAMMPS `compute pressure` and VASP stress tensors make the
    same trick work for the cell degree of freedom.
  * Higher-order derivatives (jax.hessian) would require engine Hessians;
    only first-order VJPs are exact here.
  * The callback is opaque to jax.jit fusion (it is a host call) and executes
    serially under vmap.
"""
from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


class SimulatorAdapter:
    """Differentiable wrapper around an external (energy, forces) evaluator.

    Args:
        eval_fn:  callable, positions (n_atoms, 3) float64 ndarray [Å] →
                  (energy [eV, float], forces (n_atoms, 3) ndarray [eV/Å]).
                  Runs on the host — free to subprocess anything.
        n_atoms:  number of atoms (fixes callback output shapes).
        dtype:    jnp.float32 (default, matches Coexist) or jnp.float64.
        name:     label used in __repr__ / diagnostics.
    """

    def __init__(
        self,
        eval_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
        n_atoms: int,
        dtype=jnp.float32,
        name: str = "external",
    ):
        self._eval_fn = eval_fn
        self._n = int(n_atoms)
        self._dtype = dtype
        self._name = name
        self.n_calls = 0          # engine-call counter (fwd+bwd share one call)
        self._fn = self._build()

    # ── host-side single engine call ──────────────────────────────────────────
    def _eval_both_np(self, positions) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(positions, dtype=np.float64)
        energy, forces = self._eval_fn(pos)
        self.n_calls += 1
        np_dt = np.float32 if self._dtype == jnp.float32 else np.float64
        return (np.asarray(energy, dtype=np_dt).reshape(()),
                np.asarray(forces, dtype=np_dt).reshape(self._n, 3))

    # ── JAX bridge: one callback serves forward AND backward ─────────────────
    def _build(self):
        out_shapes = (
            jax.ShapeDtypeStruct((), self._dtype),
            jax.ShapeDtypeStruct((self._n, 3), self._dtype),
        )
        eval_both = self._eval_both_np

        @jax.custom_vjp
        def energy(positions):
            E, _ = jax.pure_callback(eval_both, out_shapes, positions)
            return E

        def energy_fwd(positions):
            E, F = jax.pure_callback(eval_both, out_shapes, positions)
            return E, F          # forces ride along as the residual

        def energy_bwd(F, g):
            return (-F * g,)     # ∂E/∂R = −forces; engine not called again

        energy.defvjp(energy_fwd, energy_bwd)
        return energy

    # ── public API (mirrors XTBLayer so adapters slot in transparently) ──────
    def __call__(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Scalar energy in eV; differentiable w.r.t. positions via jax.grad."""
        return self._fn(positions)

    def forces(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Engine forces in eV/Å (via the VJP — one engine call)."""
        return -jax.grad(self.__call__)(positions)

    def energy_and_forces(self, positions: jnp.ndarray):
        E, pull = jax.vjp(self._fn, positions)
        (dE_dR,) = pull(jnp.ones((), dtype=self._dtype))
        return E, -dE_dR

    def jax_md_energy_fn(self):
        """Energy function usable by jax-md / ReversibleMDLayer."""
        fn = self._fn

        def wrapped(positions, **_):
            return fn(positions)

        return wrapped

    def __repr__(self):
        return (f"SimulatorAdapter(engine={self._name!r}, n_atoms={self._n}, "
                f"dtype={self._dtype.__name__}, engine_calls={self.n_calls})")


class ASEAdapter(SimulatorAdapter):
    """SimulatorAdapter for any ASE calculator.

    Args:
        atomic_numbers:      length-N element list (fixed per instance).
        calculator_factory:  zero-arg callable returning a fresh ASE calculator
                             (a factory, not an instance — some ASE calculators
                             are stateful across attach/detach).
        cell:                optional (3,3) cell in Å (enables periodic engines
                             like LAMMPS / VASP).
        pbc:                 periodic boundary flags (bool or 3-tuple).
        dtype, name:         forwarded to SimulatorAdapter.

    Example — three engines, one line apart:
        ASEAdapter(Z, lambda: EMT())                            # test engine
        ASEAdapter(Z, lambda: LAMMPSlib(lmpcmds=[...]), cell=C, pbc=True)
        ASEAdapter(Z, lambda: Vasp(xc="PBE", kpts=(4,4,4)), cell=C, pbc=True)
    """

    def __init__(
        self,
        atomic_numbers: Sequence[int],
        calculator_factory: Callable,
        cell=None,
        pbc=False,
        dtype=jnp.float32,
        name: str | None = None,
    ):
        from ase import Atoms

        z = list(int(a) for a in atomic_numbers)
        calc = calculator_factory()
        atoms = Atoms(numbers=z, positions=np.zeros((len(z), 3)))
        if cell is not None:
            atoms.set_cell(np.asarray(cell, dtype=np.float64))
        atoms.set_pbc(pbc)
        atoms.calc = calc
        self._atoms = atoms

        def eval_fn(pos: np.ndarray):
            self._atoms.set_positions(pos)
            E = float(self._atoms.get_potential_energy())
            F = np.asarray(self._atoms.get_forces(), dtype=np.float64)
            return E, F

        super().__init__(
            eval_fn, n_atoms=len(z), dtype=dtype,
            name=name or type(calc).__name__,
        )
