"""
MACE-MP-0 as a pure JAX energy function.

Strategy: wrap mace-torch via jax.pure_callback + custom_vjp.

  Forward : positions (JAX) → pure_callback → mace-torch → energy (JAX)
  Backward: upstream gradient g → pure_callback → mace-torch forces → g * (-F)

Result: jax.grad, jax.jit, and jax.lax.scan all work transparently.
Model weights are frozen (mace-torch side); only geometry is differentiated.

Why not mace-jax directly?
  mace-jax's graph builder (get_neighborhood) is NumPy-level, not JIT-able.
  Integrating it with JAX-MD's dynamic neighbor list is non-trivial and still
  evolving upstream. This approach ships today and is correct.
"""
from __future__ import annotations

import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np


class MACEEnergyFn:
    """
    MACE-MP-0 universal potential callable as a JAX energy function.

    After construction, instances behave like any JAX scalar function:
        E = fn(positions)            # jax.jit works
        F = -jax.grad(fn)(positions) # forces, exact autodiff
        H = jax.hessian(fn)(pos)    # Hessian, free

    The function is compatible with jax.lax.scan, so DiffMDLayer can run
    MACE trajectories with full end-to-end differentiability.

    Args:
        atomic_numbers: integer array of atomic numbers, length N. Fixed for
            the lifetime of this object (structure topology doesn't change).
        model: MACE-MP-0 size — "small", "medium" (default), or "large".
        device: "cpu", "cuda", or "mps".
        dispersion: add D3 dispersion correction (recommended for molecules).
        cell: (3,3) lattice vectors in Å, or None for isolated molecules.
        pbc: periodic boundary conditions, default (False, False, False).
    """

    def __init__(
        self,
        atomic_numbers: Sequence[int],
        model: str = "medium",
        device: str = "cpu",
        dispersion: bool = False,
        cell: np.ndarray | None = None,
        pbc: tuple[bool, bool, bool] = (False, False, False),
    ):
        self._z = np.asarray(atomic_numbers, dtype=np.int32)
        self._cell = cell
        self._pbc = pbc
        self._calc = self._load_calc(model, device, dispersion)
        self._n_atoms = len(self._z)

        # Build and cache the custom-VJP JAX function once
        self._fn = self._make_jax_fn()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_calc(self, model, device, dispersion):
        try:
            from mace.calculators import mace_mp
        except ImportError as exc:
            raise ImportError(
                "mace-torch is not installed. Run: pip install mace-torch"
            ) from exc
        return mace_mp(model=model, device=device, dispersion=dispersion,
                       default_dtype="float32")

    def _eval_energy_np(self, positions_np: np.ndarray) -> np.ndarray:
        """Evaluate energy in NumPy space (called from pure_callback)."""
        from ase import Atoms
        atoms = Atoms(numbers=self._z, positions=positions_np,
                      cell=self._cell, pbc=self._pbc)
        atoms.calc = self._calc
        return np.array(atoms.get_potential_energy(), dtype=np.float32)

    def _eval_forces_np(self, positions_np: np.ndarray) -> np.ndarray:
        """Evaluate forces in NumPy space (called from pure_callback). Shape (N, 3)."""
        from ase import Atoms
        atoms = Atoms(numbers=self._z, positions=positions_np,
                      cell=self._cell, pbc=self._pbc)
        atoms.calc = self._calc
        return np.array(atoms.get_forces(), dtype=np.float32)

    def _make_jax_fn(self):
        """
        Returns a JAX function with custom VJP wired to mace-torch forces.

        custom_vjp pattern:
          fwd: run energy via pure_callback, stash positions as residuals
          bwd: run forces via pure_callback, return g * (-F) as position gradient
        """
        eval_E = self._eval_energy_np
        eval_F = self._eval_forces_np
        n = self._n_atoms

        E_shape = jax.ShapeDtypeStruct((), jnp.float32)
        F_shape = jax.ShapeDtypeStruct((n, 3), jnp.float32)

        @jax.custom_vjp
        def energy(positions: jnp.ndarray) -> jnp.ndarray:
            return jax.pure_callback(eval_E, E_shape, positions)

        def energy_fwd(positions):
            E = energy(positions)
            return E, positions          # residuals = positions (needed for bwd)

        def energy_bwd(positions, g):
            # g: upstream scalar gradient  dL/dE
            # F: forces = -dE/dR,  so  dE/dR = -F
            # chain rule: dL/dR = dL/dE * dE/dR = g * (-F)
            F = jax.pure_callback(eval_F, F_shape, positions)
            return (-F * g,)             # same structure as energy's inputs

        energy.defvjp(energy_fwd, energy_bwd)
        return energy

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def __call__(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Scalar energy in eV. Differentiable via jax.grad."""
        return self._fn(positions)

    def forces(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Forces in eV/Å. Exact via autograd — one mace-torch call."""
        return -jax.grad(self.__call__)(positions)

    def jax_md_energy_fn(self):
        """
        Returns an energy function compatible with jax_md.simulate.nve / nvt.
        JAX-MD passes extra kwargs (neighbor, etc.) that we absorb.
        """
        fn = self._fn

        def wrapped(positions, **_):
            return fn(positions)

        return wrapped
