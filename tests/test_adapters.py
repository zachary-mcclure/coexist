"""Tests for coexist.adapters — differentiable external-engine wrappers."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from coexist.adapters import SimulatorAdapter, ASEAdapter


# ── Analytic reference engine: harmonic wells, exact gradients ────────────────

K_SPRING = 2.5
R0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.5, 0.0]])


def harmonic_engine(pos):
    """E = ½k Σ|R−R0|², F = −k(R−R0). Plays the role of an external code."""
    d = pos - R0
    return 0.5 * K_SPRING * float((d ** 2).sum()), -K_SPRING * d


@pytest.fixture
def harmonic():
    return SimulatorAdapter(harmonic_engine, n_atoms=3, name="harmonic")


def test_energy_matches_engine(harmonic):
    R = jnp.array(R0 + 0.1)
    E = float(harmonic(R))
    expected = 0.5 * K_SPRING * 3 * 3 * 0.1 ** 2   # 9 coords displaced by 0.1
    assert abs(E - expected) < 1e-5


def test_grad_is_minus_forces_exact(harmonic):
    R = jnp.array(R0 + np.random.default_rng(0).normal(0, 0.2, R0.shape))
    g = jax.grad(harmonic)(R)
    expected = K_SPRING * (np.asarray(R) - R0)      # ∂E/∂R = k(R−R0) = −F
    np.testing.assert_allclose(np.asarray(g), expected, rtol=1e-5)


def test_forces_method(harmonic):
    R = jnp.array(R0 + 0.05)
    F = harmonic.forces(R)
    np.testing.assert_allclose(np.asarray(F), -K_SPRING * 0.05 * np.ones_like(R0),
                               rtol=1e-5)


def test_single_engine_call_per_value_and_grad():
    adapter = SimulatorAdapter(harmonic_engine, n_atoms=3)
    R = jnp.array(R0 + 0.1)
    adapter.n_calls = 0
    E, F = adapter.energy_and_forces(R)
    assert adapter.n_calls == 1, "fwd+bwd must share ONE engine call"


def test_works_under_jit(harmonic):
    R = jnp.array(R0 + 0.1)

    @jax.jit
    def loss(r):
        return harmonic(r) ** 2

    g = jax.grad(loss)(R)
    E = float(harmonic(R))
    expected = 2.0 * E * K_SPRING * 0.1
    np.testing.assert_allclose(np.asarray(g), expected * np.ones_like(R0),
                               rtol=1e-4)


def test_composes_with_downstream_jax(harmonic):
    """External engine energy feeding a nonlinear JAX chain — grads flow."""
    R = jnp.array(R0 + 0.1)

    def chain(r):
        E = harmonic(r)
        return jnp.tanh(E) * jnp.sum(r ** 2)

    g = jax.grad(chain)(R)
    assert np.all(np.isfinite(np.asarray(g)))
    # FD spot check on one coordinate
    h = 1e-3
    Rp = np.asarray(R).copy(); Rp[1, 0] += h
    Rm = np.asarray(R).copy(); Rm[1, 0] -= h
    fd = (float(chain(jnp.array(Rp))) - float(chain(jnp.array(Rm)))) / (2 * h)
    assert abs(float(g[1, 0]) - fd) / (abs(fd) + 1e-8) < 5e-2


# ── ASE adapter with the EMT engine (proxy for LAMMPS/VASP) ──────────────────

ase = pytest.importorskip("ase")
from ase.calculators.emt import EMT               # noqa: E402
from ase.cluster import Icosahedron               # noqa: E402


@pytest.fixture(scope="module")
def cu13():
    atoms = Icosahedron("Cu", noshells=2)          # 13-atom cluster
    return np.array(atoms.get_positions()), [29] * len(atoms)


def test_ase_adapter_energy_matches_ase(cu13):
    pos, z = cu13
    adapter = ASEAdapter(z, lambda: EMT())
    E_adapter = float(adapter(jnp.array(pos)))

    from ase import Atoms
    ref = Atoms(numbers=z, positions=pos)
    ref.calc = EMT()
    assert abs(E_adapter - ref.get_potential_energy()) < 1e-4


def test_ase_adapter_grad_matches_ase_forces(cu13):
    pos, z = cu13
    adapter = ASEAdapter(z, lambda: EMT())
    g = jax.grad(adapter)(jnp.array(pos))

    from ase import Atoms
    ref = Atoms(numbers=z, positions=pos)
    ref.calc = EMT()
    np.testing.assert_allclose(np.asarray(g), -ref.get_forces(),
                               rtol=1e-4, atol=1e-5)


def test_ase_adapter_grad_vs_fd(cu13):
    """AD through the adapter vs central FD on the EMT engine itself."""
    pos, z = cu13
    adapter = ASEAdapter(z, lambda: EMT(), dtype=jnp.float32)
    g = np.asarray(jax.grad(adapter)(jnp.array(pos)))

    h = 1e-4
    for (i, a) in [(0, 0), (5, 1), (12, 2)]:
        pp = pos.copy(); pp[i, a] += h
        pm = pos.copy(); pm[i, a] -= h
        fd = (float(adapter(jnp.array(pp))) - float(adapter(jnp.array(pm)))) / (2 * h)
        assert abs(g[i, a] - fd) / (abs(fd) + 1e-6) < 1e-2


# ── Real LAMMPS through the adapter (skipped if lammps wheel not installed) ──
# macOS install: `uv pip install lammps`, `brew install mpich`, then symlink
# /opt/homebrew/lib/lib{mpi,pmpi}.12.dylib into site-packages/lammps/.

lammps_mod = pytest.importorskip("lammps")
from ase.build import bulk as _bulk                      # noqa: E402


def _lammps_factory():
    from ase.calculators.lammpslib import LAMMPSlib
    return LAMMPSlib(
        lmpcmds=["pair_style lj/cut 6.0", "pair_coeff * * 0.4093 2.338"],
        atom_types={"Cu": 1}, keep_alive=True, log_file=None)


def test_real_lammps_grad_matches_lammps_forces():
    cu = _bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 2, 2))
    rng = np.random.default_rng(1)
    pos = cu.get_positions() + rng.normal(0, 0.05, (len(cu), 3))

    adapter = ASEAdapter(cu.get_atomic_numbers().tolist(), _lammps_factory,
                         cell=cu.get_cell()[:], pbc=True, name="LAMMPS")
    g = np.asarray(jax.grad(adapter)(jnp.array(pos)))

    ref = cu.copy()
    ref.calc = _lammps_factory()
    ref.set_positions(pos)
    np.testing.assert_allclose(g, -ref.get_forces(), atol=1e-5)
