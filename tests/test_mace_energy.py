"""MACEEnergyFn tests — require mace-torch (pip install mace-torch)."""
import pytest

pytest.importorskip("mace", reason="mace-torch not installed")

import jax
import jax.numpy as jnp
import numpy as np

from coexist.layers.mace_energy import MACEEnergyFn

# Water molecule — small, fast to evaluate
H2O_Z = np.array([8, 1, 1])
H2O_R = jnp.array([
    [ 0.000,  0.000,  0.119],
    [ 0.000,  0.757, -0.477],
    [ 0.000, -0.757, -0.477],
], dtype=jnp.float32)


@pytest.fixture(scope="module")
def mace_fn():
    return MACEEnergyFn(H2O_Z, model="small", device="cpu")


def test_energy_is_scalar(mace_fn):
    E = mace_fn(H2O_R)
    assert E.shape == ()
    assert jnp.isfinite(E)


def test_forces_shape(mace_fn):
    F = mace_fn.forces(H2O_R)
    assert F.shape == (3, 3)


def test_forces_equal_negative_grad(mace_fn):
    """Forces must equal -jax.grad(energy)(R)."""
    F_direct = mace_fn.forces(H2O_R)
    F_grad = -jax.grad(mace_fn)(H2O_R)
    np.testing.assert_allclose(np.array(F_direct), np.array(F_grad), atol=1e-4)


def test_hessian_unsupported_by_design(mace_fn):
    """First-order VJPs only: jax.hessian = jacfwd(jacrev) needs forward-mode
    through the custom_vjp callback, which JAX cannot provide (the engine
    would have to supply Hessians -- same limitation as SimulatorAdapter).
    Locks the documented behavior; replaces aspirational hessian tests that
    predate the mace install and never ran."""
    import pytest
    with pytest.raises(TypeError, match="forward-mode"):
        jax.hessian(mace_fn)(H2O_R)


def test_jaxmd_energy_fn_wrapper(mace_fn):
    """jax_md_energy_fn must absorb extra kwargs without error."""
    wrapped = mace_fn.jax_md_energy_fn()
    E = wrapped(H2O_R, neighbor=None, some_extra_kwarg=True)
    assert jnp.isfinite(E)


def test_energy_is_jittable(mace_fn):
    jitted = jax.jit(mace_fn)
    E1 = jitted(H2O_R)
    E2 = jitted(H2O_R)   # second call uses cached XLA kernel
    np.testing.assert_allclose(float(E1), float(E2))
