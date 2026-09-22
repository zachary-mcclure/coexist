"""Tests for coexist.core.mixing — ΔH_mix, Ω, Redlich-Kister fits."""
import jax
import jax.numpy as jnp
import numpy as np

from coexist.core.mixing import (
    mixing_enthalpy, omega_regular, redlich_kister_dH, redlich_kister_fit)
from coexist.core.phase_diagram import (
    K_B, redlich_kister_solution, regular_solution, subregular_solution)


def test_mixing_enthalpy_pure_limits():
    """At x=0 with E_mix = E_A the mixing enthalpy vanishes."""
    dH = mixing_enthalpy(-100.0, -100.0, -80.0, 0.0, 32, 32, 32)
    assert abs(float(dH)) < 1e-12


def test_omega_regular_roundtrip():
    om = 0.35
    x = 0.25
    dH = om * x * (1 - x)
    np.testing.assert_allclose(float(omega_regular(dH, x)), om, rtol=1e-12)


def test_rk_fit_recovers_known_coefficients():
    L_true = jnp.array([0.30, -0.05, 0.02])
    xs = jnp.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
    dHs = redlich_kister_dH(L_true, xs)
    L_fit = redlich_kister_fit(xs, dHs, order=2)
    np.testing.assert_allclose(np.asarray(L_fit), np.asarray(L_true),
                               atol=1e-6)


def test_rk_fit_order0_is_weighted_omega():
    """Order-0 fit of exactly regular data recovers Ω at any sampling."""
    xs = jnp.array([0.2, 0.4, 0.6, 0.8])
    dHs = 0.25 * xs * (1 - xs)
    L = redlich_kister_fit(xs, dHs, order=0)
    np.testing.assert_allclose(float(L[0]), 0.25, rtol=1e-6)


def test_rk_fit_is_differentiable_in_dH():
    """∂L₀/∂ΔH_i exists and is nonzero — the engine-VJP path."""
    xs = jnp.array([0.25, 0.5, 0.75])
    dHs = jnp.array([0.05, 0.08, 0.06])

    def L0(d):
        return redlich_kister_fit(xs, d, order=1)[0]

    g = jax.grad(L0)(dHs)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.abs(g).max()) > 0.1


def test_rk_solution_matches_regular_and_subregular():
    c, T = 0.3, 800.0
    G_rk1 = redlich_kister_solution(jnp.array([0.2]))
    G_reg = regular_solution(0.2)
    np.testing.assert_allclose(float(G_rk1(c, T)), float(G_reg(c, T)),
                               rtol=1e-6)
    G_rk2 = redlich_kister_solution(jnp.array([0.2, -0.04]))
    G_sub = subregular_solution(0.2, -0.04)
    np.testing.assert_allclose(float(G_rk2(c, T)), float(G_sub(c, T)),
                               rtol=1e-6)
