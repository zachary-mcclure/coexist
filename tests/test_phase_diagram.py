"""Tests for coexist.core.phase_diagram — tangents, eutectic, implicit grads."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from coexist.core.phase_diagram import (
    K_B,
    common_tangent,
    eutectic_point,
    liquid_solution,
    regular_solution,
    subregular_solution,
    tangent_residual,
)

OMEGA = 0.10   # eV — T_c = Ω/2k ≈ 580 K


# ── symmetric case: must reproduce the analytic structure ─────────────────────

def test_symmetric_tangent_is_symmetric():
    G = regular_solution(OMEGA)
    T = 0.7 * OMEGA / (2 * K_B)
    ca, cb = common_tangent(G, G, T)
    assert 0 < float(ca) < 0.5 < float(cb) < 1
    np.testing.assert_allclose(float(ca) + float(cb), 1.0, atol=1e-5)


def test_symmetric_tangent_matches_scalar_root():
    """For G_α = G_β the tangent is horizontal: Ω(1−2c) + kT·ln(c/(1−c)) = 0."""
    G = regular_solution(OMEGA)
    T = 0.6 * OMEGA / (2 * K_B)
    ca, _ = common_tangent(G, G, T)
    resid = OMEGA * (1 - 2 * float(ca)) + K_B * T * np.log(float(ca) / (1 - float(ca)))
    assert abs(resid) < 1e-6


def test_tangent_residuals_vanish():
    G = regular_solution(OMEGA)
    T = 0.75 * OMEGA / (2 * K_B)
    ca, cb = common_tangent(G, G, T)
    assert float(tangent_residual(G, G, ca, cb, T)) < 1e-6


def test_above_Tc_no_gap():
    """Above T_c the Newton solve collapses both points toward c = 0.5."""
    G = regular_solution(OMEGA)
    T = 1.2 * OMEGA / (2 * K_B)
    ca, cb = common_tangent(G, G, T)
    assert abs(float(cb) - float(ca)) < 0.05


# ── asymmetric (subregular) case ──────────────────────────────────────────────

def test_subregular_tangent_asymmetric():
    G = subregular_solution(OMEGA, 0.03)
    T = 0.6 * OMEGA / (2 * K_B)
    ca, cb = common_tangent(G, G, T)
    # residuals vanish but the gap is NOT symmetric about 0.5
    assert float(tangent_residual(G, G, ca, cb, T)) < 1e-6
    assert abs((float(ca) + float(cb)) - 1.0) > 0.01


def test_two_different_curves():
    """Solid vs liquid at T between the melting points: solidus < liquidus."""
    Gs = regular_solution(0.02)
    Gl = liquid_solution(0.0, dH_fus_A=0.14, T_m_A=1358.0,
                         dH_fus_B=0.12, T_m_B=1235.0)
    T = 1300.0  # between T_m_B and T_m_A: A-rich solid + B-enriched liquid
    cs, cl = common_tangent(Gs, Gl, T, c_alpha_guess=0.05, c_beta_guess=0.4)
    assert float(tangent_residual(Gs, Gl, cs, cl, T)) < 1e-6
    assert float(cs) < float(cl)   # liquid enriched in the low-melting component


# ── implicit gradients through the tangent ────────────────────────────────────

def test_dcalpha_domega_implicit_vs_fd():
    T = 400.0

    def c_alpha_of_omega(om):
        G = regular_solution(om)
        ca, _ = common_tangent(G, G, T)
        return ca

    om0 = jnp.array(OMEGA)
    ad = float(jax.grad(c_alpha_of_omega)(om0))
    h = 1e-5
    fd = float((c_alpha_of_omega(om0 + h) - c_alpha_of_omega(om0 - h)) / (2 * h))
    assert abs(ad - fd) / abs(fd) < 1e-3


# ── eutectic ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def agcu_like():
    """Ag-Cu-like parameterization (solid gap persists above the liquid dip)."""
    Gs = regular_solution(0.22)
    Gl = liquid_solution(0.15, dH_fus_A=0.1374, T_m_A=1357.8,
                         dH_fus_B=0.1169, T_m_B=1234.9)
    return Gs, Gl


def test_eutectic_geometry(agcu_like):
    Gs, Gl = agcu_like
    ca, cl, cb, Te = eutectic_point(Gs, Gl)
    ca, cl, cb, Te = map(float, (ca, cl, cb, Te))
    assert 0 < ca < cl < cb < 1          # liquid between the two solid limbs
    assert 600.0 < Te < min(1357.8, 1234.9)   # below both melting points


def test_eutectic_tangency_residuals(agcu_like):
    Gs, Gl = agcu_like
    ca, cl, cb, Te = eutectic_point(Gs, Gl)
    mu = jax.grad(Gs, argnums=0)(ca, Te)
    r1 = mu - jax.grad(Gl, argnums=0)(cl, Te)
    r2 = jax.grad(Gs, argnums=0)(cb, Te) - jax.grad(Gl, argnums=0)(cl, Te)
    r3 = Gl(cl, Te) - Gs(ca, Te) - mu * (cl - ca)
    r4 = Gs(cb, Te) - Gs(ca, Te) - mu * (cb - ca)
    for r in (r1, r2, r3, r4):
        assert abs(float(r)) < 1e-5


def test_dTe_domega_solid_implicit_vs_fd(agcu_like):
    _, Gl = agcu_like

    def Te_of_omega(om):
        Gs = regular_solution(om)
        return eutectic_point(Gs, Gl)[3]

    om0 = jnp.array(0.22)
    ad = float(jax.grad(Te_of_omega)(om0))
    h = 1e-4
    fd = float((Te_of_omega(om0 + h) - Te_of_omega(om0 - h)) / (2 * h))
    assert abs(ad - fd) / abs(fd) < 1e-3
    assert ad < 0   # stronger solid repulsion → deeper eutectic
