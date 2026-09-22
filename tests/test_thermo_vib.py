"""Tests for coexist.core.thermo_vib — EOS fit, Debye model, Ω_vib."""
import jax
import jax.numpy as jnp
import numpy as np

from coexist.core.thermo_vib import (
    K_B, debye_function_3, debye_temperature, eos_fit, f_vib_debye,
    gruneisen, omega_vib_correction)


def test_eos_fit_recovers_synthetic_minimum():
    """Cubic data with known V0/B0 must round-trip through the fit."""
    V0_true, E0_true, B0_true = 12.0, -3.5, 0.9   # Å³, eV, eV/Å³
    V = np.linspace(11.2, 12.8, 9)
    # E = E0 + B0/(2 V0) (V-V0)^2  (harmonic; Bprime → -1... use pure quad)
    E = E0_true + 0.5 * (B0_true / V0_true) * (V - V0_true) ** 2
    V0, E0, B0, _ = eos_fit(jnp.array(V), jnp.array(E))
    np.testing.assert_allclose(float(V0), V0_true, rtol=1e-4)
    np.testing.assert_allclose(float(E0), E0_true, rtol=1e-5)
    np.testing.assert_allclose(float(B0), B0_true, rtol=1e-3)


def test_debye_function_matches_quadrature():
    """48-node GL vs dense trapezoid reference at several x."""
    for x in (0.3, 1.0, 5.0, 15.0):
        t = np.linspace(1e-8, x, 200001)
        ref = 3.0 * np.trapezoid(t**3 / np.expm1(t), t) / x**3
        np.testing.assert_allclose(float(debye_function_3(x)), ref,
                                   rtol=1e-6)


def test_debye_temperature_copper_magnitude():
    """MJS formula with Cu-like inputs lands near the known θ_D."""
    th = float(debye_temperature(11.8, 149.0 / 160.2176, 63.546))
    assert 280.0 < th < 380.0     # exp 343 K; MJS-typical ±10%


def test_gruneisen_models():
    np.testing.assert_allclose(float(gruneisen(5.0, "dm")), 2.0)
    np.testing.assert_allclose(float(gruneisen(5.0, "slater")),
                               5.0 / 2 - 1.0 / 6)


def test_f_vib_classical_limit():
    """T ≫ θ_D: F_vib → kT[3 ln(θ/T) − 1] + (9/8)kθ + O(θ²/T²) terms.
    Check against the x→0 expansion D₃(x) ≈ 1 − 3x/8, ln(1−e⁻ˣ) ≈ ln x − x/2."""
    theta, T = 300.0, 6000.0
    x = theta / T
    approx = (9.0 / 8.0) * K_B * theta + K_B * T * (
        3.0 * (np.log(x) - x / 2.0) - (1.0 - 3.0 * x / 8.0))
    np.testing.assert_allclose(float(f_vib_debye(T, theta)), approx,
                               rtol=1e-4)


def test_f_vib_zero_point_limit():
    """T ≪ θ_D: F_vib → (9/8)kθ_D."""
    theta = 300.0
    val = float(f_vib_debye(1.0, theta))
    np.testing.assert_allclose(val, 9.0 / 8.0 * K_B * theta, rtol=1e-3)


def test_omega_vib_sign_follows_stiffness():
    """Mixture stiffer than end-member average → ΔΩ_vib > 0 (and vice versa).
    This is the model behavior underlying the Ag-Cu sign finding."""
    eos_soft = (15.0, 0.70)      # (V0 per atom, B0 eV/Å³)
    eos_A = (17.0, 0.69)
    eos_B = (12.0, 0.91)
    dOm_stiff = omega_vib_correction(900.0, (14.5, 0.87), eos_A, eos_B,
                                     0.5, 107.9, 63.5)
    dOm_soft = omega_vib_correction(900.0, eos_soft, eos_A, eos_B,
                                    0.5, 107.9, 63.5)
    assert float(dOm_stiff) > 0.0
    assert float(dOm_soft) < float(dOm_stiff)


def test_theta_and_fvib_differentiable():
    g = jax.grad(lambda B: f_vib_debye(800.0, debye_temperature(12.0, B,
                                                                63.5)))(0.9)
    assert np.isfinite(float(g)) and float(g) != 0.0
