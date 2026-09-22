"""
Differentiable Debye-Grüneisen vibrational free energy from engine E(V) scans.

Tier 3 of the free-energy physics axis: the 0 K enthalpic Ω
over-estimates demixing at finite T because it omits excess vibrational
entropy (size-mismatched alloys are vibrationally softer than their
end-members). This module adds F_vib(T) per phase from nothing but an
engine E(V) scan — cheap (≈9 single points, reusable from the volume
relaxation), fully differentiable, no phonon calculation.

Physics (Moruzzi, Janak & Schwarz, PRB 37, 790 (1988)):
    E(V) cubic fit  →  V₀, B₀, B′
    θ_D = s · 41.63 · sqrt(r₀[bohr]·B₀[kbar]/M[amu])  K     (MJS)
    γ   = (B′ − 1)/2   (Dugdale-MacDonald; Slater = B′/2 − 1/6)
    θ_D(V) = θ₀ (V₀/V)^γ
    F_vib(T) = (9/8)k θ_D + kT[3 ln(1 − e^{−θ_D/T}) − D₃(θ_D/T)]

D₃ is the third Debye function, evaluated by fixed 48-node Gauss-Legendre
quadrature — a pure JAX expression, so every output (θ_D, F_vib, Ω_vib(T),
and any phase boundary computed from them) is differentiable back to the
engine energies via jax.grad.

MJS quote ~10% typical θ_D error vs experiment for cubic metals; the scale
factor `s` (default 1.0) is available for element-wise calibration and is
itself differentiable — a legitimate fitting knob for the inverse capstone.
"""
from __future__ import annotations

import numpy as np

import jax.numpy as jnp

K_B = 8.617333e-5           # eV/K
_A3_TO_BOHR3 = 6.74833      # 1 Å³ in bohr³
_EVA3_TO_KBAR = 1602.176    # 1 eV/Å³ in kbar

# static quadrature nodes (host-side once; closed over as constants)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)
_GL_X = jnp.asarray(_GL_X)
_GL_W = jnp.asarray(_GL_W)

__all__ = [
    "eos_fit", "debye_temperature", "gruneisen", "debye_function_3",
    "f_vib_debye", "omega_vib_correction",
]


def eos_fit(volumes, energies):
    """Cubic-polynomial equation of state from an E(V) scan (per supercell).

    Differentiable in `energies` (least squares via normal equations).
    Returns (V0, E0, B0, Bprime): equilibrium volume [Å³], minimum energy
    [eV], bulk modulus [eV/Å³], and pressure derivative B′ [-], all for
    whatever unit of structure `volumes`/`energies` describe (pass per-atom
    or per-cell consistently).
    """
    V = jnp.asarray(volumes, dtype=jnp.result_type(float))
    E = jnp.asarray(energies, dtype=jnp.result_type(float))
    # center/scale V for conditioning
    Vm = jnp.mean(V)
    v = V / Vm - 1.0
    M = jnp.stack([jnp.ones_like(v), v, v**2, v**3], axis=1)
    coef = jnp.linalg.solve(M.T @ M, M.T @ E)      # a + b v + c v² + d v³
    a, b, c, d = coef

    # stationary point of the cubic closest to v=0 (physical minimum)
    disc = jnp.sqrt(jnp.maximum(c**2 - 3.0 * d * b, 0.0))
    v1 = (-c + disc) / (3.0 * d)
    v2 = (-c - disc) / (3.0 * d)
    curv1 = 2.0 * c + 6.0 * d * v1
    v0 = jnp.where(curv1 > 0.0, v1, v2)

    E0 = a + b * v0 + c * v0**2 + d * v0**3
    Epp = (2.0 * c + 6.0 * d * v0) / Vm**2          # d²E/dV² at V0
    Eppp = 6.0 * d / Vm**3
    V0 = Vm * (1.0 + v0)
    B0 = V0 * Epp                                    # eV/Å³
    Bprime = -1.0 - V0 * Eppp / Epp
    return V0, E0, B0, Bprime


def debye_temperature(V0_per_atom, B0, M_amu, scale=1.0):
    """MJS Debye temperature [K] from per-atom volume [Å³], bulk modulus
    [eV/Å³], and mean atomic mass [amu]."""
    r0_bohr = (3.0 * V0_per_atom * _A3_TO_BOHR3 / (4.0 * jnp.pi)) ** (1.0 / 3.0)
    B_kbar = B0 * _EVA3_TO_KBAR
    return scale * 41.63 * jnp.sqrt(r0_bohr * B_kbar / M_amu)


def gruneisen(Bprime, model: str = "dm"):
    """Grüneisen parameter from B′: Dugdale-MacDonald (default) or Slater."""
    if model == "dm":
        return (Bprime - 1.0) / 2.0
    if model == "slater":
        return Bprime / 2.0 - 1.0 / 6.0
    raise ValueError(model)


def debye_function_3(x):
    """D₃(x) = (3/x³)∫₀ˣ t³/(eᵗ−1) dt by 48-node Gauss-Legendre. x > 0."""
    x = jnp.asarray(x)
    t = 0.5 * x * (_GL_X + 1.0)                     # map [-1,1] → [0,x]
    integrand = t**3 / jnp.expm1(t)
    integral = 0.5 * x * jnp.sum(_GL_W * integrand)
    return 3.0 * integral / x**3


def f_vib_debye(T, theta_D):
    """Per-atom vibrational Helmholtz free energy [eV] in the Debye model
    (zero-point + thermal)."""
    x = theta_D / T
    return (9.0 / 8.0) * K_B * theta_D + K_B * T * (
        3.0 * jnp.log(-jnp.expm1(-x)) - debye_function_3(x))


def omega_vib_correction(T, eos_mix, eos_A, eos_B, x_B, M_A, M_B,
                         scale=1.0):
    """Vibrational correction to the interaction parameter at composition x:

        ΔΩ_vib(T) = [F_vib^mix(T) − (1−x)F_vib^A(T) − x F_vib^B(T)] / [x(1−x)]

    Each `eos_*` is (V0_per_atom, B0_eV_A3) for that phase — typically from
    `eos_fit` on engine E(V) scans (per-atom units). Mixture mass is the
    concentration average (Debye model with one mean oscillator mass — v1;
    per-species partial θ_D is a refinement). Add the result to the 0 K
    enthalpic Ω: Ω_eff(T) = Ω₀ + ΔΩ_vib(T).
    """
    M_mix = (1.0 - x_B) * M_A + x_B * M_B
    th_mix = debye_temperature(eos_mix[0], eos_mix[1], M_mix, scale)
    th_A = debye_temperature(eos_A[0], eos_A[1], M_A, scale)
    th_B = debye_temperature(eos_B[0], eos_B[1], M_B, scale)
    dF = (f_vib_debye(T, th_mix)
          - (1.0 - x_B) * f_vib_debye(T, th_A)
          - x_B * f_vib_debye(T, th_B))
    return dF / (x_B * (1.0 - x_B))
