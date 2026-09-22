"""
Differentiable mixing energetics from engine single points.

The engine → free-energy seam of the phase-diagram ladder: adapters produce
per-supercell energies (JAX scalars carrying VJPs into the engine), this
module turns them into mixing enthalpies, regular-solution Ω, and
arbitrary-order Redlich-Kister coefficient fits — all in JAX, so
∂(anything downstream)/∂(engine inputs) flows through.

Free-energy physics axis of the ladder:
    0 K enthalpy  →  mixing_enthalpy / omega_regular       (this module)
    subregular    →  redlich_kister_fit                    (this module)
    vibrational   →  core/thermo_vib.py (Debye-Grüneisen)
    sampled liquid→  window-mean estimator — forward-only + error bars
"""
from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "mixing_enthalpy", "omega_regular",
    "redlich_kister_fit", "redlich_kister_dH",
]


def mixing_enthalpy(E_mix, E_pure_A, E_pure_B, x_B, n_atoms_mix,
                    n_atoms_A, n_atoms_B):
    """Per-atom mixing enthalpy ΔH_mix = E_mix/N − (1−x)·E_A/N_A − x·E_B/N_B.

    All energies are total supercell energies (eV) — typically ASEAdapter
    outputs, so this is a traced JAX expression with engine VJPs attached.
    0 K enthalpic quantity: the ideal mixing entropy belongs in the free
    energy functional's kT·[c ln c + …] term, NOT here (see cg_pf_bridge
    docstring for why folding it in double-counts).
    """
    return (E_mix / n_atoms_mix
            - (1.0 - x_B) * E_pure_A / n_atoms_A
            - x_B * E_pure_B / n_atoms_B)


def omega_regular(dH_mix, x_B):
    """Regular-solution interaction parameter Ω = ΔH_mix / [x(1−x)] (eV/atom)."""
    return dH_mix / (x_B * (1.0 - x_B))


def redlich_kister_dH(L, x):
    """ΔH_mix(x) for Redlich-Kister coefficients L = (L₀, L₁, …):
    ΔH = x(1−x) · Σ_k L_k (1−2x)^k. `x` may be scalar or (n,)."""
    L = jnp.atleast_1d(L)
    k = jnp.arange(L.shape[0])
    if jnp.ndim(x) == 0:
        series = jnp.sum(L * (1.0 - 2.0 * x) ** k)
    else:
        x = jnp.asarray(x)
        series = jnp.sum(L[None, :] * (1.0 - 2.0 * x)[:, None] ** k[None, :],
                         axis=1)
    return x * (1.0 - x) * series


def redlich_kister_fit(xs, dHs, order: int = 1):
    """Least-squares Redlich-Kister coefficients from ΔH_mix samples.

    Fits ΔH(x_i) = x_i(1−x_i)·Σ_{k≤order} L_k (1−2x_i)^k by linear least
    squares directly on ΔH (not on ΔH/[x(1−x)], which amplifies endpoint
    noise). Differentiable in `dHs` — if those are adapter outputs, the
    fitted L_k (and every phase boundary computed from them) carry engine
    VJPs.

    Args:
        xs:    (n,) compositions in (0, 1).
        dHs:   (n,) per-atom mixing enthalpies (eV), traced or concrete.
        order: highest RK order k (order=0 → regular solution, 1 → subregular).

    Returns:
        L: (order+1,) coefficients (eV/atom).
    """
    xs = jnp.asarray(xs)
    dHs = jnp.asarray(dHs)
    n_coef = order + 1
    k = jnp.arange(n_coef)
    M = (xs * (1.0 - xs))[:, None] * (1.0 - 2.0 * xs)[:, None] ** k[None, :]
    # normal equations (n is tiny); differentiable and jit-safe
    A = M.T @ M
    b = M.T @ dHs
    return jnp.linalg.solve(A, b)
