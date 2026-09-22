"""Uncertainty on the Cu-Zn / Cu-Sn tangent-line gaps.

The missing-physics attributions of the Cu-Zn and Cu-Sn sections measure
corrections against an enthalpy-only tangent-line gap. This script asks how
well that reference gap itself is known, propagating the paper's stated
input uncertainties through the same tangent construction by one jax.grad:

  - +/-5% scale on each phase's mixing enthalpy (the campaigns' occupancy
    CV band, Section "Solid solutions");
  - +/-5 meV shift on each engine promotion energy (the residual
    finite-size scale; MACE-MP-0's inverted Zn fcc/hcp ordering, -9 meV,
    shows engine polymorph errors of at least this size).

Result: sigma_gap = 4.7 meV (Cu-Sn, 1071.15 K) and 7.6 meV (Cu-Zn, 1176 K),
so Cu-Sn's Debye-corrected remainder (3.1 meV) is 0.6 sigma from zero ---
unresolved --- while Cu-Zn's post-correction remainder (17.7 meV) stays
positive at 2.3 sigma.

Inputs: examples/output/{cusn,cuzn}_peritectic.json (stored engine
energetics). Output: examples/output/gap_uncertainty.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from coexist.core.phase_diagram import (  # noqa: E402
    K_B, common_tangent, liquid_solution)

OUT = Path(__file__).parent / "output"
CGRID = jnp.linspace(1e-4, 1.0 - 1e-4, 2001)
SIGMA_MIX_REL = 0.05      # occupancy CV band on mixing enthalpies
SIGMA_PROMO_EV = 0.005    # finite-size / engine-polymorph scale


def gap_sigma(dj, T, fus_A, fus_B, fcc_ref, bcc_ref, guesses):
    """Enthalpy-only bcc gap vs the fcc-liquid tangent, and its sigma."""
    L_fcc, L_bcc = jnp.array(dj["L_fcc_eV"]), jnp.array(dj["L_bcc_eV"])
    G_L = liquid_solution(0.0, dH_fus_A=fus_A[0], T_m_A=fus_A[1],
                          dH_fus_B=fus_B[0], T_m_B=fus_B[1])
    rk = lambda L, c: sum(L[k] * (1.0 - 2.0 * c) ** k          # noqa: E731
                          for k in range(len(L)))

    def gap(theta):
        eps_f, eps_b, dp = theta[0], theta[1], theta[2:]

        def G_fcc(c, T_):
            return (fcc_ref(c, dp) + (1 + eps_f) * c * (1 - c) * rk(L_fcc, c)
                    + K_B * T_ * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))

        def G_bcc(c, T_):
            return (bcc_ref(c, dp) + (1 + eps_b) * c * (1 - c) * rk(L_bcc, c)
                    + K_B * T_ * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))

        cf, cl = common_tangent(G_fcc, G_L, T,
                                c_alpha_guess=guesses[0],
                                c_beta_guess=guesses[1])
        s = (G_L(cl, T) - G_fcc(cf, T)) / (cl - cf)
        line = G_fcc(cf, T) + s * (CGRID - cf)
        return jnp.min(G_bcc(CGRID, T) - line)

    th0 = jnp.zeros(5)
    g0 = float(gap(th0))
    grad = jax.grad(gap)(th0)
    sig = jnp.array([SIGMA_MIX_REL, SIGMA_MIX_REL,
                     SIGMA_PROMO_EV, SIGMA_PROMO_EV, SIGMA_PROMO_EV])
    return g0, float(jnp.sqrt(jnp.sum((grad * sig) ** 2))), np.asarray(grad)


results = {}

# ── Cu-Sn at the assessed 1071.15 K ──────────────────────────────────────────
ds = json.load(open(OUT / "cusn_peritectic.json"))
dSf, dCb, dSb = (ds["dE_Sn_fcc_minus_betaSn_eV"],
                 ds["dE_Cu_bcc_minus_fcc_eV"],
                 ds["dE_Sn_bcc_minus_betaSn_eV"])
g0, s0, grad = gap_sigma(
    ds, 1071.15, (0.13742, 1357.77), (0.07286, 505.08),
    lambda c, dp: c * (dSf + dp[0]),
    lambda c, dp: (1 - c) * (dCb + dp[1]) + c * (dSb + dp[2]),
    guesses=(0.05, 0.5))
DEBYE_CUSN = -18.631317009056513   # cusn_beta_gap.json
rem = g0 + DEBYE_CUSN * 1e-3
results["CuSn"] = {
    "T_K": 1071.15, "gap0_meV": g0 * 1e3, "sigma_gap_meV": s0 * 1e3,
    "dgap_dtheta": grad.tolist(),
    "post_debye_remainder_meV": rem * 1e3,
    "remainder_over_sigma": rem / s0,
}
print(f"Cu-Sn: gap = {g0*1e3:.1f} ± {s0*1e3:.1f} meV; post-Debye "
      f"remainder {rem*1e3:.1f} meV = {rem/s0:.1f} sigma  -> unresolved")

# ── Cu-Zn at the assessed 1176 K ─────────────────────────────────────────────
dz = json.load(open(OUT / "cuzn_peritectic.json"))
dZf, dCb2, dZb = (dz["dE_Zn_fcc_minus_hcp_eV"],
                  dz["dE_Cu_bcc_minus_fcc_eV"],
                  dz["dE_Zn_bcc_minus_hcp_eV"])
g0z, s0z, gradz = gap_sigma(
    dz, 1176.0, (0.13742, 1357.77), (0.07588, 692.68),
    lambda c, dp: c * (dZf + dp[0]),
    lambda c, dp: (1 - c) * (dCb2 + dp[1]) + c * (dZb + dp[2]),
    guesses=(0.78, 0.98))
REMAINDER_CUZN = 17.653721350151528e-3   # cuzn_quasichemical.json, both corr.
results["CuZn"] = {
    "T_K": 1176.0, "gap0_meV": g0z * 1e3, "sigma_gap_meV": s0z * 1e3,
    "dgap_dtheta": gradz.tolist(),
    "post_corrections_remainder_meV": REMAINDER_CUZN * 1e3,
    "remainder_over_sigma": REMAINDER_CUZN / s0z,
}
print(f"Cu-Zn: gap = {g0z*1e3:.1f} ± {s0z*1e3:.1f} meV; post-correction "
      f"remainder {REMAINDER_CUZN*1e3:.1f} meV = {REMAINDER_CUZN/s0z:.1f} "
      f"sigma  -> stays positive")

# ── Verdict robustness against the assumed promotion-energy scale ────────────
# sigma_gap^2 = var_mix + q * sigma_promo^2 with var_mix from the (fixed)
# +/-5% occupancy band and q = sum of squared promotion-energy sensitivities,
# so each verdict maps to a threshold in sigma_promo alone.


def promo_breakdown(grad, rem):
    grad = np.asarray(grad)
    var_mix = float(np.sum((grad[:2] * SIGMA_MIX_REL) ** 2))
    q = float(np.sum(grad[2:] ** 2))

    def sigma_promo_at(k):        # sigma_promo where |rem| = k * sigma_gap
        val = (rem / k) ** 2 - var_mix
        return float(np.sqrt(val / q)) if val > 0 else 0.0

    def ratio_at(sp):             # rem / sigma_gap at sigma_promo = sp
        return abs(rem) / np.sqrt(var_mix + q * sp**2)

    return {
        "promo_variance_share_at_5meV":
            q * SIGMA_PROMO_EV**2 / (var_mix + q * SIGMA_PROMO_EV**2),
        "sigma_promo_meV_where_remainder_is_1sigma": sigma_promo_at(1) * 1e3,
        "sigma_promo_meV_where_remainder_is_2sigma": sigma_promo_at(2) * 1e3,
        "remainder_sigmas_at_10meV_promo": float(ratio_at(0.010)),
    }


results["CuSn"]["robustness"] = promo_breakdown(grad, rem)
results["CuZn"]["robustness"] = promo_breakdown(gradz, REMAINDER_CUZN)
for sys_ in ("CuSn", "CuZn"):
    r = results[sys_]["robustness"]
    print(f"{sys_}: promo share of var(gap) at ±5 meV = "
          f"{r['promo_variance_share_at_5meV']:.1%}; remainder crosses "
          f"1σ at σ_promo = {r['sigma_promo_meV_where_remainder_is_1sigma']:.2f} meV, "
          f"2σ at {r['sigma_promo_meV_where_remainder_is_2sigma']:.2f} meV; "
          f"{r['remainder_sigmas_at_10meV_promo']:.2f}σ at ±10 meV")

results["recipe"] = ("+/-5% per-phase mixing-enthalpy scale (occupancy CV) "
                     "and +/-5 meV per promotion energy (finite-size / "
                     "engine-polymorph scale), one jax.grad through the "
                     "tangent construction")
with open(OUT / "gap_uncertainty.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'gap_uncertainty.json'}")
