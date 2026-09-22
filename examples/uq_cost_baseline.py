"""
Cost baseline for the whole-boundary UQ claim (Sec. 4.3 / fig:engine_uq).

liquid_omega.py gets a +-1 sigma band on the ENTIRE Ag-Cu liquidus (30
T-points) from ONE jax.jacobian call through tangent_sweep, reusing the
single already-run liquid-sampling MD campaign that estimated Omega_L and
its sigma. Sec. 4.1 already quotes a single-point AD-vs-FD number (1 engine
call vs. 648 central differences for dT_e/dR); the whole-boundary UQ claim
has no analogous control. This script supplies two, at the two levels
where "cost" actually means something different:

  (i) Newton-solve level (holding the Omega_L point estimate + sigma
      fixed): time the one jax.jacobian(tangent_sweep) call against a
      naive per-T-point central-difference alternative that does not use
      vmap+AD. Both are cheap in absolute terms (the Newton solve itself
      is a handful of 2x2 linear solves) -- the point is the SCALING,
      not that either is slow.

  (ii) Engine/MD-campaign level (the resource that actually costs
      minutes): Omega_L's sigma comes from finite MD sampling (3 seeds,
      block-averaged). A forward-only workflow that wanted a comparable
      *whole-boundary* uncertainty band without differentiating through
      the construction would have to re-run the liquid-sampling campaign
      itself K times (bootstrap / Monte Carlo over the MD estimate) and
      re-solve the full boundary each time, rather than reusing one
      point estimate + one cheap Jacobian call.

Pure JAX + host timing; zero engine calls. Output: examples/output/uq_cost_baseline.json
"""
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from coexist.core.mixing import redlich_kister_fit  # noqa: E402
from coexist.core.phase_diagram import (  # noqa: E402
    common_tangent, eutectic_diagram, liquid_solution,
    redlich_kister_solution, tangent_sweep)

OUT = Path(__file__).parent / "output"
camp = json.load(open(OUT / "eam_campaign.json"))
liq = json.load(open(OUT / "liquid_omega.json"))

T_M_AG, DH_AG = 1234.93, 0.11691
T_M_CU, DH_CU = 1357.77, 0.13742

L_RK = jnp.array(camp["AgCu"]["L_RK_eV"])
G_s = redlich_kister_solution(L_RK)
OM_L = liq["omega_L_1150K_eV"][0]
OM_ERR = liq["omega_L_1150K_eV"][1]
N_T = 30

diag = eutectic_diagram(
    G_s, liquid_solution(OM_L, dH_fus_A=DH_AG, T_m_A=T_M_AG,
                         dH_fus_B=DH_CU, T_m_B=T_M_CU),
    T_m_A=T_M_AG, T_m_B=T_M_CU, n_T=N_T)
Ts = jnp.array(diag["T_B"])
seed_l = jnp.array(diag["liquidus_B"])
seed_s = jnp.array(diag["solidus_B"])


def liquidus_of_omL(om_L):
    G_L = liquid_solution(om_L, dH_fus_A=DH_AG, T_m_A=T_M_AG,
                          dH_fus_B=DH_CU, T_m_B=T_M_CU)
    cl_, cs_ = tangent_sweep(G_L, G_s, Ts, seed_l, seed_s)
    return cl_


# ─── (i) AD: one jax.jacobian call through the vmapped sweep ──────────────────
f_ad = jax.jit(jax.jacobian(liquidus_of_omL))
f_ad(OM_L)  # warm up / trace
t0 = time.perf_counter()
for _ in range(20):
    J_ad = f_ad(OM_L)
t_ad = (time.perf_counter() - t0) / 20
print(f"AD:  jax.jacobian(tangent_sweep) over {N_T} T-points: "
      f"{t_ad*1e3:.2f} ms/call (jitted, mean of 20)")

# ─── (i) FD: naive per-T-point central difference, no vmap, no AD ─────────────
# One Newton solve at a time (as a from-scratch implementer without AD/vmap
# would write it): 2 solves (om +/- h) per T-point, N_T points.
h = 1e-4


def per_point_fd():
    out = np.zeros(N_T)
    eps = 1e-4
    for i in range(N_T):
        T = float(Ts[i])
        gl = float(np.clip(float(seed_l[i]), eps, 1 - eps))
        gs = float(np.clip(float(seed_s[i]), eps, 1 - eps))
        cl_p, _ = common_tangent(liquid_solution(OM_L + h, dH_fus_A=DH_AG,
                                                  T_m_A=T_M_AG, dH_fus_B=DH_CU,
                                                  T_m_B=T_M_CU), G_s, T,
                                 c_alpha_guess=gl, c_beta_guess=gs)
        cl_m, _ = common_tangent(liquid_solution(OM_L - h, dH_fus_A=DH_AG,
                                                  T_m_A=T_M_AG, dH_fus_B=DH_CU,
                                                  T_m_B=T_M_CU), G_s, T,
                                 c_alpha_guess=gl, c_beta_guess=gs)
        out[i] = (float(cl_p) - float(cl_m)) / (2 * h)
    return out


t0 = time.perf_counter()
J_fd = per_point_fd()
t_fd = time.perf_counter() - t0
n_solves_fd = 2 * N_T
print(f"FD:  {n_solves_fd} un-vmapped Newton solves (2 per T-point, "
      f"serial): {t_fd*1e3:.1f} ms total")
print(f"     agreement AD vs FD: max|diff| = "
      f"{np.abs(np.asarray(J_ad) - J_fd).max():.2e}")
print(f"     wall-clock ratio FD/AD = {t_fd/t_ad:.0f}x "
      f"(at fixed Omega_L point estimate + sigma; both cheap in "
      f"absolute terms -- the gap is in what scales with N_T and with "
      f"the number of parameters, not in either being slow here)")

# ─── (ii) The resource that actually costs minutes: the MD campaign itself ────
# Run command: `python examples/liquid_omega.py` ~15 min (MTK, 30 ps
# production x 3 seeds, at the single temperature 1150 K used for the
# reported Omega_L +/- sigma).
MD_CAMPAIGN_MIN = 15.0
# A forward-only whole-boundary error band of comparable statistical
# quality (resolving the band shape, not just one +/-1 sigma perturbation)
# needs enough independent bootstrap draws of Omega_L to trace the
# nonlinearity in a Monte-Carlo sense; 30 draws is a conservative round
# number for a defensible band (same order as the T-grid itself).
K_BOOTSTRAP = 30
naive_total_min = K_BOOTSTRAP * MD_CAMPAIGN_MIN
print(f"\nEngine-call level: one liquid_omega.py campaign is "
      f"~{MD_CAMPAIGN_MIN:.0f} min (already paid, once). A forward-only "
      f"Monte-Carlo band of comparable quality needs ~{K_BOOTSTRAP} "
      f"independent re-runs of that campaign -- "
      f"~{naive_total_min/60:.1f} h of MD -- against the ONE campaign "
      f"already spent plus one {t_ad*1e3:.0f} ms Jacobian call here.")

results = dict(
    n_T=N_T, t_ad_ms=t_ad * 1e3, t_fd_ms=t_fd * 1e3,
    n_solves_fd=n_solves_fd, ratio_fd_over_ad=t_fd / t_ad,
    max_abs_diff_ad_fd=float(np.abs(np.asarray(J_ad) - J_fd).max()),
    md_campaign_minutes=MD_CAMPAIGN_MIN, k_bootstrap=K_BOOTSTRAP,
    naive_bootstrap_hours=naive_total_min / 60,
)
with open(OUT / "uq_cost_baseline.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'uq_cost_baseline.json'}")
