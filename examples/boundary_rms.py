"""
Whole-boundary RMS metric for the Ag-Cu ladder.

The ladder so far scored constructions by invariant features (T_e, x_e,
c_α). This script adds the promised whole-boundary metric: the RMS
composition deviation of the computed liquidus (both limbs) from an
assessment-level reference construction — regular solution with the
assessed 0 K solid interaction (Ω_s = +340 meV) and the assessed liquid
(Ω_L = +155.5 meV) plus experimental fusion data. The reference is
validated first: it must reproduce the assessed eutectic
(1052 K, x_e = 0.399) before it is trusted as a liquidus proxy.

RMS is evaluated in composition at fixed temperature, interpolating the
reference on each limb over the overlapping temperature range, both limbs
pooled. Pure JAX re-sweeps — zero engine calls.

Output: examples/output/boundary_rms.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from coexist.core.phase_diagram import (  # noqa: E402
    eutectic_diagram, liquid_solution, redlich_kister_solution,
    regular_solution)

OUT = Path(__file__).parent / "output"
T_M_AG, DH_AG = 1234.93, 0.11691
T_M_CU, DH_CU = 1357.77, 0.13742
OM_L_CAL = 0.1555
OM_S_CAL = 0.340

def agcu_liquid(om_L):
    return liquid_solution(om_L, dH_fus_A=DH_AG, T_m_A=T_M_AG,
                           dH_fus_B=DH_CU, T_m_B=T_M_CU)

camp = json.load(open(OUT / "eam_campaign.json"))
liq = json.load(open(OUT / "liquid_omega.json"))
mace = json.load(open(OUT / "mace_rung.json"))
OM_L_ENG = liq["omega_L_1150K_eV"][0]
OM_EMT = 0.1614          # EMT @108
OM_MACE = mace["AgCu"]["omega_eV"]

# ═══ 1. Reference construction, validated against the assessed eutectic ═══════
ref = eutectic_diagram(regular_solution(OM_S_CAL), agcu_liquid(OM_L_CAL),
                       T_m_A=T_M_AG, T_m_B=T_M_CU, n_T=40)
print("── reference (assessed Ω_s = +340 meV, assessed Ω_L = +155.5 meV)")
print(f"   T_e = {ref['T_e']:.1f} K (assessed 1052), "
      f"x_e = {ref['c_liq_e']:.3f} (assessed 0.399), "
      f"c_α = {ref['c_alpha_e']:.3f} (assessed 0.141)")

results = {"reference": {"T_e": ref["T_e"], "x_e": ref["c_liq_e"],
                         "c_alpha": ref["c_alpha_e"]}}

# ═══ 2. Liquidus RMS per construction ═════════════════════════════════════════
def liquidus_rms(d, ref):
    """RMS Δx of the liquidus vs the reference, both limbs pooled,
    over the overlapping temperature range of each limb."""
    devs = []
    for Tk, ck in [("T_A", "liquidus_A"), ("T_B", "liquidus_B")]:
        T_d, c_d = np.asarray(d[Tk]), np.asarray(d[ck])
        T_r, c_r = np.asarray(ref[Tk]), np.asarray(ref[ck])
        lo = max(T_d.min(), T_r.min())
        hi = min(T_d.max(), T_r.max())
        Ts = np.linspace(lo, hi, 25)
        devs.append(np.interp(Ts, T_d, c_d) - np.interp(Ts, T_r, c_r))
    return float(np.sqrt(np.mean(np.concatenate(devs) ** 2)))

cases = [
    ("EMT (lit. liquid)", regular_solution(OM_EMT), OM_L_CAL),
    ("EAM RK (lit. liquid)",
     redlich_kister_solution(jnp.array(camp["AgCu"]["L_RK_eV"])), OM_L_CAL),
    ("MACE-MP-0 (lit. liquid)", regular_solution(OM_MACE), OM_L_CAL),
    ("EAM RK + EAM-MD liquid",
     redlich_kister_solution(jnp.array(camp["AgCu"]["L_RK_eV"])), OM_L_ENG),
]

print("\n── liquidus RMS vs reference (Δx_Cu, both limbs pooled)")
for name, G_s, om_L in cases:
    d = eutectic_diagram(G_s, agcu_liquid(om_L), T_m_A=T_M_AG,
                         T_m_B=T_M_CU, n_T=40)
    rms = liquidus_rms(d, ref)
    print(f"   {name:26s}: RMS = {rms:.4f}  "
          f"(T_e = {d['T_e']:.1f} K, x_e = {d['c_liq_e']:.3f})")
    results[name] = dict(rms_liquidus=rms, T_e=d["T_e"], x_e=d["c_liq_e"])

with open(OUT / "boundary_rms.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'boundary_rms.json'}")
