"""
Liquid Ω_L from MD sampling + boundary UQ (rung 14):
the LAST literature interaction parameter leaves the Ag-Cu forward chain.

Before this rung the Ag-Cu eutectic used Ω_L = +155.5 meV from the CALPHAD
literature. Here Ω_L is sampled from Williams-EAM liquid MD at (T, P≈0)
— melt → NPT → window-mean production, seeds for error bars — so the only
non-engine inputs left in the forward diagram are the pure-element fusion
data (ΔH_m, T_m). Then, because every boundary is differentiable:

  σ(Ω_L)  →  σ(T_e) = |∂T_e/∂Ω_L|·σ        (delta method, one jax.grad)
  σ(Ω_L)  →  a ±1σ band on the ENTIRE liquidus in one vmapped
             jacobian pass (tangent_sweep)

Sampling at two temperatures gives the Ω_L(T) slope — the engine's version
of CALPHAD's linear T-term in the liquid L0.

float64. Runtime ~4 min (18 MD trajectories × ~12 s).
Output: examples/output/liquid_omega.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from coexist.adapters.lammps_factory import (  # noqa: E402
    eam_alloy_factory, fcc_solution, sample_liquid_energy)
from coexist.core.phase_diagram import (  # noqa: E402
    eutectic_diagram, eutectic_point, liquid_solution,
    redlich_kister_solution, tangent_sweep)

OUT = Path(__file__).parent / "output"
T_M_AG, DH_AG = 1234.93, 0.11691
T_M_CU, DH_CU = 1357.77, 0.13742
SEEDS = (0, 1, 2)
results = {}

fac = eam_alloy_factory("CuAg.eam.alloy", ("Cu", "Ag"))

# ═══ 1. Sample liquid energies (108 atoms, melt→NPT→production) ═══════════════
def omega_L_at(T):
    E = {}
    for tag, elA, elB, x in [("Ag", "Ag", "Cu", 0.0), ("Cu", "Cu", "Ag", 0.0),
                             ("mix", "Ag", "Cu", 0.5)]:
        E[tag] = [sample_liquid_energy(
            fcc_solution(elA, elB, x, reps=(3, 3, 3), seed=s), fac, T_K=T,
            prod_ps=30.0, seed=s) for s in SEEDS]
    dH = np.array([E["mix"][i]["E_mean"] - 0.5 * E["Ag"][i]["E_mean"]
                   - 0.5 * E["Cu"][i]["E_mean"] for i in range(len(SEEDS))])
    sem_within = np.sqrt(np.mean([E["mix"][i]["E_sem"]**2
                                  + 0.25 * E["Ag"][i]["E_sem"]**2
                                  + 0.25 * E["Cu"][i]["E_sem"]**2
                                  for i in range(len(SEEDS))]))
    om = dH.mean() / 0.25
    om_err = max(dH.std(ddof=1) / np.sqrt(len(SEEDS)), sem_within) / 0.25
    T_check = np.mean([e["T_mean"] for e in E["mix"]])
    return om, om_err, T_check

print("── liquid Ω_L from EAM MD (108 atoms, NPT, window-mean, 3 seeds)")
om_1400, err_1400, Tc1 = omega_L_at(1400.0)
om_1150, err_1150, Tc2 = omega_L_at(1150.0)
print(f"   Ω_L(1400 K) = {om_1400*1e3:+.1f} ± {err_1400*1e3:.1f} meV "
      f"(⟨T⟩={Tc1:.0f} K)")
print(f"   Ω_L(1150 K) = {om_1150*1e3:+.1f} ± {err_1150*1e3:.1f} meV "
      f"(⟨T⟩={Tc2:.0f} K)   [CALPHAD lit: +155.5 at low T]")
slope = (om_1400 - om_1150) / 250.0
print(f"   Ω_L(T) slope ≈ {slope*1e6:+.1f} μeV/K "
      f"(CALPHAD Ag-Cu liquid L0 has ≈ −16 μeV/K)")

# use the near-eutectic sample for the diagram
OM_L, OM_ERR = om_1150, err_1150

# ═══ 2. Fully-engine eutectic + delta-method σ(T_e) ═══════════════════════════
camp = json.load(open(OUT / "eam_campaign.json"))
L_RK = jnp.array(camp["AgCu"]["L_RK_eV"])
G_s = redlich_kister_solution(L_RK)

def eut(om_L):
    G_L = liquid_solution(om_L, dH_fus_A=DH_AG, T_m_A=T_M_AG,
                          dH_fus_B=DH_CU, T_m_B=T_M_CU)
    return eutectic_point(G_s, G_L)

ca, cl, cb, Te = eut(OM_L)
g = jax.jacobian(lambda o: jnp.stack(eut(o)))(OM_L)
sTe = abs(float(g[3])) * OM_ERR
sxe = abs(float(g[1])) * OM_ERR
print(f"\n── Ag-Cu eutectic, all interactions from the engine "
      f"(exp: 1052 K, x_e 0.399, c_α 0.141)")
print(f"   T_e = {float(Te):.1f} ± {sTe:.1f} K   x_e = {float(cl):.3f} ± "
      f"{sxe:.3f}   c_α = {float(ca):.3f}")
print(f"   (CALPHAD-liquid version gave 1060.3 K / 0.325 / 0.037; "
      f"remaining non-engine inputs: fusion data only)")

# ═══ 3. Rung 14: ±1σ band on the whole liquidus in ONE jacobian pass ══════════
diag = eutectic_diagram(G_s, liquid_solution(
    OM_L, dH_fus_A=DH_AG, T_m_A=T_M_AG, dH_fus_B=DH_CU, T_m_B=T_M_CU),
    T_m_A=T_M_AG, T_m_B=T_M_CU, n_T=30)
Ts = jnp.array(diag["T_B"])                    # Cu-side limb, T_e → T_m(Cu)
seed_l = jnp.array(diag["liquidus_B"])
seed_s = jnp.array(diag["solidus_B"])

def liquidus_of_omL(om_L):
    G_L = liquid_solution(om_L, dH_fus_A=DH_AG, T_m_A=T_M_AG,
                          dH_fus_B=DH_CU, T_m_B=T_M_CU)
    cl_, cs_ = tangent_sweep(G_L, G_s, Ts, seed_l, seed_s)
    return cl_

J = jax.jacobian(liquidus_of_omL)(OM_L)        # ∂(entire liquidus)/∂Ω_L
band = np.abs(np.asarray(J)) * OM_ERR
print(f"\n── liquidus UQ band (vmapped jacobian, one pass over "
      f"{len(np.asarray(Ts))} T-points)")
print(f"   max |∂c_liq/∂Ω_L| = {np.abs(np.asarray(J)).max():.3f} /eV → "
      f"band up to ±{band.max():.4f} in x_Cu; "
      f"tightest near T_m(Cu) (±{band[-1]:.4f})")

results.update({
    "omega_L_1400K_eV": [om_1400, err_1400],
    "omega_L_1150K_eV": [om_1150, err_1150],
    "omega_L_slope_eV_per_K": slope,
    "T_e_K": float(Te), "T_e_sigma_K": sTe,
    "x_e": float(cl), "x_e_sigma": sxe, "c_alpha": float(ca),
    "dTe_dOmL_K_per_eV": float(g[3]),
    "liquidus_T": np.asarray(Ts).tolist(),
    "liquidus_c": np.asarray(liquidus_of_omL(OM_L)).tolist(),
    "liquidus_band": band.tolist(),
})
with open(OUT / "liquid_omega.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'liquid_omega.json'}")
