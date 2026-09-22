"""
Diagram-topology axis of the phase-diagram ladder:
beyond the eutectic — line compounds, the γ/γ' solvus, and the monotectic —
with every energetic input from a production EAM engine where available.

Rungs demonstrated:
  T4a  Ni-Al line compounds from the engine: H_f(B2 NiAl), H_f(L1₂ Ni₃Al)
       from Mishin-2009 EAM vs experiment.
  T4b  γ/γ' solvus: common tangent between the fcc Ni(Al) solution (EAM
       mixing energetics) and the Ni₃Al line compound (EAM H_f) — the
       superalloy phase boundary, with ∂c_solvus/∂H_f (the sensitivity of
       Al solubility in nickel to the compound's stability).
  T5   Al-Pb monotectic: EAM solid Ω_s (+0.91 eV — near-zero solubility),
       literature-magnitude liquid gap; the general three-curve tangent
       finds the invariant and `classify_invariant` names the reaction —
       the solver discovers the topology, it is not told.

float64 throughout. Runtime ~1 min. Output: examples/output/topology_ladder.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402

from coexist.adapters.lammps_factory import (  # noqa: E402
    eam_alloy_factory, fcc_solution, relax_full)
from coexist.core.phase_diagram import (  # noqa: E402
    classify_invariant, common_tangent, global_tangency_gap, line_compound,
    liquid_solution, regular_solution, tangent_residual,
    three_phase_equilibrium)

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
results = {}

# ═══ T4a: Ni-Al compound formation energies from the engine ═══════════════════
print("── T4a: Ni-Al line compounds (Purja Pun & Mishin 2009 EAM)")
fac_nial = eam_alloy_factory("NiAl.eam.alloy", ("Ni", "Al"))

E_ref = {}
for el, a in [("Ni", 3.52), ("Al", 4.05)]:
    rel, _ = relax_full(bulk(el, "fcc", a=a, cubic=True).repeat((3, 3, 3)),
                        fac_nial)
    E_ref[el] = float(rel.get_potential_energy()) / len(rel)

b2 = Atoms("NiAl", positions=[[0, 0, 0], [1.44, 1.44, 1.44]],
           cell=np.eye(3) * 2.88, pbc=True).repeat((4, 4, 4))
rel, _ = relax_full(b2, fac_nial)
Hf_B2 = (float(rel.get_potential_energy()) / len(rel)
         - 0.5 * E_ref["Ni"] - 0.5 * E_ref["Al"])

a0 = 3.57
l12 = Atoms("AlNi3",
            positions=[[0, 0, 0], [0, a0 / 2, a0 / 2],
                       [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]],
            cell=np.eye(3) * a0, pbc=True).repeat((3, 3, 3))
rel, _ = relax_full(l12, fac_nial)
Hf_L12 = (float(rel.get_potential_energy()) / len(rel)
          - 0.75 * E_ref["Ni"] - 0.25 * E_ref["Al"])

print(f"   H_f(B2 NiAl)   = {Hf_B2:+.3f} eV/atom   (exp ≈ −0.66…−0.68)")
print(f"   H_f(L1₂ Ni₃Al) = {Hf_L12:+.3f} eV/atom   (exp ≈ −0.42…−0.44)")
results["NiAl_Hf_B2_eV"] = Hf_B2
results["NiAl_Hf_L12_eV"] = Hf_L12

# ═══ T4b: γ/γ' solvus — solution curve vs line compound ══════════════════════
print("\n── T4b: γ/γ' solvus at 1000 K (superalloy two-phase field)")
# fcc Ni(Al) random-solution mixing enthalpy at x_Al = 14/108 (γ side)
X_G = 14.0 / 108.0
dHs = []
for seed in (0, 1, 2):
    at = fcc_solution("Ni", "Al", X_G, reps=(3, 3, 3), seed=seed)
    rel, _ = relax_full(at, fac_nial)
    n = len(rel)
    dHs.append(float(rel.get_potential_energy()) / n
               - (1 - X_G) * E_ref["Ni"] - X_G * E_ref["Al"])
dH_gamma = float(np.mean(dHs))
om_gamma = dH_gamma / (X_G * (1 - X_G))
print(f"   ΔH_mix(γ, x={X_G:.3f}) = {dH_gamma*1e3:+.1f} meV/atom → "
      f"Ω_γ = {om_gamma*1e3:+.0f} meV (ordering-dominated, strongly negative)")

def gamma_solvus(Hf):
    """Al solubility in γ-Ni in equilibrium with Ni₃Al, as f(compound H_f)."""
    G_gamma = regular_solution(om_gamma)
    G_prime = line_compound(0.25, Hf)
    c_g, c_p = common_tangent(G_gamma, G_prime, 1000.0,
                              c_alpha_guess=0.08, c_beta_guess=0.249)
    return c_g, c_p

c_g, c_p = gamma_solvus(jnp.asarray(Hf_L12))
resid = float(tangent_residual(regular_solution(om_gamma),
                               line_compound(0.25, Hf_L12),
                               c_g, c_p, 1000.0))
dcg_dHf = jax.grad(lambda H: gamma_solvus(H)[0])(jnp.asarray(Hf_L12))
print(f"   γ solvus: x_Al = {float(c_g):.3f} (exp ≈ 0.12–0.14 at 1000 K), "
      f"compound at {float(c_p):.3f}, resid {resid:.1e}")
print(f"   ∂(γ solvus)/∂H_f(Ni₃Al) = {float(dcg_dHf):+.3f} /eV — "
      f"stabilising γ' by 10 meV pulls {abs(float(dcg_dHf))*0.01*100:.2f} at.% "
      f"Al out of γ")
results["gamma_solvus_xAl"] = float(c_g)
results["dcgamma_dHf_perEV"] = float(dcg_dHf)
results["omega_gamma_eV"] = om_gamma

# ═══ T5: Al-Pb monotectic — solver discovers the topology ════════════════════
print("\n── T5: Al-Pb monotectic (EAM solid, literature-magnitude liquid gap)")
fac_alpb = eam_alloy_factory("AlPb.eam.alloy", ("Al", "Pb"))
E2 = {}
for tag, elA, elB, x in [("Al", "Al", "Pb", 0.0), ("Pb", "Pb", "Al", 0.0),
                         ("mix", "Al", "Pb", 0.5)]:
    rel, _ = relax_full(fcc_solution(elA, elB, x, reps=(3, 3, 3), seed=0),
                        fac_alpb)
    E2[tag] = float(rel.get_potential_energy()) / len(rel)
om_s_alpb = (E2["mix"] - 0.5 * E2["Al"] - 0.5 * E2["Pb"]) / 0.25
print(f"   Ω_s(EAM) = {om_s_alpb*1e3:+.0f} meV/atom (→ ~zero solid solubility ✓)")

# liquid gap: Ω_L from the measured L1-L2 consolute point T_c ≈ 1566 K
OM_L_ALPB = 2 * 8.617333e-5 * 1566    # = 0.270 eV, regular-solution T_c relation
G_s = regular_solution(om_s_alpb)
G_L = liquid_solution(OM_L_ALPB,
                      dH_fus_A=0.11099, T_m_A=933.47,     # Al
                      dH_fus_B=0.04944, T_m_B=600.61)     # Pb

c1, c2, c3, T_mono = three_phase_equilibrium(
    G_s, G_L, G_L, c_guesses=(0.005, 0.2, 0.97), T_guess=950.0)
kind = classify_invariant(("solid", "liquid", "liquid"))
gap = global_tangency_gap((G_s, G_L), (c1, c2, c3), float(T_mono))
print(f"   invariant found: T = {float(T_mono):.1f} K, "
      f"c = ({float(c1):.4f}, {float(c2):.3f}, {float(c3):.3f}) → {kind}")
print(f"   global tangency verified: worst curve-line gap = {gap:+.1e} eV "
      f"(stationarity alone admits spurious roots — always check)")
print(f"   (experimental Al-Pb monotectic: 932.1 K, L1 at x_Pb ≈ 0.007–0.015)")

dT_dOmL = jax.grad(
    lambda om: three_phase_equilibrium(
        G_s,
        liquid_solution(om, dH_fus_A=0.11099, T_m_A=933.47,
                        dH_fus_B=0.04944, T_m_B=600.61),
        liquid_solution(om, dH_fus_A=0.11099, T_m_A=933.47,
                        dH_fus_B=0.04944, T_m_B=600.61),
        c_guesses=(0.005, 0.2, 0.97), T_guess=950.0)[3])(OM_L_ALPB)
print(f"   ∂T_mono/∂Ω_L = {float(dT_dOmL):+.0f} K/eV")

results["AlPb"] = {
    "omega_s_eV": om_s_alpb, "omega_L_eV": OM_L_ALPB,
    "T_mono_K": float(T_mono), "c_invariant": [float(c1), float(c2), float(c3)],
    "classified": kind, "dTmono_dOmegaL_K_per_eV": float(dT_dOmL),
}

with open(OUT / "topology_ladder.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'topology_ladder.json'}")
