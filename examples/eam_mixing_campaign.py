"""
Tier 1 + Tier 3(subregular) of the phase-diagram ladder:
Cu-Ni and Ag-Cu mixing energetics from PRODUCTION EAM potentials via the
SimulatorAdapter, Redlich-Kister fits, and the Ag-Cu eutectic re-derived
from engine energetics — with gradients back into the engine.

Rungs demonstrated:
  1. Cu-Ni isomorphous (Fischer 2019 EAM): Ω, T_c vs CALPHAD; lens diagram
  2. Ag-Cu eutectic (Williams 2006 EAM): T_e, x_e, solvus limbs vs experiment
     — regular (x=0.5 only) AND subregular (RK order 2 from 7 compositions)
  ∂: ∂T_e/∂L_k (implicit, FD-checked), ∂T_e/∂R_atoms through LAMMPS
     (1 engine call vs 192 for central FD)

Everything float64. Runtime ~2-4 min (≈50 relaxations).
Outputs: examples/output/eam_campaign.json (numbers for the paper + figures).
"""
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from coexist.adapters.lammps_factory import (  # noqa: E402
    adapter_for_atoms, eam_alloy_factory, fcc_solution, relax_full)
from coexist.core.mixing import (  # noqa: E402
    omega_regular, redlich_kister_fit)
from coexist.core.phase_diagram import (  # noqa: E402
    K_B, common_tangent, eutectic_diagram, eutectic_point, liquid_solution,
    redlich_kister_solution, regular_solution)

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# Experimental fusion data (same constants as eutectic_phase_diagram.py)
T_M = {"Ag": 1234.93, "Cu": 1357.77, "Ni": 1728.0}
DH_FUS = {"Ag": 0.11691, "Cu": 0.13742, "Ni": 0.18117}   # eV/atom
OMEGA_L_AGCU = 0.1555     # eV/atom, CALPHAD liquid L0 (+15 kJ/mol)

# Fidelity protocol: 108-atom cells (3×3×3), the converged size from
# the size-scaling study (32-atom cells carry a +47 meV bias on Ω).
REPS = (3, 3, 3)
XS = np.array([14, 27, 41, 54, 67, 81, 94]) / 108.0
SEEDS = (0, 1, 2)

results = {}


def mixing_curve(name, el_A, el_B, pot_file, file_order):
    """ΔH_mix(x) with relaxation, seed stats, and the pure references."""
    fac = eam_alloy_factory(pot_file, file_order)
    t0 = time.time()

    E_pure = {}
    for el in (el_A, el_B):
        at = fcc_solution(el, el_B if el == el_A else el_A, 0.0, reps=REPS,
                          seed=0)
        rel, _ = relax_full(at, fac)
        E_pure[el] = float(rel.get_potential_energy()) / len(rel)

    dH_mean, dH_std, relaxed_mid = [], [], None
    for x in XS:
        samples = []
        for seed in SEEDS:
            at = fcc_solution(el_A, el_B, float(x), reps=REPS, seed=seed)
            rel, _ = relax_full(at, fac)
            n = len(rel)
            dh = (float(rel.get_potential_energy()) / n
                  - (1 - x) * E_pure[el_A] - x * E_pure[el_B])
            samples.append(dh)
            if abs(x - 0.5) < 1e-9 and seed == 0:
                relaxed_mid = rel
        dH_mean.append(float(np.mean(samples)))
        dH_std.append(float(np.std(samples)))

    dH_mean = np.array(dH_mean)
    L = redlich_kister_fit(jnp.array(XS), jnp.array(dH_mean), order=2)
    omega_05 = float(dH_mean[XS == 0.5][0] / 0.25)

    print(f"\n── {name} ({pot_file}) — {time.time()-t0:.0f} s")
    print(f"   ΔH_mix(x):  " + "  ".join(
        f"{x:.3f}:{m*1e3:+.1f}±{s*1e3:.1f}" for x, m, s in
        zip(XS, dH_mean, dH_std)) + "  meV/atom")
    print(f"   RK fit  L = [{', '.join(f'{v*1e3:+.1f}' for v in np.asarray(L))}] meV")
    print(f"   Ω(x=0.5) = {omega_05*1e3:+.1f} meV/atom")

    results[name] = {
        "xs": XS.tolist(), "dH_mean_eV": dH_mean.tolist(),
        "dH_std_eV": dH_std, "L_RK_eV": np.asarray(L).tolist(),
        "omega_05_eV": omega_05, "E_pure_eV_per_atom": E_pure,
    }
    return fac, E_pure, np.asarray(L), omega_05, relaxed_mid


# ═══ Ag-Cu (Williams/Mishin/Hamilton 2006) ════════════════════════════════════
fac_agcu, Epure_agcu, L_agcu, om_agcu, mid_agcu = mixing_curve(
    "AgCu", "Ag", "Cu", "CuAg.eam.alloy", ("Cu", "Ag"))

G_L = liquid_solution(OMEGA_L_AGCU,
                      dH_fus_A=DH_FUS["Ag"], T_m_A=T_M["Ag"],
                      dH_fus_B=DH_FUS["Cu"], T_m_B=T_M["Cu"])

print("\n── Ag-Cu eutectic from the engine (exp: T_e=1052 K, x_e=0.399, "
      "c_α=0.141, c_β=0.951)")
rows = {}
for label, G_s in [
        ("EAM regular", regular_solution(om_agcu)),
        ("EAM subregular RK2", redlich_kister_solution(jnp.array(L_agcu)))]:
    ca, cl, cb, Te = eutectic_point(G_s, G_L)
    rows[label] = dict(T_e=float(Te), x_e=float(cl),
                       c_alpha=float(ca), c_beta=float(cb))
    print(f"   {label:22s} T_e={float(Te):7.1f} K  x_e={float(cl):.3f}  "
          f"c_α={float(ca):.3f}  c_β={float(cb):.3f}")
print("  ")
results["AgCu_eutectic"] = rows

# ── ∂T_e/∂L_k through the triple tangent (implicit), FD check on L0 ──────────
def Te_of_L(L):
    _, _, _, Te = eutectic_point(redlich_kister_solution(L), G_L)
    return Te

gL = jax.grad(Te_of_L)(jnp.array(L_agcu))
h = 1e-3  # eV
Lp = np.asarray(L_agcu).copy(); Lp[0] += h
Lm = np.asarray(L_agcu).copy(); Lm[0] -= h
fd = (float(Te_of_L(jnp.array(Lp))) - float(Te_of_L(jnp.array(Lm)))) / (2 * h)
print(f"   ∂T_e/∂L = [{', '.join(f'{float(v):+.0f}' for v in gL)}] K/eV"
      f"   (FD on L0: {fd:+.0f} K/eV, rel err "
      f"{abs(float(gL[0])-fd)/abs(fd):.1e})")
results["AgCu_dTe_dL_K_per_eV"] = np.asarray(gL).tolist()
results["AgCu_dTe_dL0_FD"] = fd

# ── ∂T_e/∂R_atoms THROUGH LAMMPS: 1 engine call ──────────────────────────────
ad = adapter_for_atoms(mid_agcu, fac_agcu, dtype=jnp.float64, name="LAMMPS")
R0 = jnp.array(mid_agcu.get_positions())
n_at = len(mid_agcu)
E_ref = 0.5 * Epure_agcu["Ag"] + 0.5 * Epure_agcu["Cu"]

def Te_of_R(R):
    dH = ad(R) / n_at - E_ref
    G_s = regular_solution(omega_regular(dH, 0.5))
    _, _, _, Te = eutectic_point(G_s, G_L)
    return Te

calls_before = ad.n_calls
gR = jax.grad(Te_of_R)(R0)
calls = ad.n_calls - calls_before
print(f"   ∂T_e/∂R through LAMMPS: max |∂T_e/∂R_i| = "
      f"{float(jnp.abs(gR).max()):.1f} K/Å  "
      f"({calls} engine call(s); central FD would need {2*3*n_at})")
results["AgCu_dTe_dR_max_K_per_A"] = float(jnp.abs(gR).max())
results["AgCu_dTe_dR_engine_calls"] = int(calls)

# ── full diagram sweep for the figure (subregular solid) ─────────────────────
diag = eutectic_diagram(redlich_kister_solution(jnp.array(L_agcu)), G_L,
                        T_m_A=T_M["Ag"], T_m_B=T_M["Cu"], n_T=40)
results["AgCu_diagram"] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                           for k, v in diag.items()}

# ═══ Cu-Ni (Fischer/Schmitz/Eich 2019) ════════════════════════════════════════
fac_cuni, Epure_cuni, L_cuni, om_cuni, _ = mixing_curve(
    "CuNi", "Cu", "Ni", "CuNi.eam.alloy", ("Cu", "Ni"))

T_c = om_cuni / (2 * K_B)
print(f"\n── Cu-Ni miscibility gap: T_c = Ω/2k_B = {T_c:.0f} K "
      f"(CALPHAD 600-650 K; EMT 490 K relaxed [Table 2], 523 K single-point)")
results["CuNi_Tc_K"] = T_c

# Lens (liquidus/solidus): RK solid vs CALPHAD-magnitude liquid.
# An ideal liquid (Ω_L=0) against Ω_s=+111 meV shifts the lens strongly
# Ni-ward — in the real system the liquid's positive interactions largely
# cancel the solid hump. Ω_L ≈ +12 kJ/mol (An Mey 1992 assessment magnitude).
OMEGA_L_CUNI = 0.125   # eV/atom, representative CALPHAD liquid L0
G_s_cuni = redlich_kister_solution(jnp.array(L_cuni))
G_L_cuni = liquid_solution(OMEGA_L_CUNI,
                           dH_fus_A=DH_FUS["Cu"], T_m_A=T_M["Cu"],
                           dH_fus_B=DH_FUS["Ni"], T_m_B=T_M["Ni"])
# Continuation DOWNWARD from the Ni end: the isomorphous tangent lives on
# the Ni-rich branch near T_m(Ni) and slides toward the Cu end as T drops.
# (Seeding mid-composition at mid-T stalls on a non-root near c→0 — the
# residual diagnostic distinguishes the two: 1e-16 vs 1e-2.)
Ts = np.linspace(T_M["Ni"] - 4, T_M["Cu"] + 4, 40)
liq, sol = [], []
gl, gs = 0.97, 0.99
for T in Ts:
    cl, cs = common_tangent(G_L_cuni, G_s_cuni, float(T),
                            c_alpha_guess=gl, c_beta_guess=gs)
    gl, gs = float(cl), float(cs)
    liq.append(gl); sol.append(gs)
Ts = Ts[::-1]; liq = liq[::-1]; sol = sol[::-1]   # ascending T for plotting
print(f"   lens sweep: liquidus x_Ni {liq[0]:.3f}→{liq[-1]:.3f}, "
      f"solidus {sol[0]:.3f}→{sol[-1]:.3f} over {Ts[0]:.0f}-{Ts[-1]:.0f} K "
      f"(EAM solid + Ω_L={OMEGA_L_CUNI*1e3:.0f} meV CALPHAD-magnitude liquid)")
results["CuNi_lens"] = {"T": Ts.tolist(), "liquidus": liq, "solidus": sol}

with open(OUT / "eam_campaign.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'eam_campaign.json'}")
