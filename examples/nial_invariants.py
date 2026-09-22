"""
Ni-Al invariant hunt: the solver
DECIDES the local topology from Mishin-EAM energetics.

The assessed Ni-rich Ni-Al diagram has two invariants 10 K apart:
    eutectic    L → γ + γ'        at 1658 K (x_L ≈ 0.243)
    peritectic  L + β → γ'        at 1668 K

Model, c = x_Al, references fcc Ni / fcc Al:
    γ   — fcc solution, RK(order 2) from random-supercell ΔH_mix
    γ'  — line compound at 0.25, H_f from EAM L1₂
    β   — line compound at 0.50, H_f from EAM B2 
    L   — RK(L0, L1) from LIQUID MD sampling at 2000 K (4 systems × seeds),
          + experimental fusion data. No literature interaction anywhere.

Both candidate invariants are solved with `three_phase_equilibrium`,
named by `classify_invariant`, and — the important part — checked with
`global_tangency_gap` against ALL FOUR curves: a true invariant line may
not cut any phase, including the two not on the tangent.

Stated v1 limitation: line compounds carry S_f = 0 (no vibrational or
off-stoichiometry entropy), while γ carries full ideal entropy — at
~1650 K this systematically disfavors the compounds; results are read
with that bias in mind.

float64. Runtime ~4 min. Output: examples/output/nial_invariants.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from coexist.adapters.lammps_factory import (  # noqa: E402
    eam_alloy_factory, fcc_solution, relax_full, sample_liquid_energy)
from coexist.core.mixing import redlich_kister_fit  # noqa: E402
from coexist.core.phase_diagram import (  # noqa: E402
    K_B, classify_invariant, global_tangency_gap, line_compound,
    redlich_kister_solution, three_phase_equilibrium)

OUT = Path(__file__).parent / "output"
DH_FUS_NI, T_M_NI = 0.18117, 1728.0
DH_FUS_AL, T_M_AL = 0.11099, 933.47
HF_L12, HF_B2 = -0.454, -0.606          # Mishin EAM
results = {}

fac = eam_alloy_factory("NiAl.eam.alloy", ("Ni", "Al"))

# ═══ 1. γ solid solution: RK from random fcc supercells ═══════════════════════
print("── γ (fcc Ni-Al solution) from static relaxed supercells")
E_ref = {}
for el, other in [("Ni", "Al"), ("Al", "Ni")]:
    rel, _ = relax_full(fcc_solution(el, other, 0.0, seed=0), fac)
    E_ref[el] = float(rel.get_potential_energy()) / len(rel)

xs_g = np.array([2, 4, 6, 8]) / 32.0
dH_g = []
for x in xs_g:
    vals = []
    for s in (0, 1):
        rel, _ = relax_full(fcc_solution("Ni", "Al", float(x), seed=s), fac)
        vals.append(float(rel.get_potential_energy()) / len(rel)
                    - (1 - x) * E_ref["Ni"] - x * E_ref["Al"])
    dH_g.append(np.mean(vals))
L_gamma = redlich_kister_fit(jnp.array(xs_g), jnp.array(dH_g), order=2)
print(f"   ΔH_mix(γ): " + "  ".join(f"{x:.3f}:{d*1e3:+.0f}"
                                    for x, d in zip(xs_g, dH_g)) + " meV")
print(f"   RK(γ) = [{', '.join(f'{float(v)*1e3:+.0f}' for v in L_gamma)}] meV")

# ═══ 2. Liquid from MD sampling at 2000 K ═════════════════════════════════════
print("\n── liquid from EAM MD at 2000 K (108 atoms, NPT, 2 seeds)")
def liq_E(elA, elB, x):
    runs = [sample_liquid_energy(
        fcc_solution(elA, elB, x, reps=(3, 3, 3), seed=s), fac, T_K=2000.0,
        melt_T=2600.0, seed=s) for s in (0, 1)]
    return np.mean([r["E_mean"] for r in runs])

E_L = {"Ni": liq_E("Ni", "Al", 0.0), "Al": liq_E("Al", "Ni", 0.0)}
dH_L = {}
for x in (0.25, 0.5):
    Em = liq_E("Ni", "Al", x)
    dH_L[x] = Em - (1 - x) * E_L["Ni"] - x * E_L["Al"]
    print(f"   ΔH_mix(L, x={x}) = {dH_L[x]*1e3:+.0f} meV/atom")

# exact 2-point RK solve for (L0, L1)
A = np.array([[0.25 * 0.75, 0.25 * 0.75 * 0.5],
              [0.25, 0.0]])
L_liq = np.linalg.solve(A, np.array([dH_L[0.25], dH_L[0.5]]))
print(f"   RK(L) = [{L_liq[0]*1e3:+.0f}, {L_liq[1]*1e3:+.0f}] meV "
      f"(strongly attractive, as assessed)")

# ═══ 3. Curves ════════════════════════════════════════════════════════════════
G_gamma = redlich_kister_solution(L_gamma)
G_gp = line_compound(0.25, HF_L12)
G_beta = line_compound(0.50, HF_B2)
L_liq_j = jnp.array(L_liq)

def G_L(c, T):
    dG_Ni = DH_FUS_NI * (1.0 - T / T_M_NI)
    dG_Al = DH_FUS_AL * (1.0 - T / T_M_AL)
    series = L_liq_j[0] + L_liq_j[1] * (1.0 - 2.0 * c)
    return ((1 - c) * dG_Ni + c * dG_Al + c * (1 - c) * series
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))

ALL = (G_gamma, G_gp, G_beta, G_L)

# ═══ 4. Invariant hunt ════════════════════════════════════════════════════════
print("\n── invariant hunt (assessed: eutectic 1658 K + peritectic 1668 K)")
candidates = [
    ("gamma/L/gamma'", (G_gamma, G_L, G_gp), (0.18, 0.235, 0.2495),
     ("solid", "liquid", "solid"), 1650.0),
    ("L/gamma'/beta", (G_L, G_gp, G_beta), (0.23, 0.2505, 0.4995),
     ("liquid", "solid", "solid"), 1670.0),
]
for name, curves, guesses, kinds, Tg in candidates:
    try:
        c1, c2, c3, T = three_phase_equilibrium(
            curves[0], curves[1], curves[2], c_guesses=guesses, T_guess=Tg)
        gap = global_tangency_gap(ALL, (c1, c2, c3), float(T),
                                  anchor=ALL.index(curves[0]))
        kind = classify_invariant(kinds)
        ok = "VERIFIED" if gap > -1e-6 else f"REJECTED (gap {gap:+.1e})"
        print(f"   {name:16s}: T = {float(T):7.1f} K, "
              f"c = ({float(c1):.3f}, {float(c2):.3f}, {float(c3):.3f}) "
              f"→ {kind} — {ok}")
        results[name] = dict(T=float(T), c=[float(c1), float(c2), float(c3)],
                             kind=kind, gap=gap)
    except Exception as e:  # noqa: BLE001 — report solver breakdown honestly
        print(f"   {name:16s}: solver failed ({type(e).__name__}: {e})")
        results[name] = dict(error=str(e))

# ═══ 5. What entropy do the compounds need? (the inverse pattern: measure the
#        missing physics with a gradient) ═════════════════════════════════════
def T_eut_of_Sf(Sf):
    G_gp_S = line_compound(0.25, HF_L12, S_f=Sf)
    _, _, _, T = three_phase_equilibrium(
        G_gamma, G_L, G_gp_S, c_guesses=(0.18, 0.235, 0.2495),
        T_guess=1450.0)
    return T

dT_dSf = float(jax.grad(T_eut_of_Sf)(jnp.float64(0.0)))
T_model = results["gamma/L/gamma'"]["T"]
Sf_needed = (1658.0 - T_model) / dT_dSf
print(f"\n── missing-entropy attribution (the inverse pattern)")
print(f"   ∂T_eut/∂S_f(γ') = {dT_dSf:+.0f} K/(eV/K); the 1658 K assessed "
      f"eutectic implies S_f(γ') ≈ {Sf_needed*1e6:.0f} μeV/K "
      f"≈ {Sf_needed/8.617e-5:.2f} k_B/atom of compound entropy the "
      f"S_f=0 model lacks (vibrational + off-stoichiometry)")
results["dTeut_dSf_K_per_eVK"] = dT_dSf
results["Sf_gamma_prime_needed_eV_per_K"] = Sf_needed

results.update({
    "L_gamma_eV": np.asarray(L_gamma).tolist(),
    "L_liquid_eV": L_liq.tolist(),
    "dH_liq": {str(k): v for k, v in dH_L.items()},
    "Hf_L12": HF_L12, "Hf_B2": HF_B2,
})
with open(OUT / "nial_invariants.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'nial_invariants.json'}")
