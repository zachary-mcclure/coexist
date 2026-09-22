"""
Ni-Al liquid resampled AT THE INVARIANT TEMPERATURES.

An earlier version sampled the Ni-Al liquid at 2000 K and used a temperature-independent
RK(L0, L1) down at the 1418 K model eutectic — a 580 K extrapolation the
Ag-Cu slope measurement (≈ −135 μeV/K) says is worth tens of meV. Here the
liquid is sampled at 1500 K and 1670 K (bracketing both the model eutectic
and the assessed invariants), fitted to RK(L0, L1) at each temperature,
and the invariant hunt re-run with a linear-in-T liquid interaction —
the engine's version of CALPHAD's linear T-term.

Pure Ni at 1500/1670 K is an undercooled liquid (T_m = 1728 K): that is
the correct reference state for liquid mixing (the fusion term carries the
solid-liquid part), but each trajectory is checked for crystallization
(an E_pot drop of order ΔH_fus between seeds would flag it).

Production ensemble: isotropic MTK NPT.

float64. Runtime ~5 min (16 MD trajectories).
Output: examples/output/nial_liquid_retemp.json
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
SEEDS = (0, 1)
T_LO, T_HI = 1500.0, 1670.0
results = {}

fac = eam_alloy_factory("NiAl.eam.alloy", ("Ni", "Al"))

# ═══ 1. γ solid solution: RK from random fcc supercells ═════════════
print("── γ (fcc Ni-Al solution) from static relaxed supercells")
E_ref = {}
for el, other in [("Ni", "Al"), ("Al", "Ni")]:
    rel, _ = relax_full(fcc_solution(el, other, 0.0, reps=(3, 3, 3), seed=0),
                        fac)
    E_ref[el] = float(rel.get_potential_energy()) / len(rel)

xs_g = np.array([7, 14, 20, 27]) / 108.0   # 108-atom converged cells
dH_g = []
for x in xs_g:
    vals = []
    for s in SEEDS:
        rel, _ = relax_full(fcc_solution("Ni", "Al", float(x),
                                         reps=(3, 3, 3), seed=s), fac)
        vals.append(float(rel.get_potential_energy()) / len(rel)
                    - (1 - x) * E_ref["Ni"] - x * E_ref["Al"])
    dH_g.append(np.mean(vals))
L_gamma = redlich_kister_fit(jnp.array(xs_g), jnp.array(dH_g), order=2)
print(f"   RK(γ) = [{', '.join(f'{float(v)*1e3:+.0f}' for v in L_gamma)}] meV")

# ═══ 2. Liquid sampled at the invariant temperatures ══════════════════════════
def liq_E(elA, elB, x, T):
    runs = [sample_liquid_energy(
        fcc_solution(elA, elB, x, reps=(3, 3, 3), seed=s), fac, T_K=T,
        melt_T=2600.0, prod_ps=30.0, seed=s) for s in SEEDS]
    E_seeds = [r["E_mean"] for r in runs]
    spread = max(E_seeds) - min(E_seeds)
    if spread > 0.5 * DH_FUS_NI:          # crystallization guard
        print(f"   WARNING: seed spread {spread*1e3:.0f} meV at "
              f"x={x}, T={T} — possible crystallization")
    return np.mean(E_seeds), spread

L_RK_T = {}
for T in (T_LO, T_HI):
    print(f"\n── liquid from EAM MD at {T:.0f} K (MTK NPT, {len(SEEDS)} seeds)")
    E_L, spreads = {}, {}
    E_L["Ni"], spreads["Ni"] = liq_E("Ni", "Al", 0.0, T)
    E_L["Al"], spreads["Al"] = liq_E("Al", "Ni", 0.0, T)
    dH_L = {}
    for x in (0.25, 0.5):
        Em, sp = liq_E("Ni", "Al", x, T)
        dH_L[x] = Em - (1 - x) * E_L["Ni"] - x * E_L["Al"]
        print(f"   ΔH_mix(L, x={x}) = {dH_L[x]*1e3:+.0f} meV/atom "
              f"(seed spread {sp*1e3:.0f} meV)")
    A = np.array([[0.25 * 0.75, 0.25 * 0.75 * 0.5],
                  [0.25, 0.0]])
    L_RK_T[T] = np.linalg.solve(A, np.array([dH_L[0.25], dH_L[0.5]]))
    print(f"   RK(L, {T:.0f} K) = [{L_RK_T[T][0]*1e3:+.0f}, "
          f"{L_RK_T[T][1]*1e3:+.0f}] meV")
    results[f"L_liquid_{T:.0f}K_eV"] = L_RK_T[T].tolist()
    results[f"dH_liq_{T:.0f}K"] = {str(k): v for k, v in dH_L.items()}

sl = (L_RK_T[T_HI] - L_RK_T[T_LO]) / (T_HI - T_LO)
print(f"\n   dL0/dT ≈ {sl[0]*1e6:+.0f} μeV/K, dL1/dT ≈ {sl[1]*1e6:+.0f} μeV/K")
results["dL_dT_eV_per_K"] = sl.tolist()

# ═══ 3. Curves with a linear-in-T liquid ══════════════════════════════════════
G_gamma = redlich_kister_solution(L_gamma)
G_gp = line_compound(0.25, HF_L12)
G_beta = line_compound(0.50, HF_B2)
L_lo = jnp.array(L_RK_T[T_LO])
sl_j = jnp.array(sl)

def G_L(c, T):
    dG_Ni = DH_FUS_NI * (1.0 - T / T_M_NI)
    dG_Al = DH_FUS_AL * (1.0 - T / T_M_AL)
    L_T = L_lo + (T - T_LO) * sl_j
    series = L_T[0] + L_T[1] * (1.0 - 2.0 * c)
    return ((1 - c) * dG_Ni + c * dG_Al + c * (1 - c) * series
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))

ALL = (G_gamma, G_gp, G_beta, G_L)

# ═══ 4. Invariant hunt, corrected liquid ══════════════════════════════════════
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
        # anchor the verifier line on the curve c1 actually belongs to
        # (mispairing spuriously rejected a true invariant)
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

# ═══ 5. Missing-entropy attribution at the corrected liquid ═══════════════════
if "T" in results.get("gamma/L/gamma'", {}):
    def T_eut_of_Sf(Sf):
        G_gp_S = line_compound(0.25, HF_L12, S_f=Sf)
        _, _, _, T = three_phase_equilibrium(
            G_gamma, G_L, G_gp_S, c_guesses=(0.18, 0.235, 0.2495),
            T_guess=results["gamma/L/gamma'"]["T"])
        return T

    dT_dSf = float(jax.grad(T_eut_of_Sf)(jnp.float64(0.0)))
    T_model = results["gamma/L/gamma'"]["T"]
    Sf_lin = (1658.0 - T_model) / dT_dSf
    # exact 1D Newton: T_eut(S_f*) = 1658 K (the linear estimate misses
    # the curvature of the solve)
    g = jax.grad(T_eut_of_Sf)
    Sf = jnp.float64(0.0)
    for _ in range(8):
        Sf = Sf - (T_eut_of_Sf(Sf) - 1658.0) / g(Sf)
    Sf_exact = float(Sf)
    # the root must satisfy its own equation before it is stored: an
    # undamped Newton that wandered would otherwise write a plausible
    # garbage number to the JSON with nothing downstream to catch it
    T_resid = abs(float(T_eut_of_Sf(Sf)) - 1658.0)
    assert T_resid < 1e-6, f"S_f Newton did not converge: |T-1658| = {T_resid:.2e} K"
    print(f"\n── missing-entropy attribution (corrected liquid)")
    print(f"   ∂T_eut/∂S_f(γ') = {dT_dSf:+.0f} K/(eV/K) → linear estimate "
          f"{Sf_lin/8.617e-5:.2f} k_B/atom")
    print(f"   exact solve T(S_f*) = 1658 K → S_f* = {Sf_exact*1e6:.0f} "
          f"μeV/K = {Sf_exact/8.617e-5:.2f} k_B/atom")
    results["dTeut_dSf_K_per_eVK"] = dT_dSf
    results["Sf_gamma_prime_needed_eV_per_K"] = Sf_lin
    results["Sf_exact_eV_per_K"] = Sf_exact
    results["Sf_exact_kB"] = Sf_exact / 8.617e-5

results["L_gamma_eV"] = np.asarray(L_gamma).tolist()
results["Hf_L12"] = HF_L12
results["Hf_B2"] = HF_B2
if "T" in results.get("gamma/L/gamma'", {}):
    T_fig = results["gamma/L/gamma'"]["T"]
    results["L_liquid_eV"] = [float(L_RK_T[T_LO][i]
                                    + (T_fig - T_LO) * sl[i])
                              for i in range(2)]
with open(OUT / "nial_liquid_retemp.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'nial_liquid_retemp.json'}")
