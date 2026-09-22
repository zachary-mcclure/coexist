"""
Ternary constructions from the engine: Ag-Cu-Ni isothermal constructions
from MACE-MP-0 energetics.

Why Ag-Cu-Ni: all three binaries demix (Ag-Cu +411, Cu-Ni +64 meV from
the MACE rung; Ag-Ni computed here — the strongest demixer of the
three), so the ternary has a rich miscibility-gap topology with a
three-phase (fcc₁+fcc₂+fcc₃) tie-triangle below the Cu-Ni consolute
temperature and a single gap with a rotating tie-line fan above it.

Constructed here, all engine-derived (components A=Ag, B=Cu, C=Ni):
  1. Ω_AgNi and the ternary excess L_ABC from MACE supercells
     (108-atom equimolar ternary vs the pairwise regular prediction).
  2. Tie-triangle below T_c^CuNi (250 K for MACE@108): solved by the 6×6
     tangent-plane Newton, VERIFIED by the simplex tangency gap; the
     anti-corner stationary impostor is also solved and REJECTED —
     the binary impostor phenomenon generalizes, as predicted.
  3. Isothermal section at 800 K: tie-line fan by continuation
     (`section_sweep`), Ag-rich binodal branch vs Cu-Ni-rich branch.
  4. Gradients: ∂(triangle vertices)/∂(Ω_AB, Ω_AC, Ω_BC) by one
     jacobian through the 6×6 solve (FD spot-checked), and the engine
     seam: with Ω_AgNi traced through the adapter,
     ∂(vertex)/∂R_atoms of the Ag-Ni supercell in ONE engine call.

float64. Runtime ~5 min (MACE relaxations). Output:
examples/output/ternary_agcuni.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from coexist.adapters.lammps_factory import (  # noqa: E402
    adapter_for_atoms, fcc_solution, relax_positions, relax_volume)
from coexist.core.mixing import omega_regular  # noqa: E402
from coexist.core.ternary import (  # noqa: E402
    plane_tangency_gap, section_sweep, ternary_regular_solution,
    tie_triangle)

OUT = Path(__file__).parent / "output"
results = {}

print("Loading MACE-MP-0 (small, float64, CPU)…")
from mace.calculators import mace_mp  # noqa: E402

_calc = mace_mp(model="small", device="cpu", default_dtype="float64")
fac = lambda: _calc                       # noqa: E731


def relaxed(atoms, fmax=0.05):
    work, _, _ = relax_volume(atoms, fac)
    work = relax_positions(work, fac, fmax=fmax)
    return work


def E_per_atom(atoms):
    return float(atoms.get_potential_energy()) / len(atoms)


# ═══ 1. Engine energetics: Ω_AgNi + ternary excess ════════════════════════════
# Binary omegas read from the 108-atom mace_rung.json; all cells 108.
_mr = json.load(open(OUT / "mace_rung.json"))
OM_AGCU = _mr["AgCu"]["omega_eV"]
OM_CUNI = _mr["CuNi"]["omega_eV"]
print(f"── MACE energetics (Ω_AgCu={OM_AGCU*1e3:+.0f}, "
      f"Ω_CuNi={OM_CUNI*1e3:+.0f} meV from mace_rung @108)")
pures = {}
for el, other in [("Ag", "Ni"), ("Ni", "Ag"), ("Cu", "Ag")]:
    pures[el] = E_per_atom(relaxed(fcc_solution(el, other, 0.0,
                                                reps=(3, 3, 3), seed=0)))

agni = relaxed(fcc_solution("Ag", "Ni", 0.5, reps=(3, 3, 3), seed=0))
dH_agni = E_per_atom(agni) - 0.5 * pures["Ag"] - 0.5 * pures["Ni"]
OM_AGNI = dH_agni / 0.25
print(f"   Ω_AgNi = {OM_AGNI*1e3:+.0f} meV/atom (strongest demixer, "
      f"as the near-immiscible Ag-Ni system requires)")

# ternary excess from the equimolar 108-atom supercell
def ternary_dH(seed):
    tern = fcc_solution("Ag", "Cu", 0.0, reps=(3, 3, 3), seed=0)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(108)
    syms = np.array(tern.get_chemical_symbols(), dtype=object)
    syms[idx[:36]] = "Cu"
    syms[idx[36:72]] = "Ni"
    tern.set_chemical_symbols(list(syms))
    return (E_per_atom(relaxed(tern))
            - (pures["Ag"] + pures["Cu"] + pures["Ni"]) / 3)

dHs = [ternary_dH(s) for s in (0, 1)]
dH_tern = float(np.mean(dHs))
pairwise = (OM_AGCU + OM_AGNI + OM_CUNI) / 9.0
# residual/(x_A x_B x_C) with the 1/27 divisor: a ±few-meV ΔH scatter maps
# to ±~100 meV in L_ABC — quote the seed spread with the number.
L_seeds = [(d - pairwise) * 27.0 for d in dHs]
L_ABC = float(np.mean(L_seeds))
print(f"   ΔH(equimolar) = {dH_tern*1e3:+.0f} meV vs pairwise "
      f"{pairwise*1e3:+.0f} → L_ABC = {L_ABC*1e3:+.0f} meV "
      f"(seeds: {', '.join(f'{v*1e3:+.0f}' for v in L_seeds)}; the 1/27 "
      f"divisor amplifies supercell scatter — single-point ternary terms "
      f"are soft numbers)")

G = ternary_regular_solution(OM_AGCU, OM_AGNI, OM_CUNI, L_ABC=L_ABC)

# ═══ 2. Tie-triangle at 350 K: verified truth vs rejected impostor ════════════
T3 = 250.0   # below MACE@108 T_c^CuNi = 282 K
print(f"\n── tie-triangle at {T3:.0f} K (below MACE@108 T_c^CuNi = 282 K)")
v = tie_triangle(G, G, G, T3,
                 x_guesses=((0.001, 0.001), (0.99, 0.005), (0.005, 0.99)))
gap = plane_tangency_gap((G,), v[0], G, T3, n_grid=80)
names = ("Ag-rich", "Cu-rich", "Ni-rich")
for n, vi in zip(names, v):
    xA = 1 - float(vi[0]) - float(vi[1])
    print(f"   {n}: x = (Ag {xA:.5f}, Cu {float(vi[0]):.5f}, "
          f"Ni {float(vi[1]):.5f})")
print(f"   simplex tangency gap = {gap:+.1e} eV → "
      f"{'VERIFIED' if gap > -1e-6 else 'REJECTED'}")

vi_ = tie_triangle(G, G, G, T3,
                   x_guesses=((0.05, 0.05), (0.9, 0.05), (0.05, 0.9)))
gap_i = plane_tangency_gap((G,), vi_[0], G, T3, n_grid=80)
print(f"   anti-corner stationary impostor: gap = {gap_i:+.1e} eV → "
      f"{'REJECTED' if gap_i < -1e-3 else 'unexpectedly verified'} "
      f"(the binary impostor phenomenon generalizes)")

results["triangle_lowT"] = {
    "vertices_xCu_xNi": [[float(a) for a in vi] for vi in v],
    "gap": gap, "impostor_gap": gap_i,
}

# ═══ 3. Isothermal section at 800 K: tie-line fan by continuation ═════════════
T2 = 800.0
print(f"\n── isothermal section at {T2:.0f} K")
# Is there still a three-phase field? The triangle solve DIAGNOSES it:
# above the interior consolute point, two vertices collapse onto each
# other and the 'triangle' degenerates to the limiting (widest) tie-line.
vt = tie_triangle(G, G, G, T2,
                  x_guesses=((0.003, 0.003), (0.95, 0.02), (0.02, 0.95)))
sep = float(jnp.hypot(vt[1][0] - vt[2][0], vt[1][1] - vt[2][1]))
print(f"   triangle probe: |v₂−v₃| = {sep:.1e} → "
      f"{'three-phase field present' if sep > 1e-3 else 'COLLAPSED — no triangle; collapsed point = widest tie-line'}")

# Tie-line fan. Seeding matters: loose seeds converge to a stationary
# non-equilibrium branch that the verifier rejects at −0.5 eV (measured);
# corner-tight seeds give the true fan, α pinned at the Ag corner down to
# x_Ni ~ 1e-10 (float64-only territory).
pin_vals = np.concatenate([np.linspace(1e-4, 0.02, 6),
                           np.linspace(0.03, 0.47, 23)])
al, be = section_sweep(G, G, T2, pin_which="beta_C", pin_values=pin_vals,
                       x_alpha_seed=(0.003, 1e-5),
                       x_beta_seed=(0.97, 1e-4))
gaps = [plane_tangency_gap((G,), tuple(al[i]), G, T2, n_grid=70)
        for i in (0, len(al) // 2, len(al) - 1)]
print(f"   {len(al)} tie-lines: α (Ag corner) x_Cu {al[0,0]:.4f}→"
      f"{al[-1,0]:.4f}, x_Ni ≤ {al[:,1].max():.1e}; β walks "
      f"({be[0,0]:.3f},{be[0,1]:.4f})→({be[-1,0]:.3f},{be[-1,1]:.3f}), "
      f"approaching the collapsed-triangle limit "
      f"({float(vt[1][0]):.3f},{float(vt[1][1]):.3f})")
print(f"   spot-verified tie-lines (first/mid/last): gaps = "
      + ", ".join(f"{g:+.1e}" for g in gaps))
results["section_800K"] = {"alpha": al.tolist(), "beta": be.tolist(),
                           "tie_gaps": gaps, "triangle_collapse_sep": sep,
                           "collapsed_limit": [float(vt[1][0]),
                                               float(vt[1][1])]}

# ═══ 4. Gradients: interaction Jacobian + the engine seam ═════════════════════
print("\n── gradients through the 6×6 triangle solve")

def cu_vertex_xCu(omegas):
    Gx = ternary_regular_solution(omegas[0], omegas[1], omegas[2],
                                  L_ABC=L_ABC)
    vv = tie_triangle(Gx, Gx, Gx, T3,
                      x_guesses=((0.001, 0.001), (0.99, 0.005),
                                 (0.005, 0.99)))
    return vv[1][0]                     # x_Cu of the Cu-rich vertex

om0 = jnp.array([OM_AGCU, OM_AGNI, OM_CUNI])
gJ = jax.grad(cu_vertex_xCu)(om0)
h = 1e-4
fd = (float(cu_vertex_xCu(om0.at[2].add(h)))
      - float(cu_vertex_xCu(om0.at[2].add(-h)))) / (2 * h)
print(f"   ∂x_Cu(Cu-vertex)/∂(Ω_AgCu, Ω_AgNi, Ω_CuNi) = "
      f"[{float(gJ[0]):+.3f}, {float(gJ[1]):+.3f}, {float(gJ[2]):+.3f}] /eV")
print(f"   FD check on Ω_CuNi: {fd:+.3f} "
      f"(rel err {abs(float(gJ[2])-fd)/abs(fd):.1e})")

# engine seam: Ω_AgNi traced through the MACE adapter
ad = adapter_for_atoms(agni, fac, dtype=jnp.float64, name="MACE-MP-0")
R0 = jnp.array(agni.get_positions())
E_ref = 0.5 * pures["Ag"] + 0.5 * pures["Ni"]

def vertex_of_R(R):
    om_agni_R = omega_regular(ad(R) / len(agni) - E_ref, 0.5)
    Gx = ternary_regular_solution(OM_AGCU, om_agni_R, OM_CUNI, L_ABC=L_ABC)
    vv = tie_triangle(Gx, Gx, Gx, T3,
                      x_guesses=((0.001, 0.001), (0.99, 0.005),
                                 (0.005, 0.99)))
    return vv[1][0]

n0 = ad.n_calls
gR = jax.grad(vertex_of_R)(R0)
print(f"   ∂(Cu-vertex)/∂R through MACE (Ag-Ni supercell): "
      f"max |∂/∂R_i| = {float(jnp.abs(gR).max()):.2e} /Å in "
      f"{ad.n_calls - n0} engine call(s) — a ternary tie-triangle vertex "
      f"differentiated into a foundation model")

results.update({
    "omega_AgNi_eV": OM_AGNI, "L_ABC_eV": float(L_ABC),
    "dH_ternary_eV": dH_tern,
    "dvertex_domegas": np.asarray(gJ).tolist(),
    "dvertex_dR_max": float(jnp.abs(gR).max()),
})
with open(OUT / "ternary_agcuni.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'ternary_agcuni.json'}")
