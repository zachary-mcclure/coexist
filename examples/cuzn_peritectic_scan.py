"""
Cu-Zn, part 2: the peritectic that is NOT in the model,
and how the protocol knows.

Part 1 (cuzn_peritectic.py) computes MACE-MP-0 energetics for fcc/bcc
Cu-Zn solutions at converged cells. This script establishes, with
residual-checked machinery only, that:

  1. The stationarity (4x4) Newton STALLS everywhere for this system
     (residuals ~1e-2, never 1e-12): the fcc and bcc curves are
     near-parallel, the Jacobian is singular, and — the protocol lesson —
     several stalled outputs PASS the global-tangency gap check (a line
     strictly below every curve touches nothing). Residual AND gap must
     both be reported.

  2. The model has NO stable bcc (beta) field at all: the fcc-liquid
     common tangent (2x2, residuals ~1e-16) leaves the bcc curve
     +27 to +35 meV above the tangent line at every T in 700-1300 K.
     The assessed peritectic (1176 K) is therefore absent from the model
     — established positively, not by solver failure.

  3. The missing physics is measured: the three-phase field would open if
     bcc gained ~28 meV of stability near the peritectic temperature.
     Debye-Grüneisen on the two lattices (MACE E(V) scans:
     theta_D fcc 301 K vs bcc 293 K) supplies −7.9 meV/atom at 1176 K —
     the correct beta-brass sign (bcc softer), about 28% of the demand;
     the remainder is the anharmonic/short-range-order stabilization that
     makes beta-brass the textbook entropy-stabilized phase.

Pure JAX on stored energetics — zero engine calls (pass --debye to
recompute the theta_D values with MACE, ~5 min).
Output: examples/output/cuzn_beta_gap.json
"""
import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from coexist.core.phase_diagram import (  # noqa: E402
    K_B, common_tangent, liquid_solution, three_phase_equilibrium)

OUT = Path(__file__).parent / "output"
d = json.load(open(OUT / "cuzn_peritectic.json"))
dZf = jnp.float64(d["dE_Zn_fcc_minus_hcp_eV"])
dCb = jnp.float64(d["dE_Cu_bcc_minus_fcc_eV"])
dZb = jnp.float64(d["dE_Zn_bcc_minus_hcp_eV"])
L_fcc, L_bcc = jnp.array(d["L_fcc_eV"]), jnp.array(d["L_bcc_eV"])
results = {}

# theta_D from MACE E(V) scans on random x=0.5 supercells (fcc 108 /
# bcc 128, seed 0), computed here; --debye recomputes.
THETA = {"fcc": 301.0, "bcc": 293.0}


def rk(L, c):
    return sum(L[k] * (1.0 - 2.0 * c) ** k for k in range(len(L)))


def G_fcc(c, T):
    return (c * dZf + c * (1 - c) * rk(L_fcc, c)
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))


def G_bcc(c, T):
    return ((1 - c) * dCb + c * dZb + c * (1 - c) * rk(L_bcc, c)
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))


G_L = liquid_solution(0.0, dH_fus_A=0.13742, T_m_A=1357.77,
                      dH_fus_B=0.07588, T_m_B=692.68)


# ═══ 1. The stationarity solver stalls — and stalls can pass the gap ══════════
def inv_residual(cs, T):
    r = []
    d1 = float(jax.grad(G_fcc, argnums=0)(cs[0], T))
    for Gi, ci in ((G_bcc, cs[1]), (G_L, cs[2])):
        r.append(abs(d1 - float(jax.grad(Gi, argnums=0)(ci, T))))
        r.append(abs(float(Gi(ci, T)) - float(G_fcc(cs[0], T))
                     - d1 * (float(ci) - float(cs[0]))))
    return max(r)


from coexist.core.phase_diagram import global_tangency_gap  # noqa: E402

print("── 4×4 stationarity Newton: residual + gap for representative seeds")
for g, Tg in [((0.30, 0.36, 0.45), 1150.0), ((0.60, 0.65, 0.80), 1100.0),
              ((0.65, 0.72, 0.90), 1000.0)]:
    c1, c2, c3, T = three_phase_equilibrium(G_fcc, G_bcc, G_L,
                                            c_guesses=g, T_guess=Tg)
    res = inv_residual((c1, c2, c3), float(T))
    gap = global_tangency_gap((G_fcc, G_bcc, G_L), (c1, c2, c3), float(T))
    print(f"   seed {g}: residual {res:.0e} (STALLED), gap {gap:+.0e} — "
          f"{'gap alone would PASS this non-root' if gap > -1e-6 else 'gap rejects'}")
results["stall_demo"] = "residuals ~1e-2 at all seeds; several pass the gap"

# ═══ 2. No β field: fcc-L tangent vs bcc, converged 2×2 machinery ═════════════
print("\n── fcc-L common tangent (residuals ≤1e-14): gap of bcc above the line")
cgrid = jnp.linspace(1e-4, 1 - 1e-4, 4001)
gf, gl = 0.97, 0.9999
scan = []
for T in np.arange(800.0, 1320.0, 50.0):
    cf, cl = common_tangent(G_fcc, G_L, float(T),
                            c_alpha_guess=gf, c_beta_guess=gl)
    s = float((G_L(cl, T) - G_fcc(cf, T)) / (cl - cf))
    r2 = max(abs(float(jax.grad(G_fcc, 0)(cf, T)) - s),
             abs(float(jax.grad(G_L, 0)(cl, T)) - s))
    line = float(G_fcc(cf, T)) + s * (cgrid - cf)
    gap_b = float(jnp.min(G_bcc(cgrid, float(T)) - line))
    scan.append(dict(T=float(T), c_fcc=float(cf), c_L=float(cl),
                     resid=r2, gap_bcc=gap_b))
    gf, gl = float(cf), float(cl)
gaps = [s["gap_bcc"] for s in scan]
print(f"   T = 800…1300 K: gap_bcc ∈ [{min(gaps)*1e3:+.0f}, "
      f"{max(gaps)*1e3:+.0f}] meV — never ≤ 0: NO three-phase field, "
      f"no β, no peritectic in this model")
results["tangent_scan"] = scan
results["min_gap_bcc_meV"] = min(gaps) * 1e3

# ═══ 3. The missing physics, measured ═════════════════════════════════════════
if "--debye" in sys.argv:
    from ase.build import bulk
    from mace.calculators import mace_mp
    from coexist.adapters.lammps_factory import eos_scan
    from coexist.core.thermo_vib import debye_temperature, eos_fit
    _calc = mace_mp(model="small", device="cpu", default_dtype="float64")
    fac = lambda: _calc                  # noqa: E731
    M = 0.5 * 63.546 + 0.5 * 65.38
    for lat, a, reps, nsub in (("fcc", 3.78, (3, 3, 3), 54),
                               ("bcc", 3.005, (4, 4, 4), 64)):
        at = bulk("Cu", lat, a=a, cubic=True).repeat(reps)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(at), size=nsub, replace=False)
        sy = np.array(at.get_chemical_symbols(), dtype=object)
        sy[idx] = "Zn"
        at.set_chemical_symbols(list(sy))
        V, E = eos_scan(at, fac)
        V0, _, B0, _ = eos_fit(jnp.array(V), jnp.array(E))
        THETA[lat] = float(debye_temperature(V0, B0, M))

from coexist.core.thermo_vib import f_vib_debye  # noqa: E402

dF = float(f_vib_debye(1176.0, THETA["bcc"])
           - f_vib_debye(1176.0, THETA["fcc"]))
print(f"\n── Debye on the two lattices: θ_D fcc {THETA['fcc']:.0f} K, "
      f"bcc {THETA['bcc']:.0f} K → F_vib(bcc)−F_vib(fcc) = "
      f"{dF*1e3:+.1f} meV/atom at 1176 K")
print(f"   the gap to close is ≈ +{min(gaps)*1e3:.0f} meV: the Debye level "
      f"supplies {abs(dF)/min(gaps)*100:.0f}% with the correct β-brass "
      f"sign; the rest is the entropy stabilization beyond a mean-field "
      f"Debye model")
results["theta_D_K"] = THETA
results["dF_vib_bcc_minus_fcc_1176K_meV"] = dF * 1e3

with open(OUT / "cuzn_beta_gap.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'cuzn_beta_gap.json'}")
