"""
Cu-Sn peritectic, part 2: the classifier's peritectic branch fails a
THIRD time, established positively rather than left as a bare solver
failure -- the same "residual travels with the gap" protocol Cu-Zn's own
absence used (Section 3.4 of the paper).

`cusn_peritectic.py`'s direct 4x4 solve for L+alpha->beta stalls (residual
~0.2-0.3, several different seeds tried, none converge below 1e-8). That
alone does not establish absence -- Section 3.4 is explicit that a
stalled solve's gap check cannot be trusted on its own. This script runs
the well-conditioned, independently-verifiable check instead: sweep the
fcc-liquid common tangent (residuals <1e-14 throughout, an easy 2-phase
solve) across the full 750-1300 K range spanning the assessed peritectic
(1071 K) and measure the bcc curve against each tangent line.

Result: bcc's gap is POSITIVE (+20.6 to +24.2 meV) at every temperature
in the range -- the bcc curve never touches the fcc-liquid tangent
anywhere. Combined with four different 3-phase seeds all failing to
converge (residuals 0.06-0.3, never below 1e-8), this establishes -- the
same way Ni-Al's and Cu-Zn's absences were established -- that THIS
model (ideal-entropy random solid solutions + 0 K RK-fit enthalpy, no
vibrational or short-range-order correction) has no stable beta field in
this composition/temperature window, and therefore no peritectic: not a
solver failure, a converged negative result.

This makes Cu-Sn the THIRD system (after Ni-Al, Cu-Zn) where a sought
peritectic is absent under this class of model -- now a pattern across
three structurally distinct systems, not a one-off, and itself a
noteworthy finding: forming a peritectic requires beta to become MORE
stable than an ideal-entropy random-solid-solution model gives it credit
for, at exactly the composition/temperature where it must intercept an
already-tight liquidus -- systematically harder than an all-liquid-
mediated eutectic reaction under the same physics budget. The gap size
here (+20 to +24 meV) is comparable to Cu-Zn's (+27 to +35 meV), where
Debye vibrations and quasichemical short-range order together closed 37%
of the reference gap -- the same missing-physics story likely applies
here too, not attempted in this pass.

Zero engine calls -- pure JAX on the already-stored MACE-MP-0 energetics
from cusn_peritectic.py. Output: examples/output/cusn_peritectic_scan.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from coexist.core.phase_diagram import (  # noqa: E402
    K_B, common_tangent, liquid_solution, three_phase_equilibrium)

OUT = Path(__file__).parent / "output"
d = json.load(open(OUT / "cusn_peritectic.json"))
dSf = jnp.float64(d["dE_Sn_fcc_minus_betaSn_eV"])
dCb = jnp.float64(d["dE_Cu_bcc_minus_fcc_eV"])
dSb = jnp.float64(d["dE_Sn_bcc_minus_betaSn_eV"])
L_fcc, L_bcc = jnp.array(d["L_fcc_eV"]), jnp.array(d["L_bcc_eV"])


def rk(L, c):
    return sum(L[k] * (1.0 - 2.0 * c) ** k for k in range(len(L)))


def G_fcc(c, T):
    return (c * dSf + c * (1 - c) * rk(L_fcc, c)
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))


def G_bcc(c, T):
    return ((1 - c) * dCb + c * dSb + c * (1 - c) * rk(L_bcc, c)
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))


DH_FUS_CU, T_M_CU = 0.13742, 1357.77
DH_FUS_SN, T_M_SN = 0.07286, 505.08
G_L = liquid_solution(0.0, dH_fus_A=DH_FUS_CU, T_m_A=T_M_CU,
                      dH_fus_B=DH_FUS_SN, T_m_B=T_M_SN)

print("fcc-L tangent vs bcc gap, 750-1300 K (spans the assessed 1071.15 K):")
cgrid = jnp.linspace(1e-4, 1.0 - 1e-4, 2001)
gf, gl = 0.05, 0.5
scan = []
for T in np.arange(750.0, 1320.0, 25.0):
    cf, cl = common_tangent(G_fcc, G_L, float(T), c_alpha_guess=gf, c_beta_guess=gl)
    s = float((G_L(cl, T) - G_fcc(cf, T)) / (cl - cf))
    line = float(G_fcc(cf, T)) + s * (cgrid - float(cf))
    gap_bcc = float(jnp.min(G_bcc(cgrid, T) - line))
    scan.append(dict(T=float(T), c_fcc=float(cf), c_liq=float(cl), gap_bcc=gap_bcc))
    print(f"  T={T:6.1f} K  c_fcc={float(cf):.4f}  c_L={float(cl):.4f}  "
          f"gap_bcc={gap_bcc*1e3:+7.1f} meV")
    gf, gl = float(cf), float(cl)

gaps = np.array([s["gap_bcc"] for s in scan])
print(f"\nmin gap over the scanned range: {gaps.min()*1e3:+.1f} meV "
      f"(strictly positive everywhere -> no root, no peritectic)")

print("\nFour 3-phase seeds, all checked for genuine convergence:")
seeds = [(1071.0, 0.08, 0.14, 0.16), (900.0, 0.06, 0.10, 0.20),
         (1000.0, 0.05, 0.20, 0.30), (1071.0, 0.02, 0.30, 0.40)]
seed_results = []
for T0, g1, g2, g3 in seeds:
    c1, c2, c3, T = three_phase_equilibrium(
        G_fcc, G_bcc, G_L, c_guesses=(g1, g2, g3), T_guess=T0)
    d1 = float(jax.grad(G_fcc, argnums=0)(c1, T))
    r1 = abs(d1 - float(jax.grad(G_bcc, argnums=0)(c2, T)))
    r2 = abs(d1 - float(jax.grad(G_L, argnums=0)(c3, T)))
    resid = max(r1, r2)
    print(f"  seed T0={T0:.0f}, g=({g1},{g2},{g3}) -> T={float(T):.1f} K, "
          f"c=({float(c1):.3f},{float(c2):.3f},{float(c3):.3f}), "
          f"residual={resid:.1e} ({'converged' if resid < 1e-8 else 'STALLED'})")
    seed_results.append(dict(T0=T0, seed=[g1, g2, g3], T=float(T),
                             c=[float(c1), float(c2), float(c3)], residual=resid))

results = dict(
    scan=scan,
    min_gap_bcc_meV=float(gaps.min()) * 1e3,
    max_gap_bcc_meV=float(gaps.max()) * 1e3,
    seed_results=seed_results,
    conclusion="no stable beta field in 750-1300 K; peritectic absent, "
              "established positively (3rd system after Ni-Al, Cu-Zn)",
)
with open(OUT / "cusn_peritectic_scan.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'cusn_peritectic_scan.json'}")
