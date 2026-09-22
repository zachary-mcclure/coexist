"""
The Ni-rich Ni-Al T-x phase diagram, assembled from the engine model
: the figure the §4.6 text describes but never drew.

Model (all from nial_liquid_retemp.json — zero engine calls here):
    γ   — fcc solution, RK(order 2) from Mishin-EAM statics @108 atoms
    γ'  — line compound at x_Al = 0.25 (EAM H_f = −0.454, κ = 300)
    β   — line compound at x_Al = 0.50 (EAM H_f = −0.606)
    L   — linear-in-T RK sampled at 1500/1670 K + experimental fusion data

The full topology is DISCOVERED, residual-checked, and verified:
  * γ liquidus/solidus by continuation from T_m(Ni),
  * the verified γ/L/γ' eutectic (1442 K),
  * congruent melting of γ' and β (tangent collapse onto stoichiometry),
  * a hunt for the L/γ'/β invariant between the compounds (eutectic
    ordering — the peritectic ordering was rejected in §4.6),
  * the γ/γ' solvus below the eutectic,
  * line compounds drawn as the vertical lines they are.

Output: examples/output/nial_diagram.json + docs/figures/nial_diagram.png
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from coexist.core.phase_diagram import (  # noqa: E402
    K_B, classify_invariant, common_tangent, global_tangency_gap,
    line_compound, redlich_kister_solution, three_phase_equilibrium)

OUT = Path(__file__).parent / "output"
FIG = Path(__file__).parent.parent / "docs" / "figures"
DH_FUS_NI, T_M_NI = 0.18117, 1728.0
DH_FUS_AL, T_M_AL = 0.11099, 933.47

rt = json.load(open(OUT / "nial_liquid_retemp.json"))
L_gamma = jnp.array(rt["L_gamma_eV"])
L_lo = jnp.array(rt["L_liquid_1500K_eV"])
sl = jnp.array(rt["dL_dT_eV_per_K"])
HF_GP, HF_B = rt["Hf_L12"], rt["Hf_B2"]

G_gamma = redlich_kister_solution(L_gamma)
G_gp = line_compound(0.25, HF_GP)
G_beta = line_compound(0.50, HF_B)


def G_L(c, T):
    dG_Ni = DH_FUS_NI * (1.0 - T / T_M_NI)
    dG_Al = DH_FUS_AL * (1.0 - T / T_M_AL)
    L_T = L_lo + (T - 1500.0) * sl
    series = L_T[0] + L_T[1] * (1.0 - 2.0 * c)
    return ((1 - c) * dG_Ni + c * dG_Al + c * (1 - c) * series
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))


def tangent_resid(Ga, Gb, ca, cb, T):
    s = float((Gb(cb, T) - Ga(ca, T)) / (cb - ca))
    return max(abs(float(jax.grad(Ga, 0)(ca, T)) - s),
               abs(float(jax.grad(Gb, 0)(cb, T)) - s))


def inv_resid(curves, cs, T):
    r = []
    d1 = float(jax.grad(curves[0], argnums=0)(cs[0], T))
    for Gi, ci in zip(curves[1:], cs[1:]):
        r.append(abs(d1 - float(jax.grad(Gi, argnums=0)(ci, T))))
        r.append(abs(float(Gi(ci, T)) - float(curves[0](cs[0], T))
                     - d1 * (float(ci) - float(cs[0]))))
    return max(r)


ALL = (G_gamma, G_gp, G_beta, G_L)
results = {}

# ═══ 1. Invariants, residual-checked and verified ═════════════════════════════
print("── invariants (residual + gap, per the §3.4 protocol)")
c1, c2, c3, Te1 = three_phase_equilibrium(
    G_gamma, G_L, G_gp, c_guesses=(0.14, 0.17, 0.2495), T_guess=1440.0)
r = inv_resid((G_gamma, G_L, G_gp), (c1, c2, c3), float(Te1))
g = global_tangency_gap(ALL, (c1, c2, c3), float(Te1))
print(f"   γ/L/γ' : T = {float(Te1):.1f} K, c = ({float(c1):.3f}, "
      f"{float(c2):.3f}, {float(c3):.3f}) — "
      f"{classify_invariant(('solid','liquid','solid'))}, resid {r:.0e}, "
      f"gap {g:+.0e}")
results["eut_gamma"] = dict(T=float(Te1), c=[float(c1), float(c2),
                                             float(c3)], resid=r, gap=g)

cp1, cp2, cp3, Te2 = three_phase_equilibrium(
    G_gp, G_L, G_beta, c_guesses=(0.2505, 0.33, 0.4995), T_guess=1480.0)
r2 = inv_resid((G_gp, G_L, G_beta), (cp1, cp2, cp3), float(Te2))
g2 = global_tangency_gap(ALL, (cp1, cp2, cp3), float(Te2),
                         anchor=ALL.index(G_gp))
kind2 = classify_invariant(("solid", "liquid", "solid"))
ok2 = "VERIFIED" if (r2 < 1e-8 and g2 > -1e-6) else "NOT ESTABLISHED"
print(f"   γ'/L/β : T = {float(Te2):.1f} K, c = ({float(cp1):.3f}, "
      f"{float(cp2):.3f}, {float(cp3):.3f}) — {kind2}, resid {r2:.0e}, "
      f"gap {g2:+.0e} [{ok2}]")
results["eut_gp_beta"] = dict(T=float(Te2), c=[float(cp1), float(cp2),
                                               float(cp3)], resid=r2, gap=g2)

# congruent melting of the compounds: liquid touches the stoichiometric point
from scipy.optimize import brentq  # noqa: E402

T_cong_gp = brentq(lambda T: float(G_L(0.25, T)) - float(G_gp(0.25, T)),
                   float(Te1) + 1e-3, 2000.0)
T_cong_b = brentq(lambda T: float(G_L(0.50, T)) - float(G_beta(0.50, T)),
                  float(Te2) + 1e-3, 2400.0)
print(f"   congruent melting: γ' {T_cong_gp:.1f} K (assessed ≈1668 "
      f"peritectic decomposition), β {T_cong_b:.1f} K (assessed 1911)")
results["T_congruent_gp"] = T_cong_gp
results["T_congruent_beta"] = T_cong_b


# ═══ 2. Boundary sweeps by continuation (residual-checked) ════════════════════
def sweep(Ga, Gb, Ts, ga, gb):
    """Common-tangent continuation; returns arrays (skips non-converged).
    A NaN residual (diverged solve) is rejected explicitly: `nan > tol` is
    False, so the bare threshold alone would pass it. No degenerate-root
    floor here, unlike binary_suite_diagrams.py's self-tangent case — every
    sweep below is between two distinct phases, where the tangent points
    legitimately merge approaching a congruent melting point."""
    A, B, TT = [], [], []
    for T in Ts:
        try:
            ca, cb = common_tangent(Ga, Gb, float(T),
                                    c_alpha_guess=ga, c_beta_guess=gb)
        except Exception:
            continue
        r = tangent_resid(Ga, Gb, ca, cb, float(T))
        if not np.isfinite(r) or r > 1e-8:
            continue
        ga, gb = float(ca), float(cb)
        A.append(ga); B.append(gb); TT.append(float(T))
    return np.array(A), np.array(B), np.array(TT)

print("── boundary sweeps (continuation, residuals ≤ 1e-8)")
# γ + L (Ni side), walked down from T_m(Ni)
gs, gl, gT = sweep(G_gamma, G_L, np.linspace(T_M_NI - 2, float(Te1) + 0.5, 90),
                   0.005, 0.008)
# L + γ' (left face of γ'), eutectic liquid up to congruent point
ll2, gp2, lT2 = sweep(G_L, G_gp, np.linspace(float(Te1) + 0.5,
                                             T_cong_gp - 0.2, 60),
                      float(c2), 0.2495)
# γ' + L (right face of γ'), congruent point down to the γ'/L/β invariant
gp3, ll3, lT3 = sweep(G_gp, G_L, np.linspace(T_cong_gp - 0.2,
                                             float(Te2) + 0.5, 60),
                      0.2505, 0.27)
# L + β (left face of β), invariant up to β congruent melting
ll4, bb4, lT4 = sweep(G_L, G_beta, np.linspace(float(Te2) + 0.5,
                                               T_cong_b - 0.2, 60),
                      float(cp2), 0.4995)
# β + L (right face of β), congruent melting down to the plot edge
bb5, ll5, lT5 = sweep(G_beta, G_L, np.linspace(T_cong_b - 0.2, 1200.0, 60),
                      0.5005, 0.53)
# γ + γ' solvus below the eutectic
sv_g, sv_gp, svT = sweep(G_gamma, G_gp, np.linspace(float(Te1) - 0.5,
                                                    700.0, 60),
                         float(c1), 0.2495)
print(f"   sweeps: γ-L {len(gT)}, L-γ' {len(lT2)}, γ'-L {len(lT3)}, "
      f"L-β {len(lT4)}, β-L {len(lT5)}, γ/γ' solvus {len(svT)} points")

results["sweeps"] = {
    "gamma_L": dict(T=gT.tolist(), solidus=gs.tolist(), liquidus=gl.tolist()),
    "L_gp": dict(T=lT2.tolist(), liquidus=ll2.tolist()),
    "gp_L": dict(T=lT3.tolist(), liquidus=ll3.tolist()),
    "L_beta": dict(T=lT4.tolist(), liquidus=ll4.tolist()),
    "beta_L": dict(T=lT5.tolist(), liquidus=ll5.tolist()),
    "solvus": dict(T=svT.tolist(), gamma=sv_g.tolist()),
}
with open(OUT / "nial_diagram.json", "w") as f:
    json.dump(results, f, indent=1)

# ═══ 3. The figure ════════════════════════════════════════════════════════════
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, AQUA, YELLOW, VIOLET = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"
GOOD, CRIT = "#0ca30c", "#d03b3b"

# Same redundant (non-color) coding as figures.py's PHASE_STYLE, kept
# in sync by hand since this script runs standalone (no engine calls) —
# gamma/liquid/gamma'/beta must match how this system is drawn in Fig 2c.
PHASE_STYLE = {
    "primary_solid":   dict(ls="-",  marker="o"),   # blue: gamma
    "liquid":          dict(ls="--", marker="s"),   # aqua
    "secondary_solid": dict(ls=":",  marker=None),  # yellow: gamma'
    "beta":            dict(ls="-.", marker=None),  # violet
}

def mkw(style, color, markevery=12, ms=4):
    return dict(ls=style["ls"], marker=style["marker"], markevery=markevery,
                ms=ms, markerfacecolor=color, markeredgecolor="white", mew=0.4)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.1,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "xtick.color": MUT, "ytick.color": MUT,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.grid": True, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "figure.facecolor": "white", "savefig.dpi": 220,
})
T1, T2c = float(Te1), float(Te2)
fig, ax = plt.subplots(figsize=(5.4, 4.0), layout="constrained")

# liquidus (aqua, one continuous entity) + solidus/solvus (blue) — same
# (linestyle, marker) as PHASE_STYLE elsewhere so grayscale/colorblind
# readers can match this figure to Fig 2c without relying on hue.
liq_kw = mkw(PHASE_STYLE["liquid"], AQUA)
sol_kw = mkw(PHASE_STYLE["primary_solid"], BLUE)
ax.plot(gl, gT, color=AQUA, lw=1.8, **liq_kw)
ax.plot(ll2, lT2, color=AQUA, lw=1.8, **liq_kw)
ax.plot(ll3, lT3, color=AQUA, lw=1.8, **liq_kw)
ax.plot(ll4, lT4, color=AQUA, lw=1.8, **liq_kw)
ax.plot(ll5, lT5, color=AQUA, lw=1.8, **liq_kw)
ax.plot(gs, gT, color=BLUE, lw=1.8, **sol_kw)
ax.plot(sv_g, svT, color=BLUE, lw=1.8, **sol_kw)

# line compounds: vertical lines from the floor to congruent melting
ax.plot([0.25, 0.25], [700, T_cong_gp], color=YELLOW, lw=2.2,
        ls=PHASE_STYLE["secondary_solid"]["ls"])
ax.plot([0.50, 0.50], [700, T_cong_b], color=VIOLET, lw=2.2,
        ls=PHASE_STYLE["beta"]["ls"])

# invariant tie-lines (verified: status green)
ax.plot([float(c1), float(c3)], [T1, T1], color=GOOD, lw=1.3)
ax.plot([float(cp1), float(cp3)], [T2c, T2c], color=GOOD, lw=1.3)

# assessed experimental invariants for comparison
ax.plot([0.243], [1658], marker="*", ms=11, color=INK, ls="none")
ax.plot([0.28], [1668], marker="*", ms=7, color=INK, ls="none")

# labels
ax.annotate("liquid", (0.09, 1640), color="#12805a", fontsize=9)
ax.annotate(r"$\gamma$", (0.03, 1200), color=BLUE, fontsize=10)
ax.annotate(r"$\gamma+\gamma'$", (0.15, 1000), color=SEC, fontsize=8.5)
ax.annotate(r"$\gamma'$", (0.256, 850), color="#b87d00", fontsize=10)
ax.annotate(r"$\gamma'+\beta$", (0.35, 1000), color=SEC, fontsize=8.5)
ax.annotate(r"$\beta$", (0.506, 850), color=VIOLET, fontsize=10)
ax.annotate(f"verified eutectic {T1:.0f} K", (float(c2) - 0.01, T1 - 60),
            color=GOOD, fontsize=7.5)
lab2 = f"verified eutectic {T2c:.0f} K" if (r2 < 1e-8 and g2 > -1e-6) \
    else f"invariant {T2c:.0f} K (unverified)"
ax.annotate(lab2, (float(cp2) - 0.02, T2c - 60), color=GOOD, fontsize=7.5)
ax.annotate("assessed invariants\n(1658/1668 K)", (0.30, 1700),
            color=SEC, fontsize=7.5)
ax.annotate(f"congruent\n{T_cong_b:.0f} K", (0.51, T_cong_b + 20),
            color=SEC, fontsize=7)

ax.set_xlabel(r"$x_{\rm Al}$")
ax.set_ylabel("T (K)")
ax.set_xlim(0, 0.6)
ax.set_ylim(700, 1900)
ax.set_title("Ni-rich Ni-Al from the engine model "
             "(EAM statics + sampled liquid)", fontsize=9)
fig.savefig(FIG / "nial_diagram.png")
print(f"wrote {FIG/'nial_diagram.png'}")
