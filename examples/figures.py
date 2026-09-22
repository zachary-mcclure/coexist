"""
Manuscript figures — all data from examples/output/*.json + pure-JAX diagram
re-sweeps (no engine calls). Style per the dataviz method: validated
categorical palette (blue/aqua/yellow/violet, fixed entity order), direct
labels on every series (relief rule for aqua/yellow on white), status
green/red only for verified/rejected and always with text, recessive
grid/axes, one axis per panel.

Entities → colors (stable across figures):
  engines:  EAM=blue, MACE=aqua, EMT=yellow;  experiment = ink markers
  phases:   solid₁=blue, liquid=aqua, solid₂=yellow, β=violet
  verdicts: verified=status-good, rejected impostor=status-critical

Outputs: docs/figures/{engine_uq,topology,inverse,attribution}.png
Runtime ~2 min (diagram sweeps are host-side continuation loops).
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
    K_B, eutectic_diagram, line_compound, liquid_solution,
    redlich_kister_solution, regular_solution, three_phase_equilibrium)

OUT = Path(__file__).parent / "output"
FIG = Path(__file__).parent.parent / "docs" / "figures"
FIG.mkdir(exist_ok=True)

# ── palette (validated reference set; see dataviz references/palette.md) ──────
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, AQUA, YELLOW, VIOLET = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"
GOOD, CRIT = "#0ca30c", "#d03b3b"

# Redundant (non-color) coding: fixed (linestyle, marker) per entity, stable
# across figures. Color stays primary/attractive; these are secondary so
# multi-series panels still separate under grayscale printing or for
# colorblind readers, without adding a second legend to consult.
ENGINE_STYLE = {
    "EAM":       dict(ls="-",  marker="o"),
    "MACE-MP-0": dict(ls="--", marker="s"),
    "EMT":       dict(ls=":",  marker="^"),
}
PHASE_STYLE = {
    "primary_solid":   dict(ls="-",  marker="o"),   # blue: solid/fcc/gamma
    "liquid":          dict(ls="--", marker="s"),   # aqua
    "secondary_solid": dict(ls=":",  marker="^"),   # yellow: dia/gamma-prime
    "beta":            dict(ls="-.", marker="D"),   # violet
}

def mkw(style, color, markevery=40, ms=4):
    """Expand a PHASE/ENGINE_STYLE entry into ax.plot marker kwargs."""
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
    "legend.frameon": False, "legend.fontsize": 8,
    "legend.handlelength": 2.6, "legend.handletextpad": 0.6,
    "figure.facecolor": "white", "savefig.dpi": 220,
})

def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

T_M_AG, DH_AG = 1234.93, 0.11691
T_M_CU, DH_CU = 1357.77, 0.13742

camp = json.load(open(OUT / "eam_campaign.json"))
liq = json.load(open(OUT / "liquid_omega.json"))
inv = json.load(open(OUT / "inverse_capstone.json"))
alsi = json.load(open(OUT / "alsi_rung.json"))
nial = json.load(open(OUT / "nial_liquid_retemp.json"))  # model
topo = json.load(open(OUT / "topology_ladder.json"))


def agcu_liquid(om_L):
    return liquid_solution(om_L, dH_fus_A=DH_AG, T_m_A=T_M_AG,
                           dH_fus_B=DH_CU, T_m_B=T_M_CU)


def plot_diagram(ax, d, color, lw=1.8, label=None, alpha=1.0, solvus=True,
                 ls="-", marker=None, markevery=6):
    """Liquidus+solidus(+solvus) of a eutectic_diagram dict, one color.

    ls/marker are the redundant channel (see ENGINE_STYLE) so Fig 1's
    right panel and Fig 3's left panel still separate under grayscale.
    """
    keys = [("T_A", "liquidus_A"), ("T_A", "solidus_A"),
            ("T_B", "liquidus_B"), ("T_B", "solidus_B")]
    if solvus:
        keys += [("T_solvus", "solvus_A"), ("T_solvus", "solvus_B")]
    for Tk, ck in keys:
        ax.plot(d[ck], d[Tk], color=color, lw=lw, alpha=alpha, ls=ls,
                marker=marker, markevery=markevery, ms=4,
                markerfacecolor=color, markeredgecolor="white", mew=0.4)
    ax.plot([d["c_alpha_e"], d["c_beta_e"]], [d["T_e"]] * 2,
            color=color, lw=lw * 0.7, alpha=alpha, ls=ls)
    if label:
        ax.plot([], [], color=color, lw=lw, alpha=alpha, ls=ls,
                marker=marker, ms=4, markerfacecolor=color,
                markeredgecolor="white", mew=0.4, label=label)


# ═══ Figure 1: fully-engine diagram + UQ band | engine-fidelity overlay ═══════
print("fig 1: engine + UQ …")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0), layout="constrained")

OM_L_S = liq["omega_L_1150K_eV"][0]
d_full = eutectic_diagram(
    redlich_kister_solution(jnp.array(camp["AgCu"]["L_RK_eV"])),
    agcu_liquid(OM_L_S), T_m_A=T_M_AG, T_m_B=T_M_CU, n_T=30)
plot_diagram(a1, d_full, BLUE, label="EAM solid + EAM-MD liquid")
Tb = np.array(liq["liquidus_T"])
cb = np.array(liq["liquidus_c"])
bb = np.array(liq["liquidus_band"])
# The delta-method band is a first-order expansion of the liquidus about
# Omega_L; within ~45 K of T_e the boundary's own slope in composition
# diverges approaching the invariant and the linearization exceeds the
# physical [0,1] range. Clip the drawn band there and say so explicitly
# (Sec. 4.3) rather than let a clipped fill silently look like a tight band.
near_Te = Tb < (Tb.min() + 45.0)
lo = np.clip(cb - bb, 0.0, 1.0)
hi = np.clip(cb + bb, 0.0, 1.0)
a1.fill_betweenx(Tb[~near_Te], lo[~near_Te], hi[~near_Te], color=BLUE,
                 alpha=0.18, lw=0, label=r"$\pm1\sigma$ from sampled $\Omega_L$")
a1.fill_betweenx(Tb[near_Te], lo[near_Te], hi[near_Te], color=BLUE,
                 alpha=0.18, lw=0, hatch="////", edgecolor=MUT)
a1.annotate("linearization\nexceeds $[0,1]$ here", (0.62, Tb[near_Te].max() + 8),
            fontsize=6.5, color=MUT, ha="left")
a1.plot([0.399], [1052], marker="*", ms=11, color=INK, ls="none",
        label="experiment")
a1.plot([0.141, 0.951], [1052, 1052], marker="*", ms=7, color=INK,
        ls="none")
a1.annotate(f"$T_e = {liq['T_e_K']:.0f}\\pm{liq['T_e_sigma_K']:.0f}$ K",
            (0.43, 990), fontsize=8, color=SEC)
a1.set_xlabel(r"$x_{\rm Cu}$"); a1.set_ylabel("T (K)")
a1.set_title("Fully engine-derived Ag-Cu + UQ", fontsize=9)
a1.set_xlim(0, 1); a1.set_ylim(600, 1400)
a1.legend(loc="lower left", fontsize=7)
despine(a1)

OM_L_CAL = 0.1555
mace_r = json.load(open(OUT / "mace_rung.json"))
for om, col, name in [(0.1614, YELLOW, "EMT"),
                      (mace_r["AgCu"]["omega_eV"], AQUA, "MACE-MP-0")]:
    d = eutectic_diagram(regular_solution(om), agcu_liquid(OM_L_CAL),
                         T_m_A=T_M_AG, T_m_B=T_M_CU, n_T=30)
    # EMT@108: its miscibility gap closes above the plot range (T_c=490 K),
    # so the continuation "solvus" is an artifact — draw boundaries only.
    plot_diagram(a2, d, col, lw=1.5, label=name, solvus=(name != "EMT"),
                 **ENGINE_STYLE[name])
d_eam = eutectic_diagram(
    redlich_kister_solution(jnp.array(camp["AgCu"]["L_RK_eV"])),
    agcu_liquid(OM_L_CAL), T_m_A=T_M_AG, T_m_B=T_M_CU, n_T=30)
plot_diagram(a2, d_eam, BLUE, label="EAM", **ENGINE_STYLE["EAM"])
a2.plot([0.399], [1052], marker="*", ms=11, color=INK, ls="none",
        label="experiment")
a2.annotate("EMT", (0.55, 1290), color=YELLOW, fontsize=8, weight="bold")
a2.annotate("EAM", (0.05, 930), color=BLUE, fontsize=8, weight="bold")
a2.annotate("MACE", (0.75, 980), color=AQUA, fontsize=8, weight="bold")
a2.set_xlabel(r"$x_{\rm Cu}$")
a2.set_title("Engine fidelity (same CALPHAD liquid)", fontsize=9)
a2.set_xlim(0, 1); a2.set_ylim(600, 1400)
a2.legend(loc="lower left", fontsize=7)
despine(a2)
fig.savefig(FIG / "engine_uq.png")
plt.close(fig)

# ═══ Figure 1b: fidelity strip — Table 1 at a glance ══════════════════════════
print("fig 1b: fidelity strip …")
fig, ax = plt.subplots(figsize=(4.0, 2.0), layout="constrained")
rows = [
    ("EMT",   1240.0, 0.308, YELLOW),
    ("EAM",   1063.8, 0.009, BLUE),
    ("MACE-MP-0", 1053.8, 0.012, AQUA),
    ("CALPHAD\nassessment", 1052.0, None, INK),
]
ys = np.arange(len(rows))[::-1]
for (name, Te, rms, col), y in zip(rows, ys):
    size = 60 + 2200 * rms if rms is not None else 90
    marker = "o" if rms is not None else "*"
    ax.scatter([Te], [y], s=size, color=col, alpha=0.85,
               marker=marker, zorder=3,
               edgecolor="white", linewidth=0.6)
    lab = f"{Te:.0f} K" + (f"  (RMS {rms:.3f})" if rms is not None else "  (ref.)")
    ax.annotate(lab, (Te, y), xytext=(8, 0), textcoords="offset points",
                fontsize=7, va="center", color=SEC)
ax.axvline(1052.0, color=MUT, lw=0.8, ls=":", zorder=1)
ax.annotate("exp.\n1052 K", (1052.0, len(rows) - 0.3), fontsize=6.5,
            color=MUT, ha="center")
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_xlabel(r"$T_e$ (K)  —  marker area $\propto$ liquidus RMS", fontsize=8)
ax.set_xlim(1030, 1260)
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.grid(axis="x", alpha=0.5)
ax.grid(axis="y", visible=False)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.savefig(FIG / "fidelity_strip.png")
plt.close(fig)

# ═══ Figure 2: topology gallery ═══════════════════════════════════════════════
print("fig 2: topology …")
fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(6.6, 2.7),
                                 layout="constrained")
cs = np.linspace(1e-4, 1 - 1e-4, 400)

# — Al-Pb: verified monotectic + rejected impostor —
# The impostor root is seed-dependent: under the parameter set of
# tests/test_topology.py (Ω_s 1% lower, rounded fusion data) seed
# (0.005, 0.2, 0.97) converges to the documented stationary impostor
# 103 K above the true monotectic. Use that set for the whole panel so
# both roots come from ONE consistent free-energy model.
G_s = regular_solution(0.9)
G_l = liquid_solution(0.27, dH_fus_A=0.111, T_m_A=933.5,
                      dH_fus_B=0.049, T_m_B=600.6)
c1, c2, c3, Tm = three_phase_equilibrium(G_s, G_l, G_l,
                                         c_guesses=(0.001, 0.05, 0.95),
                                         T_guess=900.0)
s1, s2, s3, Ts = three_phase_equilibrium(G_s, G_l, G_l,
                                         c_guesses=(0.005, 0.2, 0.97),
                                         T_guess=950.0)
Tm_f, Ts_f = float(Tm), float(Ts)
Gs_v = [float(G_s(jnp.float64(c), Tm_f)) for c in cs]
Gl_v = [float(G_l(jnp.float64(c), Tm_f)) for c in cs]
b1.plot(cs, Gs_v, color=BLUE, lw=1.8, **mkw(PHASE_STYLE["primary_solid"], BLUE))
b1.plot(cs, Gl_v, color=AQUA, lw=1.8, **mkw(PHASE_STYLE["liquid"], AQUA))
mu = float(jax.grad(G_s, argnums=0)(c1, jnp.float64(Tm_f)))
g0 = float(G_s(c1, jnp.float64(Tm_f)))
b1.plot(cs, g0 + mu * (cs - float(c1)), color=GOOD, lw=1.3)
b1.annotate(f"verified {Tm_f:.0f} K", (0.30, -0.045), color=GOOD,
            fontsize=7.5, ha="center")
mu_s = float(jax.grad(G_s, argnums=0)(s1, jnp.float64(Ts_f)))
g0_s = float(G_s(s1, jnp.float64(Ts_f)))
Gl_i = np.array([float(G_l(jnp.float64(c), Ts_f)) for c in cs])
b1.plot(cs, Gl_i, color=AQUA, lw=1.0, alpha=0.35)
line_i = g0_s + mu_s * (cs - float(s1))
b1.plot(cs, line_i, color=CRIT, lw=1.3, ls="--")
i_cut = np.argmin(Gl_i - line_i)
b1.annotate("", xy=(cs[i_cut], Gl_i[i_cut]),
            xytext=(cs[i_cut] - 0.02, Gl_i[i_cut] - 0.030),
            arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.0))
b1.annotate(f"impostor {Ts_f:.0f} K cuts the\nfaint liquid — rejected",
            (0.50, -0.072), color=CRIT, fontsize=7.5, ha="center")
b1.annotate("solid", (0.10, 0.055), color=BLUE, fontsize=8)
b1.annotate("liquid", (0.80, -0.026), color=AQUA, fontsize=8)
b1.set_xlabel(r"$x_{\rm Pb}$"); b1.set_ylabel("G (eV/atom)")
b1.set_title("Al-Pb monotectic", fontsize=9)
b1.set_ylim(-0.09, 0.09)
despine(b1)

# — zoomed inset at the crossing: the −1.2e-2 eV violation, legible at print size —
c_cut = cs[i_cut]
zwin = (max(1e-4, c_cut - 0.06), min(1 - 1e-4, c_cut + 0.06))
iz = (cs >= zwin[0]) & (cs <= zwin[1])
axins = b1.inset_axes([0.46, 0.58, 0.42, 0.38])
axins.plot(cs[iz], Gl_i[iz], color=AQUA, lw=1.4)
axins.plot(cs[iz], line_i[iz], color=CRIT, lw=1.6, ls="--")
gap_here = float((Gl_i[i_cut] - line_i[i_cut]))
axins.plot([c_cut], [Gl_i[i_cut]], marker="o", ms=4, color=CRIT)
axins.annotate(f"${gap_here*1e3:+.1f}$ meV", (c_cut, Gl_i[i_cut]),
               xytext=(0, 8), textcoords="offset points", fontsize=6,
               color=CRIT, ha="center", va="bottom")
axins.set_xticks([]); axins.set_yticks([])
for s in axins.spines.values():
    s.set_color(MUT); s.set_linewidth(0.7)
b1.indicate_inset_zoom(axins, edgecolor=MUT, lw=0.6)

# — Al-Si two-lattice at T_e —
Te = alsi["T_e_K"]
om_f, om_d = alsi["omega_fcc_eV"], alsi["omega_dia_eV"]
dSi, dAl = (alsi["E_promotion_Si_fcc_minus_dia_eV"],
            alsi["E_promotion_Al_dia_minus_fcc_eV"])
ideal = lambda c, T: K_B * T * (c * np.log(c) + (1 - c) * np.log(1 - c))
Gf = cs * dSi + om_f * cs * (1 - cs) + ideal(cs, Te)
Gd = (1 - cs) * dAl + om_d * cs * (1 - cs) + ideal(cs, Te)
Gl2 = ((1 - cs) * 0.11099 * (1 - Te / 933.47)
       + cs * 0.52040 * (1 - Te / 1687.0) + ideal(cs, Te))
b2.plot(cs, Gf, color=BLUE, lw=1.8, **mkw(PHASE_STYLE["primary_solid"], BLUE))
b2.plot(cs, Gd, color=YELLOW, lw=1.8, **mkw(PHASE_STYLE["secondary_solid"], YELLOW))
b2.plot(cs, Gl2, color=AQUA, lw=1.8, **mkw(PHASE_STYLE["liquid"], AQUA))
i1 = np.argmin(np.abs(cs - alsi["fcc_solvus"]))
i3 = np.argmin(np.abs(cs - alsi["si_limb"]))
b2.plot([cs[i1], cs[i3]], [Gf[i1], Gd[i3]], color=GOOD, lw=1.3)
b2.annotate(f"verified eutectic {Te:.0f} K", (0.5, -0.045), color=GOOD,
            fontsize=7.5, ha="center")
b2.annotate("fcc Al(Si)", (0.03, 0.20), color=BLUE, fontsize=8)
b2.annotate("diamond Si(Al)", (0.42, 0.33), color="#b87d00", fontsize=8)
b2.annotate("liquid", (0.62, 0.035), color=AQUA, fontsize=8)
b2.set_xlabel(r"$x_{\rm Si}$")
b2.set_title("Al-Si (two lattices, MACE)", fontsize=9)
b2.set_ylim(-0.07, 0.5)
despine(b2)

# — Ni-Al at the verified eutectic —
Tn = nial["gamma/L/gamma'"]["T"]
Lg = np.array(nial["L_gamma_eV"])
Ll = np.array(nial["L_liquid_eV"])
csn = np.linspace(1e-4, 0.6, 400)
rk = lambda L, c: c * (1 - c) * sum(
    L[k] * (1 - 2 * c) ** k for k in range(len(L)))
Gg = rk(Lg, csn) + ideal(csn, Tn)
Gln = ((1 - csn) * 0.18117 * (1 - Tn / 1728.0)
       + csn * 0.11099 * (1 - Tn / 933.47) + rk(Ll, csn) + ideal(csn, Tn))
Ggp = nial["Hf_L12"] + 300.0 * (csn - 0.25) ** 2
Gb = nial["Hf_B2"] + 300.0 * (csn - 0.50) ** 2
b3.plot(csn, Gg, color=BLUE, lw=1.8, **mkw(PHASE_STYLE["primary_solid"], BLUE))
b3.plot(csn, Gln, color=AQUA, lw=1.8, **mkw(PHASE_STYLE["liquid"], AQUA))
b3.plot(csn, np.where(np.abs(csn - 0.25) < 0.06, Ggp, np.nan),
        color=YELLOW, lw=1.8,
        **mkw(PHASE_STYLE["secondary_solid"], YELLOW, markevery=8))
b3.plot(csn, np.where(np.abs(csn - 0.50) < 0.06, Gb, np.nan),
        color=VIOLET, lw=1.8, **mkw(PHASE_STYLE["beta"], VIOLET, markevery=8))
cc = nial["gamma/L/gamma'"]["c"]
iA = np.argmin(np.abs(csn - cc[0]))
slope = (nial["Hf_L12"] - Gg[iA]) / (0.25 - cc[0])
b3.plot(csn, Gg[iA] + slope * (csn - cc[0]), color=GOOD, lw=1.2)
b3.annotate(f"verified eutectic {Tn:.0f} K", (0.15, -0.63), color=GOOD,
            fontsize=7.5, ha="center")
b3.annotate("assessed peritectic: no root —\nmodel forms a 2nd eutectic"
            " (1419 K)", (0.44, -0.13), color=SEC, fontsize=7,
            ha="center")
b3.annotate(r"$\gamma$", (0.05, -0.13), color=BLUE, fontsize=9)
b3.annotate("liquid", (0.10, -0.32), color=AQUA, fontsize=8)
b3.annotate(r"$\gamma'$", (0.245, -0.50), color="#b87d00", fontsize=9)
b3.annotate(r"$\beta$", (0.50, -0.66), color=VIOLET, fontsize=9)
b3.set_xlabel(r"$x_{\rm Al}$")
b3.set_title("Ni-Al (EAM + sampled liquid)", fontsize=9)
b3.set_ylim(-0.75, 0.1)
despine(b3)
fig.savefig(FIG / "topology.png")
plt.close(fig)

# ═══ Figure 3: inverse capstone ═══════════════════════════════════════════════
print("fig 3: inverse …")
fig, (d1, d2) = plt.subplots(1, 2, figsize=(6.6, 3.0),
                             layout="constrained",
                             gridspec_kw={"width_ratios": [1.5, 1]})
plot_diagram(d1, d_eam, BLUE, lw=1.5, alpha=0.75, ls="--",
             label="raw EAM + lit. liquid")
d_star = inv["diagram_at_theta_star"]
plot_diagram(d1, d_star, AQUA, label=r"fitted $\theta^*$",
             marker="s", markevery=6)
d1.plot([0.399, 0.141], [1052, 1052], marker="*", ms=10, color=INK,
        ls="none", label="targets ($T_e$, $c_\\alpha$) + exp $x_e$")
d1.annotate("raw", (0.06, 1105), color=BLUE, fontsize=8, weight="bold")
d1.annotate(r"fitted $\theta^*$", (0.30, 890), color="#12805a",
            fontsize=8, weight="bold")
d1.set_xlabel(r"$x_{\rm Cu}$"); d1.set_ylabel("T (K)")
d1.set_title("Inverse fit through the eutectic solve", fontsize=9)
d1.set_xlim(0, 1); d1.set_ylim(600, 1400)
d1.legend(loc="lower left", fontsize=7)
despine(d1)

vals = [inv.get("omega_s_raw_L0_meV", 326), inv["omega_s_star_L0_meV"], 340]
names = ["raw EAM\n(0 K)", "fitted\n$\\lambda^*$", "CALPHAD\n(0 K)"]
cols = [BLUE, AQUA, MUT]
bars = d2.bar(names, vals, color=cols, width=0.62,
              edgecolor=INK, linewidth=1.1)
for r, h in zip(bars, ["//", "xx", ".."]):
    r.set_hatch(h)
for r, v in zip(bars, vals):
    d2.annotate(f"{v:.0f}", (r.get_x() + r.get_width() / 2, v + 6),
                ha="center", fontsize=8, color=INK)
d2.annotate("gap = effective high-$T$\nsoftening "
            f"$\\approx{round(inv['omega_s_star_L0_meV'] - inv.get('omega_s_raw_L0_meV', 326))}$ meV",
            (1.0, 400), ha="center", fontsize=7.5, color=SEC)
d2.set_ylabel(r"$L_0(\Omega_s)$ (meV/atom)")
d2.set_title("Fit lands below 0 K values", fontsize=9)
d2.set_ylim(0, 460)
d2.grid(axis="x", visible=False)
despine(d2)
fig.savefig(FIG / "inverse.png")
plt.close(fig)

# (The missing-physics attribution summary is a LaTeX table in the paper
# — Table "tab:attribution" in the Discussion — not a figure: five
# quantities in incommensurate units gain nothing from bar axes. Its
# numbers live in cuzn_quasichemical.json / cusn_quasichemical.json /
# gap_uncertainty.json / inverse_capstone.json / nial_liquid_retemp.json.)

# ═══ Figure 4b: engine-coverage matrix ═════════════════════════════════════════
# Not loaded from JSON (like Table 1's own construction summary, this is a
# hand-authored classification of what each system's construction actually
# draws on) -- every cell below is traceable to a specific sentence in the
# .tex (see the inline citation in each row). Four states only, applied
# identically across every column (no per-column special-casing):
#   E = engine     (computed from LAMMPS/MACE via this paper's machinery)
#   L = literature (an external assessed/experimental value)
#   I = ideal / not modeled (assumed zero -- whether because the system's
#       construction never attempted this term, or because "ideal" is
#       ITSELF the finding a Figure 4 gradient measures the cost of)
#   N = not applicable (no such physical term for this system, e.g. no
#       liquid phase at all in Cu-Ni/Zr-Nb, no line compound outside Ni-Al)
print("fig 4b: engine-coverage matrix …")
LIT_COLOR = "#5b7fb5"
STATE_COLOR = {"E": GOOD, "L": LIT_COLOR, "I": "#fab219", "N": "#f2f1ec"}
STATE_SYMBOL = {"E": "●", "L": "■", "I": "▲", "N": "–"}
STATE_TXT = {"E": "white", "L": "white", "I": INK, "N": MUT}
STATE_LABEL = {"E": "engine", "L": "literature",
              "I": "ideal / not modeled", "N": "not applicable"}
# Compound entropy (S_f) is not a column: only Ni-Al has a line compound at
# all, so a whole column would be four-fifths empty space for one data
# point (a real critique of the first draft) -- it is instead a row-level
# annotation on Ni-Al, alongside the other per-row notes below.
COLS = [("Solid statics", r"$\Omega_s,\,H_f$"),
       ("Liquid mixing", r"$\Omega_L$"),
       ("Vibrational", r"$\theta_D$"),
       ("Config. SRO", "quasichemical")]
ROWS = [
    ("Ag-Cu", "eutectic", ["E", "E", "E", "I"], False),
    ("Cu-Ni", "miscibility gap", ["E", "N", "I", "I"], False),
    ("Ni-Al", "eutectic", ["E", "E", "E", "I"], True),
    ("Al-Pb", "monotectic", ["E", "L", "I", "I"], False),
    ("Al-Si", "eutectic", ["E", "I", "I", "I"], True),
    ("Cu-Zn", "peritectic (absent)", ["E", "I", "E", "E"], False),
    ("Cu-Sn", "peritectic (absent)", ["E", "I", "E", "I"], True),
    ("Zr-Nb", "monotectoid", ["E", "N", "E", "I"], False),
]
# Row-level notes: the 3 "ideal" classifications that are themselves a
# measured finding elsewhere in the paper (marked with a dagger on the
# system name), vs. every other amber cell, which is addressed once, in
# general, in the caption rather than row by row (see fig caption text) --
# the two situations need different remediation, not the same footnote.
ROW_NOTE = {
    "Ni-Al": "compound entropy $S_f{=}0$ by construction, the flagship gap Sec. 4.6 measures",
    "Al-Si": "ideal liquid, no literature interaction anywhere — itself the Sec. 4.5 finding",
    "Cu-Sn": "quasichemical SRO attempted; derived inapplicable, not simply unattempted (Sec. 4.8)",
}

n_rows, n_cols = len(ROWS), len(COLS)
cell_w, cell_h = 1.35, 1.0
label_w = 2.35
header_h = 1.15
notes = [(sys, ROW_NOTE[sys]) for sys, _, _, has_note in ROWS if has_note
        for sys in [sys]]
fig, ax = plt.subplots(
    figsize=(label_w + n_cols * cell_w + 0.4, 8.2),
    layout="constrained")

for j, (title, sub) in enumerate(COLS):
    xc = label_w + (j + 0.5) * cell_w
    ax.text(xc, n_rows + 0.78, title, ha="center", va="bottom",
           fontsize=7.0, weight="bold", color=INK)
    ax.text(xc, n_rows + 0.42, sub, ha="center", va="bottom",
           fontsize=7.6, color=SEC)

for i, (sys, kind, states, has_note) in enumerate(ROWS):
    y = n_rows - 1 - i
    label = sys + ("$^\\dagger$" if has_note else "")
    ax.text(label_w - 0.18, y + 0.56, label, ha="right", va="center",
           fontsize=9.5, weight="bold", color=INK)
    ax.text(label_w - 0.18, y + 0.20, kind, ha="right", va="center",
           fontsize=7.2, color=SEC, style="italic")
    for j, s in enumerate(states):
        x = label_w + j * cell_w
        ax.add_patch(plt.Rectangle((x, y), cell_w - 0.06, cell_h - 0.06,
                                   facecolor=STATE_COLOR[s], edgecolor="white",
                                   linewidth=1.6))
        ax.text(x + (cell_w - 0.06) / 2, y + (cell_h - 0.06) / 2, STATE_SYMBOL[s],
               ha="center", va="center", fontsize=12, color=STATE_TXT[s])
ax.set_xlim(0, label_w + n_cols * cell_w)
ax.set_ylim(-3.0 - 0.48 * len(notes), n_rows + 1.3)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(False)

# legend: two rows of two, symbol+color+label (redundant coding). One row
# overflowed the figure width once "not applicable" was reached (it landed
# past the right edge and clipped); a 2x2 layout keeps every label inside.
leg_y = -0.85
leg_row_gap = 0.62
for s, (lx, ly) in zip(("E", "L", "I", "N"),
                       [(0.0, leg_y), (3.9, leg_y),
                        (0.0, leg_y - leg_row_gap), (3.9, leg_y - leg_row_gap)]):
    ax.add_patch(plt.Rectangle((lx, ly), 0.4, 0.4, facecolor=STATE_COLOR[s],
                               edgecolor="white", linewidth=1.4))
    ax.text(lx + 0.2, ly + 0.2, STATE_SYMBOL[s], ha="center", va="center",
           fontsize=9, color=STATE_TXT[s])
    ax.text(lx + 0.58, ly + 0.2, STATE_LABEL[s], ha="left", va="center",
           fontsize=8, color=SEC)
y_note = leg_y - leg_row_gap - 0.55
for sys, txt in notes:
    ax.annotate(f"$\\dagger$ {sys}: {txt}", (0.0, y_note), fontsize=8.4,
               color=MUT, ha="left")
    y_note -= 0.48
ax.annotate(
    "Every other amber cell (no $\\dagger$) reflects a construction that "
    "already meets its own reported accuracy without the term,\n"
    "or an open extension named explicitly in its section (e.g. Cu-Ni's "
    "340 K MACE miss) — not a uniform omission.",
    (0.0, y_note - 0.22), fontsize=8.4, color=MUT, ha="left")
ax.set_title("Where each system's thermodynamics comes from", fontsize=11,
            weight="bold", loc="left", pad=16)
fig.savefig(FIG / "coverage.png")
plt.close(fig)

# ═══ Figure 5: ternary constructions (Gibbs triangles) ════════════════════════
print("fig 5: ternary …")
tern = json.load(open(OUT / "ternary_agcuni.json"))

def bary(xB, xC):
    """(x_Cu, x_Ni) → cartesian in the Gibbs triangle (Ag origin,
    Cu right, Ni top)."""
    return np.asarray(xB) + 0.5 * np.asarray(xC), \
        (np.sqrt(3) / 2) * np.asarray(xC)

def triangle_frame(ax, labels=("Ag", "Cu", "Ni")):
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(tri_x, tri_y, color=AXIS, lw=1.0)
    ax.annotate(labels[0], (-0.03, -0.02), ha="right", va="top",
                fontsize=9, color=INK)
    ax.annotate(labels[1], (1.03, -0.02), ha="left", va="top",
                fontsize=9, color=INK)
    ax.annotate(labels[2], (0.5, np.sqrt(3) / 2 + 0.03), ha="center",
                fontsize=9, color=INK)
    ax.set_aspect("equal")
    ax.axis("off")

fig, (t1, t2, t3) = plt.subplots(1, 3, figsize=(9.6, 3.1), layout="constrained")

# — 250 K verified tie-triangle (block-solved) —
triangle_frame(t1)
tri = json.load(open(OUT / "ternary_triangle_250K.json"))
V = [[tri[k][1], tri[k][2]] for k in ("v1", "v2", "v3")]
vx, vy = zip(*[bary(v[0], v[1]) for v in V])
t1.fill(list(vx) + [vx[0]], list(vy) + [vy[0]], color=BLUE, alpha=0.10)
t1.plot(list(vx) + [vx[0]], list(vy) + [vy[0]], color=GOOD, lw=1.6)
t1.plot(vx, vy, "o", color=BLUE, ms=6, mec="white", mew=0.8)
t1.annotate("fcc$_1$ (Ag)", bary(0.02, 0.05), color=SEC, fontsize=8)
t1.annotate("fcc$_2$", bary(0.72, 0.16), color=SEC, fontsize=8)
t1.annotate("fcc$_3$", bary(0.10, 0.72), color=SEC, fontsize=8)
gap = tri["gap_eV"]
imp = tern["triangle_lowT"]["impostor_gap"]
t1.set_title(f"250 K tie-triangle — verified ({gap:+.0e} eV)\n"
             f"block-solved; impostor rejected at ${imp:+.1f}$ eV",
             fontsize=8.5)

# — 800 K tie-line fan —
triangle_frame(t2)
al = np.array(tern["section_800K"]["alpha"])
be = np.array(tern["section_800K"]["beta"])
for a, b in zip(al[::2], be[::2]):
    ax_, ay_ = bary(*a)
    bx_, by_ = bary(*b)
    t2.plot([ax_, bx_], [by_ * 0 + ay_, by_], color=GRID, lw=0.7)
ax_, ay_ = bary(al[:, 0], al[:, 1])
bx_, by_ = bary(be[:, 0], be[:, 1])
t2.plot(ax_, ay_, color=BLUE, lw=1.8,
        **mkw(PHASE_STYLE["primary_solid"], BLUE, markevery=15))
t2.plot(bx_, by_, color=AQUA, lw=1.8,
        **mkw(PHASE_STYLE["liquid"], AQUA, markevery=15))
cx, cy = bary(*tern["section_800K"]["collapsed_limit"])
t2.plot([cx], [cy], marker="D", ms=5, color=INK, ls="none")
t2.annotate("α binodal\n(Ag corner)", bary(0.02, 0.10), color=BLUE,
            fontsize=8)
t2.annotate("β binodal", bary(0.75, 0.30), color="#12805a", fontsize=8)
t2.annotate("collapsed-triangle\nlimit", (cx + 0.03, cy - 0.02),
            color=SEC, fontsize=7.5)
t2.set_title("800 K section — verified tie-line fan\n"
             "(no three-phase field: triangle collapses)", fontsize=8.5)

# — T-continuation: the triangle shrinking toward the collapse point —
triangle_frame(t3)
sweep = json.load(open(OUT / "ternary_triangle_sweep.json"))
n = len(sweep["frames"])
for i, fr in enumerate(sweep["frames"]):
    shade = 0.15 + 0.65 * (i / (n - 1))          # dark (low T) -> light (high T)
    col = tuple((1 - shade) * np.array([0.16, 0.47, 0.84]) + shade * np.array([1, 1, 1]))
    V = [fr[k][1:] for k in ("v1", "v2", "v3")]
    vx, vy = zip(*[bary(v[0], v[1]) for v in V])
    t3.plot(list(vx) + [vx[0]], list(vy) + [vy[0]], color=col, lw=1.6)
    t3.plot(vx, vy, "o", color=col, ms=4, mec="white", mew=0.5)
    t3.annotate(f"{fr['T']:.0f} K", bary(V[1][0] + 0.02, V[1][1] - 0.01),
                color=SEC, fontsize=6.5)
t3.annotate(f"$T_c^{{\\rm CuNi}}={sweep['T_c_CuNi']:.0f}$ K", (0.88, 0.52),
            color=SEC, fontsize=7.5, ha="center")
t3.set_title("$T$-continuation, $250\\to280$ K\n"
             "triangle shrinks to the collapse point", fontsize=8.5)

fig.savefig(FIG / "ternary.png")
plt.close(fig)

# ═══ Figure 0: Methods concept schematic (adapter + implicit layer) ═══════════
print("fig 0: adapter/implicit-layer schematic …")
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

fig, ax = plt.subplots(figsize=(7.1, 2.9), layout="constrained")
ax.set_xlim(0, 11.3); ax.set_ylim(0, 4.2)
ax.axis("off")


def box(x, y, w, h, text, face, edge=INK, fs=8, tcolor=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                                facecolor=face, edgecolor=edge, linewidth=1.4, zorder=2))
    ax.annotate(text, (x + w / 2, y + h / 2), ha="center", va="center",
                fontsize=fs, color=tcolor, zorder=3)


def arrow(x0, y0, x1, y1, color=INK, style="-|>", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 color=color, lw=lw, linestyle=ls,
                                 mutation_scale=12, zorder=1))


y_fwd, y_bwd, h = 2.55, 0.35, 0.95
box(0.15, y_fwd, 1.9, h, "Engine\n(LAMMPS / MACE)", BLUE + "22", edge=BLUE)
box(2.55, y_fwd, 1.55, h, "$(E,\\,\\mathbf{F})$\none call", "white", edge=MUT)
box(4.55, y_fwd, 2.15, h, "Free energy\n$G_\\varphi(c,T)$", AQUA + "22", edge=AQUA)
box(7.15, y_fwd, 2.6, h, "Implicit Newton\nlayer (fixed iters)", VIOLET + "22", edge=VIOLET)

arrow(2.05, y_fwd + h / 2, 2.55, y_fwd + h / 2, color=INK)
arrow(4.10, y_fwd + h / 2, 4.55, y_fwd + h / 2, color=INK)
arrow(6.70, y_fwd + h / 2, 7.15, y_fwd + h / 2, color=INK)
arrow(9.75, y_fwd + h / 2, 10.05, y_fwd + h / 2, color=INK)
ax.annotate("boundary\n$(T_e,c_e,\\ldots)$", (10.15, y_fwd + h / 2), fontsize=7.5,
            color=SEC, ha="left", va="center")

ax.annotate("forward pass", (0.15, y_fwd + h + 0.18), fontsize=7.5, color=SEC,
            weight="bold")

# backward pass row
box(7.15, y_bwd, 2.6, h, "IFT derivative\n(no engine calls)", VIOLET + "22", edge=VIOLET)
box(4.55, y_bwd, 2.15, h, "$\\partial G/\\partial\\theta$", AQUA + "22", edge=AQUA)
box(2.55, y_bwd, 1.55, h, "VJP residual\n$-\\mathbf{F}\\cdot g$", "white", edge=MUT)
box(0.15, y_bwd, 1.9, h, "$\\partial(\\text{boundary})/$\n$\\partial(\\text{engine input})$",
    BLUE + "22", edge=BLUE, fs=7.5)

arrow(7.15, y_bwd + h / 2, 6.70, y_bwd + h / 2, color=SEC, ls="--")
arrow(4.55, y_bwd + h / 2, 4.10, y_bwd + h / 2, color=SEC, ls="--")
arrow(2.55, y_bwd + h / 2, 2.05, y_bwd + h / 2, color=SEC, ls="--")
ax.annotate("backward pass — reuses the stored forces, zero new engine calls",
            (0.15, y_bwd - 0.32), fontsize=7.5, color=SEC, weight="bold")

ax.annotate("$\\mathbf{F} = -\\partial E/\\partial\\mathbf{R}$ stored as the\n"
            "vector-Jacobian-product residual",
            (3.30, (y_fwd + y_bwd + h) / 2), fontsize=6.8, color=MUT,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID, lw=0.7))

fig.savefig(FIG / "adapter_schematic.png")
plt.close(fig)

print(f"wrote 7 figures to {FIG}")
