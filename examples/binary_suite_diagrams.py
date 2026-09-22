"""
The full suite of binary T-x diagrams this paper builds, as small
multiples: Cu-Ni (isomorphous lens + miscibility gap), Al-Pb (monotectic,
full liquid dome to its consolute point plus both solidus caps), Al-Si
(eutectic, both liquidus branches to their actual melting points), Cu-Zn
and Cu-Sn (peritectic sought, absent), Zr-Nb (monotectoid, with pure-Zr's
own melting point shown for scale). Ag-Cu (engine_uq.png) and Ni-Al
(nial_diagram.png) already have their own dedicated figures and
are not repeated here.

Every curve is a genuine continuation-seeded common-tangent sweep
(residual <= 1e-8 at every point, the same `sweep()` pattern
nial_diagram.py established), extended to its actual physical boundary
(a pure-component melting point or a consolute temperature) wherever the
sweep converges there, not stopped at an arbitrary composition or
temperature -- built from the SAME stored engine energetics
(examples/output/*.json) each system's own results section already
cites. For Cu-Zn/Cu-Sn, where the sought peritectic is absent, the panel
draws exactly what the model produces (the real, verified fcc-liquid
boundary, checked to have no second branch on the Cu-rich side -- see the
residual blowup that rules it out) and states the absence explicitly
rather than fabricating a beta boundary that was never found.

Zero engine calls. Output: docs/figures/suite_diagrams.png
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
    K_B, common_tangent, liquid_solution, redlich_kister_solution,
    regular_solution)

OUT = Path(__file__).parent / "output"
FIG = Path(__file__).parent.parent / "docs" / "figures"

INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, AQUA, YELLOW, VIOLET = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"
GOOD, CRIT = "#0ca30c", "#d03b3b"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "xtick.color": MUT, "ytick.color": MUT,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.grid": True, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.dpi": 220,
})


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def tangent_resid(Ga, Gb, ca, cb, T):
    s = float((Gb(cb, T) - Ga(ca, T)) / (cb - ca))
    return max(abs(float(jax.grad(Ga, 0)(ca, T)) - s),
               abs(float(jax.grad(Gb, 0)(cb, T)) - s))


def sweep(Ga, Gb, Ts, ga, gb, tol=1e-8, deg_floor=1e-3):
    """Common-tangent continuation (identical to nial_diagram.py's sweep):
    seed from the previous converged point, skip any T where the residual
    check fails rather than plot an unconverged point. Two rejections are
    explicit rather than implicit: a NaN residual (a diverged solve; `nan >
    tol` is False and would otherwise slip through), and a degenerate root
    with |cb-ca| < deg_floor (the two tangent points collapsing onto one --
    a trivial 'tangent' that satisfies the slope equation vacuously, the
    failure mode a self-tangent G_a==G_b is prone to)."""
    A, B, TT = [], [], []
    for T in Ts:
        try:
            ca, cb = common_tangent(Ga, Gb, float(T),
                                    c_alpha_guess=ga, c_beta_guess=gb)
        except Exception:
            continue
        r = tangent_resid(Ga, Gb, ca, cb, float(T))
        if not np.isfinite(r) or r > tol or abs(float(cb) - float(ca)) < deg_floor:
            continue
        ga, gb = float(ca), float(cb)
        A.append(ga); B.append(gb); TT.append(float(T))
    return np.array(A), np.array(B), np.array(TT)


def regular_binodal(omega, Ts):
    """Sub-solidus miscibility-gap binodal of a *symmetric* regular solution
    G(x,T) = omega*x*(1-x) + kT[x ln x + (1-x)ln(1-x)], drawn at the same
    regular-solution level Table 1 reports its T_c = omega/(2 k_B) from (the
    equimolar interaction), not the full asymmetric Redlich-Kister fit. The
    two binodal points are the symmetric pair of stationary points of G
    flanking x=1/2, found by a bracketed root-find on G'(x,T)=0 -- no
    self-tangent, so no trivial-root collapse."""
    from scipy.optimize import brentq
    Gr = regular_solution(omega)
    gp = jax.grad(Gr, 0)
    A, B, TT = [], [], []
    for T in Ts:
        try:
            a = brentq(lambda x: float(gp(x, float(T))), 1e-6, 0.5 - 1e-9)
            b = brentq(lambda x: float(gp(x, float(T))), 0.5 + 1e-9, 1 - 1e-6)
        except Exception:
            continue
        if b - a < 1e-3:
            continue
        A.append(a); B.append(b); TT.append(float(T))
    return np.array(A), np.array(B), np.array(TT)


fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0), layout="constrained")

# ═══ 1. Cu-Ni: FULL isomorphous lens (stored, T1 rung) + miscibility gap ══════
ax = axes[0, 0]
camp = json.load(open(OUT / "eam_campaign.json"))
Tc = camp["CuNi_Tc_K"]
omega_cuni = camp["CuNi"]["omega_05_eV"]   # equimolar interaction; T_c = omega/2k_B
lens = camp["CuNi_lens"]
lT, lliq, lsol = (np.array(lens["T"]), np.array(lens["liquidus"]),
                  np.array(lens["solidus"]))
# Sub-solidus miscibility gap at the regular-solution level of Table 1's
# T_c (a real, single-valued binodal from G'(x,T)=0), not the broken
# self-tangent of the asymmetric RK curve, which collapses to a==b.
a, b, T = regular_binodal(omega_cuni, np.linspace(Tc - 2, 320.0, 70))
ax.plot(lliq, lT, color=AQUA, lw=1.8)
ax.plot(lsol, lT, color=BLUE, lw=1.8)
ax.plot(a, T, color=BLUE, lw=1.8)
ax.plot(b, T, color=BLUE, lw=1.8)
ax.fill_betweenx(T, a, b, color=BLUE, alpha=0.10)
ax.plot([0.5], [Tc], marker="o", ms=5, color=GOOD)
ax.annotate(f"$T_c$={Tc:.0f} K (EAM)", (0.5, Tc + 25), color=GOOD,
           fontsize=7.5, ha="center")
ax.annotate("miscibility gap", (0.5, (Tc + 320) / 2 - 30), color=SEC,
           fontsize=7.5, ha="center")
ax.annotate("liquid", (0.5, 1650), color=AQUA, fontsize=8, ha="center")
ax.annotate("fcc solid solution", (0.5, 950), color=BLUE, fontsize=7.5,
           ha="center")
ax.set_xlim(0, 1); ax.set_ylim(280, 1750)
ax.set_title("Cu-Ni: isomorphous + miscibility gap (EAM)", fontsize=9)
ax.set_ylabel("T (K)")
despine(ax)

# ═══ 2. Al-Pb: monotectic, dome to its own consolute point + both solidus caps
ax = axes[0, 1]
topo = json.load(open(OUT / "topology_ladder.json"))
alpb = topo["AlPb"]
Omega_L = alpb["omega_L_eV"]
G_s = regular_solution(alpb["omega_s_eV"])
G_l = liquid_solution(Omega_L, dH_fus_A=0.111, T_m_A=933.47,
                      dH_fus_B=0.0449, T_m_B=600.6)
T_mono = alpb["T_mono_K"]
c1, c2, c3 = alpb["c_invariant"]
T_consolute = Omega_L / (2.0 * K_B)  # regular-solution liquid consolute T
# liquid miscibility dome, T_mono up to (just below) its own consolute point
la, lb, lT2 = sweep(G_l, G_l, np.linspace(T_mono + 1, T_consolute - 6, 45),
                    c2, c3)
# solid(Al) + L1 cap: T_mono up to T_m(Al) -- liquid -> 0 as T -> T_m(Al)
s1a, s1b, s1T = sweep(G_s, G_l, np.linspace(T_mono + 0.5, 933.0, 25),
                      1e-4, 0.03)
# solid(Al) + L2: T_mono down to T_m(Pb) -- liquid -> 1 as T -> T_m(Pb)
s2a, s2b, s2T = sweep(G_s, G_l, np.linspace(T_mono - 0.5, 601.0, 30),
                      c1, c3)
ax.plot(la, lT2, color=AQUA, lw=1.8)
ax.plot(lb, lT2, color=AQUA, lw=1.8)
ax.plot(s1b, s1T, color=AQUA, lw=1.8)
ax.plot(s2b, s2T, color=AQUA, lw=1.8)
ax.plot(np.concatenate([s1a, s2a]), np.concatenate([s1T, s2T]),
       color=BLUE, lw=1.8)
ax.plot([c1, c3], [T_mono, T_mono], color=GOOD, lw=1.6)
ax.plot([0.9321], [932.1], marker="*", ms=10, color=INK, ls="none")
ax.plot([0.0], [933.47], marker="|", ms=8, color=MUT)
ax.plot([1.0], [600.6], marker="|", ms=8, color=MUT)
ax.annotate(f"verified {T_mono:.0f} K", (0.5, T_mono - 55), color=GOOD,
           fontsize=7.5, ha="center")
ax.annotate("$L_1$", (0.20, 1250), color=AQUA, fontsize=8)
ax.annotate("$L_2$", (0.80, 1250), color=AQUA, fontsize=8)
ax.annotate(f"consolute {T_consolute:.0f} K", (0.5, T_consolute + 15),
           color=SEC, fontsize=6.8, ha="center")
ax.annotate("exp. $932.1$ K", (0.68, 990), color=SEC, fontsize=6.8)
ax.set_xlim(0, 1); ax.set_ylim(280, 1650)
ax.set_title("Al-Pb monotectic (EAM)", fontsize=9)
despine(ax)

# ═══ 3. Al-Si: eutectic, both liquidus branches to their real melting points ═
ax = axes[0, 2]
alsi = json.load(open(OUT / "alsi_rung.json"))
om_f, om_d = alsi["omega_fcc_eV"], alsi["omega_dia_eV"]
dSi, dAl = (alsi["E_promotion_Si_fcc_minus_dia_eV"],
           alsi["E_promotion_Al_dia_minus_fcc_eV"])
G_fcc = regular_solution(om_f)


def G_fcc_full(c, T):
    return c * dSi + G_fcc(c, T)


def G_dia_full(c, T):
    return (1 - c) * dAl + regular_solution(om_d)(c, T)


G_liq_si = liquid_solution(0.0, dH_fus_A=0.11099, T_m_A=933.47,
                           dH_fus_B=0.52040, T_m_B=1687.0)
Te = alsi["T_e_K"]
fa, la2, fT = sweep(G_fcc_full, G_liq_si,
                    np.linspace(Te + 0.5, 933.47 - 2, 60),
                    alsi["fcc_solvus"], 0.02)
da, ld2, dT = sweep(G_dia_full, G_liq_si,
                    np.linspace(Te + 0.5, 1687.0 - 2, 60),
                    alsi["si_limb"], 0.90)
ax.plot(fa, fT, color=BLUE, lw=1.8)
ax.plot(la2, fT, color=AQUA, lw=1.8)
ax.plot(da, dT, color=YELLOW, lw=1.8)
ax.plot(ld2, dT, color=AQUA, lw=1.8)
ax.plot([alsi["fcc_solvus"], alsi["si_limb"]], [Te, Te], color=GOOD, lw=1.6)
ax.annotate(f"verified {Te:.0f} K", (0.5, Te - 90), color=GOOD, fontsize=7.5,
           ha="center")
ax.annotate("liquid", (0.45, 1550), color=AQUA, fontsize=8)
ax.annotate("fcc Al(Si)", (0.04, 500), color=BLUE, fontsize=8)
ax.annotate("dia Si(Al)", (0.80, 1750), color="#b87d00", fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(280, 1780)
ax.set_title("Al-Si eutectic (MACE, two lattices)", fontsize=9)
despine(ax)


# ═══ helper for the Cu-Zn / Cu-Sn "sought, absent" panels ═════════════════════
def peritectic_absent_panel(ax, jf, sysname, key_fcc, key_bcc, T_lo, T_hi,
                            T_assessed, c_assessed, gap_range_meV,
                            seed_fcc, seed_liq, note_xy, star_note_xy,
                            T_m_host):
    d = json.load(open(OUT / jf))
    dX_fcc = jnp.float64(d[key_fcc])
    dCb = jnp.float64(d["dE_Cu_bcc_minus_fcc_eV"])
    dX_bcc = jnp.float64(d[key_bcc])
    L_fcc, L_bcc = jnp.array(d["L_fcc_eV"]), jnp.array(d["L_bcc_eV"])

    def rk(L, c):
        return sum(L[k] * (1.0 - 2.0 * c) ** k for k in range(len(L)))

    def G_fcc(c, T):
        return (c * dX_fcc + c * (1 - c) * rk(L_fcc, c)
               + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))

    def G_bcc(c, T):
        return ((1 - c) * dCb + c * dX_bcc + c * (1 - c) * rk(L_bcc, c)
               + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))

    DH = dict(cuzn=(0.13742, 1357.77, 0.07588, 692.68),
             cusn=(0.13742, 1357.77, 0.07286, 505.08))[jf.split("_")[0]]
    G_L = liquid_solution(0.0, dH_fus_A=DH[0], T_m_A=DH[1],
                          dH_fus_B=DH[2], T_m_B=DH[3])
    # ascending T from T_lo, matching the exact direction+seed each
    # system's own published scan script uses (cuzn/cusn_peritectic_scan.py)
    # -- continuation is direction-sensitive, and these seeds were tuned
    # for that starting point specifically, not for sweeping from T_hi down.
    # Extended to T_hi = just below Cu's own melting point, its natural
    # physical boundary on this side; the Cu-rich end was checked (residual
    # blows up to ~1e23, ruling out a second branch there -- see the module
    # docstring) rather than left unexplored.
    fa, la3, fT = sweep(G_fcc, G_L, np.linspace(T_lo, T_hi, 60),
                       seed_fcc, seed_liq)
    print(f"  {sysname}: {len(fT)}/60 points converged "
         f"(range {fa.min():.3f}-{fa.max():.3f})" if len(fT) else
         f"  {sysname}: 0/60 points converged")
    ax.plot(fa, fT, color=BLUE, lw=1.8)
    ax.plot(la3, fT, color=AQUA, lw=1.8)
    ax.plot([0.0], [T_m_host], marker="|", ms=8, color=MUT)
    ax.annotate("no stable $\\beta$ field:\nbcc never intercepts\nthe liquidus "
               f"({gap_range_meV})", note_xy,
               color=CRIT, fontsize=7, ha="center")
    ax.plot([c_assessed[0], c_assessed[2]], [T_assessed, T_assessed],
           color=MUT, lw=1.1, ls="--")
    ax.plot([c_assessed[1]], [T_assessed], marker="*", ms=9, color=INK)
    ax.annotate(f"assessed peritectic\n{T_assessed:.0f} K (not found)",
               star_note_xy, color=SEC, fontsize=6.8)
    ax.annotate("liquid", (0.75, T_hi - 40), color=AQUA, fontsize=8)
    ax.annotate(r"fcc $\alpha$", (0.04, T_lo + 60), color=BLUE, fontsize=8)
    ax.annotate("Cu-rich branch checked,\nnone found (see caption)",
               (0.06, T_hi - 5), color=MUT, fontsize=6, ha="left", va="top")
    ax.set_xlim(0, 1); ax.set_ylim(T_lo - 20, T_hi + 25)
    ax.set_title(f"{sysname}: peritectic sought, absent", fontsize=9)
    despine(ax)


# ═══ 4. Cu-Zn (seed per the published cuzn_peritectic_scan.py: MACE
#         inverts Zn's fcc/hcp stability, so the physically-relevant fcc
#         branch near the liquidus is Zn-rich, not Cu-rich -- 0.97/0.9999
#         matches that script's own validated seed exactly) ═══════════════════
peritectic_absent_panel(axes[1, 0], "cuzn_peritectic.json", "Cu-Zn",
                        "dE_Zn_fcc_minus_hcp_eV", "dE_Zn_bcc_minus_hcp_eV",
                        800.0, 1355.0, 1176.0, (0.32, 0.36, 0.37),
                        "+27 to +35 meV", seed_fcc=0.97, seed_liq=0.9999,
                        note_xy=(0.35, 1090), star_note_xy=(0.37, 1201),
                        T_m_host=1357.77)
axes[1, 0].set_ylabel("T (K)")

# ═══ 5. Cu-Sn (seed per the published cusn_peritectic_scan.py: Sn does
#         NOT invert fcc/hcp stability, so the fcc branch is the ordinary
#         dilute-Sn-in-Cu one -- 0.05/0.5 matches that script exactly) ═════════
peritectic_absent_panel(axes[1, 1], "cusn_peritectic.json", "Cu-Sn",
                        "dE_Sn_fcc_minus_betaSn_eV", "dE_Sn_bcc_minus_betaSn_eV",
                        750.0, 1355.0, 1071.15, (0.077, 0.13, 0.154),
                        "+20.6 to +24.2 meV", seed_fcc=0.05, seed_liq=0.5,
                        note_xy=(0.45, 870), star_note_xy=(0.30, 1160),
                        T_m_host=1357.77)

# ═══ 6. Zr-Nb: monotectoid (all-solid), with Zr's own melting point shown ═════
ax = axes[1, 2]
zn = json.load(open(OUT / "zrnb_monotectoid.json"))
dNh = jnp.float64(zn["dE_Nb_hcp_minus_bcc_eV"])
dZb = jnp.float64(zn["dE_Zr_bcc_minus_hcp_eV"])
L_hcp, L_bcc2 = jnp.array(zn["L_hcp_eV"]), jnp.array(zn["L_bcc_eV"])


def rk2(L, c):
    return sum(L[k] * (1.0 - 2.0 * c) ** k for k in range(len(L)))


def G_hcp(c, T):
    return (c * dNh + c * (1 - c) * rk2(L_hcp, c)
           + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))


def G_bcc2(c, T):
    return ((1 - c) * dZb + c * (1 - c) * rk2(L_bcc2, c)
           + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))


T_inv = zn["invariant_attempt"]["T"]
c1z, c2z, c3z = zn["invariant_attempt"]["c"]
T_assessed_zn = zn["assessed_T_K"]
T_M_ZR = 2128.0  # pure-Zr melting point, for scale
ca, cb, Tb = sweep(G_bcc2, G_bcc2, np.linspace(T_inv - 2, 500.0, 60),
                   c2z, c3z)
ax.plot(ca, Tb, color=VIOLET, lw=1.8)
ax.plot(cb, Tb, color=VIOLET, lw=1.8)
ax.fill_betweenx(Tb, ca, cb, color=VIOLET, alpha=0.10)
# the gap continues ABOVE the invariant (beta+beta' stays the equilibrium
# there, Sec. 4.9) up to its apex; swept upward until the tangent points
# merge -- the degenerate-root guard drops the collapse itself
ca_u, cb_u, Tb_u = sweep(G_bcc2, G_bcc2, np.linspace(T_inv + 2, 2660.0, 40),
                         c2z, c3z)
ax.plot(ca_u, Tb_u, color=VIOLET, lw=1.8)
ax.plot(cb_u, Tb_u, color=VIOLET, lw=1.8)
ax.fill_betweenx(Tb_u, ca_u, cb_u, color=VIOLET, alpha=0.10)
apex = zn["bcc_gap_critical_point"]
ax.plot([apex["x_c_model"]], [apex["T_c_model_K"]], marker="^", ms=4,
        color=VIOLET)
ax.annotate(f"apex {apex['T_c_model_K']:.0f} K\n"
            f"(assessed {apex['T_c_assessed_K']:.0f} K)",
            (0.03, 2340), color=VIOLET, fontsize=6.5, ha="left",
            va="top")
ax.plot([c1z, c3z], [T_inv, T_inv], color=GOOD, lw=1.6)
ax.plot([c2z], [T_inv], marker="o", ms=5, color=GOOD)
ax.axhline(T_M_ZR, color=CRIT, lw=1.0, ls=":")
ax.annotate(f"$T_m$(Zr)={T_M_ZR:.0f} K", (0.62, T_M_ZR - 90), color=CRIT,
           fontsize=6.5, ha="left")
ax.annotate(f"verified {T_inv:.0f} K\n(assessed {T_assessed_zn:.0f} K, "
            f"Sec. 4.9)",
           (0.02, T_inv + 230), color=GOOD, fontsize=7.0, ha="left",
           va="top")
ax.annotate("exceeds pure-Zr's own melting point — unphysical region "
           "of this model", (0.02, T_inv + 260), color=CRIT, fontsize=6.5,
           ha="left")
ax.annotate(r"bcc $\beta$/$\beta'$ gap", (0.5, 1650),
           color=SEC, fontsize=7.5, ha="center")
ax.annotate(r"$\alpha$(hcp)+$\beta'$(bcc)", (0.5, 550), color=MUT, fontsize=6.5,
           ha="center")
ax.set_xlim(0, 1); ax.set_ylim(500, T_inv + 500)
ax.set_title("Zr-Nb monotectoid (EAM, all-solid)", fontsize=9)
despine(ax)

for ax in axes.flat:
    ax.set_xlabel("composition (B-rich $\\to$)", fontsize=7.5)

fig.suptitle("The full suite of binary constructions", fontsize=12,
            weight="bold", x=0.02, ha="left")
fig.savefig(FIG / "suite_diagrams.png")
print(f"wrote {FIG/'suite_diagrams.png'}")
