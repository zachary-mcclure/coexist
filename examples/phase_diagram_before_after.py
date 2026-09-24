"""
The payoff figure: a phase diagram we RECREATED by training the potential to it.

The joint tune drove MACE-MP-0's Cu-Ni equimolar mixing energy from Omega ~ 42
to ~108 meV/atom (benchmark Omega_CuNi), targeting the assessed consolute
temperature ~625 K. Here we take those two Omega values, build the regular-
solution solid free-energy curve for each, and trace the miscibility-gap
binodal with the library's OWN differentiable common-tangent solver
(coexist.core.phase_diagram.common_tangent on the curve against itself). The
result is the Cu-Ni solid miscibility gap as MACE gave it vs. after training —
the inverse-design claim made visible: specify a boundary feature (T_c), let
reverse-mode reshape the potential, read back the corrected diagram.

Omega values are auto-read from the joint run's benchmark JSON (reg 0.1 by
default); override with --om-pre / --om-tuned (meV) or --json PATH.

Run:  python examples/phase_diagram_before_after.py
Out:  examples/output/cuni_gap_before_after.png
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coexist.core.phase_diagram import (regular_solution, liquid_solution,
                                        common_tangent, tangent_residual, K_B)

# Cu-Ni is isomorphous: one fcc solid solution, a lens-shaped melting region
# between the two melting points, and the solid miscibility gap below it.
DH_FUS_CU, T_M_CU = 0.1374, 1357.77      # eV/atom, K  (component at x=0)
DH_FUS_NI, T_M_NI = 0.1811, 1728.0       # eV/atom, K  (component at x=1)

OUT = Path(__file__).parent / "output"; OUT.mkdir(exist_ok=True)
T_C_ASSESSED = 625.0   # assessed Cu-Ni consolute temperature (K)


def omegas_from_run(json_path):
    """Pull (Omega_pretrained, Omega_tuned) in eV from a joint benchmark JSON."""
    d = json.load(open(json_path))
    return d["bench_pre"]["Omega_CuNi"] / 1e3, d["bench_post"]["Omega_CuNi"] / 1e3


def binodal(omega, n=90):
    """Trace one regular-solution miscibility gap: (c_left, c_right, T) arrays.

    For each T below T_c = Omega/2k_B, the common tangent of G against itself
    (seeded on opposite sides) gives the two coexisting solid compositions. The
    gap narrows toward (0.5, T_c), so we warm-start each solve from the previous
    temperature's root to keep Newton converged right up to the apex.
    """
    G = regular_solution(omega)
    Tc = omega / (2 * K_B)
    Ts = np.linspace(0.03 * Tc, 0.9995 * Tc, n)
    cL, cR, Tk = [], [], []
    ga, gb = 0.02, 0.98                       # seeds, warm-started below
    for T in Ts:
        ca, cb = common_tangent(G, G, jnp.asarray(T),
                                 c_alpha_guess=ga, c_beta_guess=gb, n_iter=120)
        ca, cb = float(ca), float(cb)
        if float(tangent_residual(G, G, ca, cb, jnp.asarray(T))) < 1e-7 \
                and cb - ca > 1e-3:
            cL.append(ca); cR.append(cb); Tk.append(float(T))
            ga, gb = ca, cb                   # track the narrowing gap upward
    cL.append(0.5); cR.append(0.5); Tk.append(Tc)   # close the dome at the apex
    return np.array(cL), np.array(cR), np.array(Tk), Tc


def melting_lens(omega, n=70):
    """Solidus/liquidus lens for the fcc solution vs the liquid, swept between
    the two melting points. Returns (solidus_c, liquidus_c, T)."""
    G_s = regular_solution(omega)
    G_L = liquid_solution(0.0, dH_fus_A=DH_FUS_CU, T_m_A=T_M_CU,
                          dH_fus_B=DH_FUS_NI, T_m_B=T_M_NI)   # Cu-Ni liquid ~ideal
    cs, cl, Tk = [], [], []
    gs, gl = 0.02, 0.01
    for T in np.linspace(T_M_CU + 0.5, T_M_NI - 0.5, n):
        a, b = common_tangent(G_s, G_L, jnp.asarray(T),
                              c_alpha_guess=gs, c_beta_guess=gl, n_iter=120)
        a, b = float(a), float(b)               # a = solidus (solid), b = liquidus
        if float(tangent_residual(G_s, G_L, a, b, jnp.asarray(T))) < 1e-7 \
                and 0 < b < 1 and 0 < a < 1:
            cs.append(a); cl.append(b); Tk.append(float(T)); gs, gl = a, b
    cs = [0.0] + cs + [1.0]; cl = [0.0] + cl + [1.0]        # close at the melting points
    Tk = [T_M_CU] + Tk + [T_M_NI]
    return np.array(cs), np.array(cl), np.array(Tk)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="joint benchmark JSON to read Omega from")
    ap.add_argument("--om-pre", type=float, help="pretrained Omega_CuNi, meV (override)")
    ap.add_argument("--om-tuned", type=float, help="tuned Omega_CuNi, meV (override)")
    args = ap.parse_args()

    if args.om_pre is not None and args.om_tuned is not None:
        om_pre, om_tuned = args.om_pre / 1e3, args.om_tuned / 1e3
        src = "command line"
    else:
        j = args.json or next(iter(sorted(
            glob.glob(str(OUT / "mace_finetune_joint_all_*_reg0.1_*.json")))), None)
        if j:
            om_pre, om_tuned = omegas_from_run(j); src = Path(j).name
        else:
            om_pre, om_tuned = 0.0424, 0.1076   # the reg-0.1 108-atom run
            src = "built-in defaults (reg-0.1 run)"

    print(f"Omega source: {src}")
    print(f"  pretrained  Omega = {om_pre*1e3:6.1f} meV  ->  T_c = {om_pre/(2*K_B):5.0f} K")
    print(f"  tuned       Omega = {om_tuned*1e3:6.1f} meV  ->  T_c = {om_tuned/(2*K_B):5.0f} K")
    print(f"  assessed target                          T_c = {T_C_ASSESSED:5.0f} K")

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    # melting lens (top): fusion-dominated, essentially unchanged by the tune,
    # so draw it once as the shared, untuned part of the diagram.
    cs, cl, Tl = melting_lens(om_pre)
    ax.plot(cs, Tl, "-", color="0.4", lw=1.8)
    ax.plot(cl, Tl, "-", color="0.4", lw=1.8,
            label="solidus / liquidus (untuned melting)")
    for omega, color, ls, lab in [(om_pre, "0.5", "--", "MACE-MP-0 (pretrained)"),
                                  (om_tuned, "C3", "-", "tuned (trained to T_c)")]:
        # miscibility gap (bottom) — the feature the tune reshapes
        cL, cR, Tk, Tc = binodal(omega)
        dome_c = np.concatenate([cL, cR[::-1]])
        dome_T = np.concatenate([Tk, Tk[::-1]])
        ax.plot(dome_c, dome_T, ls, color=color, lw=2,
                label=f"{lab}:  Ω={omega*1e3:.0f} meV, T_c={Tc:.0f} K")
        ax.plot([0.5], [Tc], "o", color=color, ms=5)

    ax.axhline(T_C_ASSESSED, ls=":", color="C2", lw=1.1)
    ax.text(0.015, T_C_ASSESSED + 12, f"assessed T_c ≈ {T_C_ASSESSED:.0f} K",
            color="C2", fontsize=8)
    ax.plot([0], [T_M_CU], "k_", ms=8); ax.plot([1], [T_M_NI], "k_", ms=8)
    ax.text(0.01, T_M_CU + 12, "$T_m$(Cu)", fontsize=7, color="0.3")
    ax.text(0.99, T_M_NI + 12, "$T_m$(Ni)", fontsize=7, color="0.3", ha="right")
    ax.set_xlim(0, 1); ax.set_ylim(0, T_M_NI + 120)
    ax.set_xlabel("composition  x(Ni)"); ax.set_ylabel("temperature  (K)")
    ax.set_title("Cu–Ni phase diagram: miscibility gap recreated by training")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.annotate("liquid", (0.62, 1620), ha="center", fontsize=8, color="0.35")
    ax.annotate("fcc solid solution", (0.5, 1050),
                ha="center", fontsize=8, color="0.35")
    ax.annotate("miscibility gap", (0.5, T_C_ASSESSED * 0.42),
                ha="center", fontsize=8, color="0.35")
    fig.tight_layout()
    out = OUT / "cuni_gap_before_after.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")
