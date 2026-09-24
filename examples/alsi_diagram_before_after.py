"""
Al-Si eutectic, recreated by supplying the measured liquid attraction.

Companion to mace_finetune_alsi_liquid.py. That re-tune drives MACE's Al-Si
liquid interaction from ~0 to the ~ -0.19 eV the gradient measured as missing;
dT_e/dOmega_L = +304 K/eV, so the eutectic should drop from ~907 K onto the
~850 K experiment. Here we draw the full eutectic diagram (both liquidus
branches + the eutectic tie-line) before and after, from the solid model in
alsi_rung.json with only Omega_L changed --- the loop made visible:
measure the deficit, train it in, read back the corrected diagram.

Omega_L (tuned) is read from the re-tune JSON if present; otherwise pass
--omega-l-tuned to preview the expected result.

Run:  PYTHONPATH=. python examples/alsi_diagram_before_after.py [--omega-l-tuned -0.187]
Out:  examples/output/alsi_diagram_before_after.png
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coexist.core.phase_diagram import (K_B, liquid_solution, common_tangent,
                                        three_phase_equilibrium, tangent_residual)

OUT = Path(__file__).parent / "output"
DH_FUS_AL, T_M_AL = 0.11099, 933.47
DH_FUS_SI, T_M_SI = 0.52040, 1687.0
T_E_EXP, X_E_EXP = 850.0, 0.122

p = json.load(open(OUT / "alsi_rung.json"))
OM_FCC, OM_DIA = p["omega_fcc_eV"], p["omega_dia_eV"]
DE_SI = p["E_promotion_Si_fcc_minus_dia_eV"]
DE_AL = p["E_promotion_Al_dia_minus_fcc_eV"]


def curves(om_L):
    def G_fcc(c, T):
        return (c * DE_SI + OM_FCC * c * (1 - c)
                + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))
    def G_dia(c, T):
        return ((1 - c) * DE_AL + OM_DIA * c * (1 - c)
                + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))
    G_L = liquid_solution(om_L, dH_fus_A=DH_FUS_AL, T_m_A=T_M_AL,
                          dH_fus_B=DH_FUS_SI, T_m_B=T_M_SI)
    return G_fcc, G_L, G_dia


def eutectic(om_L):
    G_fcc, G_L, G_dia = curves(om_L)
    c1, c2, c3, Te = three_phase_equilibrium(G_fcc, G_L, G_dia,
                                             c_guesses=(0.01, 0.10, 0.995), T_guess=880.0)
    return float(c1), float(c2), float(c3), float(Te)


def liquidus(G_sol, G_L, T_hi, T_lo, seed_sol, seed_L, solid_first=True, n=45):
    """Trace a solid+L liquidus from a pure melting point down to the eutectic,
    warm-starting the common tangent each step."""
    cs_sol, cs_L, Tk = [], [], []
    gs, gL = seed_sol, seed_L
    for T in np.linspace(T_hi, T_lo, n):
        if solid_first:
            a, b = common_tangent(G_sol, G_L, float(T), c_alpha_guess=gs,
                                  c_beta_guess=gL, n_iter=100)
            resid = tangent_residual(G_sol, G_L, a, b, float(T))
            csol, cL = float(a), float(b)
        else:
            a, b = common_tangent(G_L, G_sol, float(T), c_alpha_guess=gL,
                                  c_beta_guess=gs, n_iter=100)
            resid = tangent_residual(G_L, G_sol, a, b, float(T))
            cL, csol = float(a), float(b)
        if float(resid) < 1e-7 and 0 < cL < 1 and 0 < csol < 1:
            cs_sol.append(csol); cs_L.append(cL); Tk.append(float(T))
            gs, gL = csol, cL
    return np.array(cs_sol), np.array(cs_L), np.array(Tk)


def build(om_L):
    G_fcc, G_L, G_dia = curves(om_L)
    c_fcc, x_e, c_dia, Te = eutectic(om_L)
    # Al-rich liquidus: fcc(Al) + L, from T_m(Al) [liquid ~0] down to eutectic
    sfcc, lfcc, Tf = liquidus(G_fcc, G_L, T_M_AL - 1, Te + 1, seed_sol=0.002,
                              seed_L=0.02, solid_first=True)
    # Si-rich liquidus: L + dia(Si), from T_m(Si) [liquid ~1] down to eutectic
    sdia, ldia, Td = liquidus(G_dia, G_L, T_M_SI - 1, Te + 1, seed_sol=0.998,
                              seed_L=0.98, solid_first=False)
    return dict(Te=Te, x_e=x_e, c_fcc=c_fcc, c_dia=c_dia,
                sfcc=sfcc, lfcc=lfcc, Tf=Tf, sdia=sdia, ldia=ldia, Td=Td)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="re-tune JSON to read tuned Omega_L from")
    ap.add_argument("--omega-l-tuned", type=float,
                    help="tuned liquid interaction, eV (override / preview)")
    args = ap.parse_args()

    if args.omega_l_tuned is not None:
        om_tuned = args.omega_l_tuned; src = "command line"
    else:
        j = args.json or next(iter(sorted(
            glob.glob(str(OUT / "mace_finetune_alsi_liquid_*.json")))), None)
        if j:
            om_tuned = json.load(open(j))["omega_l_tuned"]; src = Path(j).name
        else:
            om_tuned = -0.187; src = "preview default (-0.187 eV)"

    print(f"Omega_L tuned source: {src}")
    pre, tun = build(0.0), build(om_tuned)
    print(f"  pretrained (Omega_L=0):      T_e = {pre['Te']:.0f} K, x_e = {pre['x_e']:.3f}")
    print(f"  tuned (Omega_L={om_tuned*1e3:+.0f} meV): T_e = {tun['Te']:.0f} K, x_e = {tun['x_e']:.3f}")
    print(f"  experiment:                  T_e = {T_E_EXP:.0f} K, x_e = {X_E_EXP:.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for d, color, ls, lab in [(pre, "0.5", "--", f"MACE-MP-0 (ideal liquid): $T_e$={pre['Te']:.0f} K"),
                              (tun, "C3", "-", f"tuned liquid ($\\Omega_L$={om_tuned*1e3:+.0f} meV): $T_e$={tun['Te']:.0f} K")]:
        ax.plot(d["lfcc"], d["Tf"], ls, color=color, lw=2, label=lab)
        ax.plot(d["ldia"], d["Td"], ls, color=color, lw=2)
        ax.plot([d["c_fcc"], d["c_dia"]], [d["Te"], d["Te"]], ls, color=color, lw=1.2)
    ax.plot([X_E_EXP], [T_E_EXP], "*", color="C2", ms=13, label="experiment", zorder=6)
    ax.set_xlim(0, 1); ax.set_xlabel("composition  $x_{Si}$")
    ax.set_ylabel("temperature  (K)")
    ax.set_title("Al-Si eutectic: recreated by training in the liquid attraction")
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    out = OUT / "alsi_diagram_before_after.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")
