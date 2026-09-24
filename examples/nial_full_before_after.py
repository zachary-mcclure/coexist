"""
Full Ni-rich Ni-Al diagram, before/after training the potential.

Companion to nial_solvus_before_after.py, but the WHOLE Ni-rich diagram rather
than just the solvus: the gamma solidus/liquidus, the gamma/L/gamma' eutectic,
the gamma' line-compound liquidus, and the gamma/gamma' solvus. The gamma
solution is fit as Redlich-Kister from three MACE compositions (a single
regular-solution point would misrepresent the liquidus), gamma' and beta are
MACE line-compound formation energies, all evaluated pretrained then tuned on
the frozen relaxed geometry. The liquid is the literature-sampled model
(nial_liquid_retemp.json), held fixed --- the untuned scaffold, as the melting
lens is for Cu-Ni.

Run:  PYTHONPATH=. python examples/nial_full_before_after.py [--reps 3]
Out:  examples/output/nial_full_before_after.png
"""
import argparse, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coexist.core.phase_diagram import (K_B, redlich_kister_solution, line_compound,
                                        common_tangent, three_phase_equilibrium,
                                        tangent_residual)

OUT = Path(__file__).parent / "output"
CKPT = OUT / "mace_finetune_joint_all_108_reg0.1_anc5.0.pt"
DH_FUS_NI, T_M_NI = 0.18117, 1728.0
DH_FUS_AL, T_M_AL = 0.11099, 933.47

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()
REPS, N = args.reps, 4 * args.reps**3
rng = np.random.default_rng(0)
calc = mace_mp(model="small", device="cpu", default_dtype="float64")

# held liquid (literature-sampled), exactly as nial_diagram.py
rt = json.load(open(OUT / "nial_liquid_retemp.json"))
L_lo = jnp.array(rt["L_liquid_1500K_eV"]); sl = jnp.array(rt["dL_dT_eV_per_K"])


def G_L(c, T):
    dG_Ni = DH_FUS_NI * (1 - T / T_M_NI); dG_Al = DH_FUS_AL * (1 - T / T_M_AL)
    L_T = L_lo + (T - 1500.0) * sl
    series = L_T[0] + L_T[1] * (1 - 2 * c)
    return ((1 - c) * dG_Ni + c * dG_Al + c * (1 - c) * series
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))


def relax(atoms, fmax=0.04):
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=200)
    return atoms


def epa(atoms):
    a = atoms.copy(); a.calc = calc
    return a.get_potential_energy() / len(a)


def overlay(ckpt):
    state = torch.load(ckpt, map_location="cpu")
    sd = dict(calc.models[0].named_parameters())
    for nm, w in state.items():
        if nm in sd: sd[nm].data.copy_(w)


def fcc(el, a):
    return bulk(el, "fcc", a=a, cubic=True).repeat((REPS,) * 3)


def fcc_mix(a, frac):
    at = fcc("Ni", a); nums = at.get_atomic_numbers()
    nums[rng.permutation(N)[: int(round(frac * N))]] = 13; at.set_atomic_numbers(nums)
    return at


print(f"Relaxing Ni-Al structures ({N} atoms)…")
S_ni = relax(fcc("Ni", 3.52)); S_al = relax(fcc("Al", 4.05))
XG = [0.05, 0.125, 0.20]
S_g = [relax(fcc_mix(3.55, x)) for x in XG]                     # gamma at 3 comps -> RK fit
gp = fcc("Ni", 3.57); sc = gp.get_scaled_positions()           # gamma' L12
corner = np.all(np.abs(sc * REPS - np.round(sc * REPS)) < 1e-6, axis=1)
nums = gp.get_atomic_numbers(); nums[np.where(corner)[0]] = 13; gp.set_atomic_numbers(nums)
S_gp = relax(gp)
beta = bulk("Ni", "bcc", a=2.88, cubic=True).repeat((REPS,) * 3)  # NiAl B2
nb = beta.get_atomic_numbers(); nb[1::2] = 13; beta.set_atomic_numbers(nb)
S_b = relax(beta)


def params():
    """RK gamma coefficients + gamma'/beta formation energies at current weights."""
    e_ni, e_al = epa(S_ni), epa(S_al)
    # RK fit: dHmix/(x(1-x)) = L0 + L1(1-2x)
    y, A = [], []
    for x, s in zip(XG, S_g):
        dH = epa(s) - (1 - x) * e_ni - x * e_al
        y.append(dH / (x * (1 - x))); A.append([1.0, 1 - 2 * x])
    L0, L1 = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)[0]
    HF_GP = epa(S_gp) - 0.75 * e_ni - 0.25 * e_al
    HF_B = epa(S_b) - 0.5 * e_ni - 0.5 * e_al
    return float(L0), float(L1), float(HF_GP), float(HF_B)


def build(L0, L1, HF_GP, HF_B):
    G_g = redlich_kister_solution(jnp.array([L0, L1]))
    G_gp = line_compound(0.25, HF_GP); G_beta = line_compound(0.50, HF_B)

    def sweep(Ga, Gb, Ts, ga, gb):
        A, B, TT = [], [], []
        for T in Ts:
            try:
                ca, cb = common_tangent(Ga, Gb, float(T), c_alpha_guess=ga, c_beta_guess=gb)
            except Exception:
                continue
            r = float(tangent_residual(Ga, Gb, ca, cb, float(T)))
            if not np.isfinite(r) or r > 1e-7:
                continue
            ga, gb = float(ca), float(cb); A.append(ga); B.append(gb); TT.append(float(T))
        return np.array(A), np.array(B), np.array(TT)

    c1, c2, c3, Te = three_phase_equilibrium(G_g, G_L, G_gp,
                                             c_guesses=(0.14, 0.17, 0.2495), T_guess=1440.0)
    Te = float(Te)
    gs, gl, gT = sweep(G_g, G_L, np.linspace(T_M_NI - 2, Te + 0.5, 80), 0.005, 0.008)
    ll, gpl, lT = sweep(G_L, G_gp, np.linspace(Te + 0.5, 1850.0, 60), float(c2), 0.2495)
    svg, svgp, svT = sweep(G_g, G_gp, np.linspace(Te - 0.5, 700.0, 60), float(c1), 0.2495)
    return dict(Te=Te, c_gamma=float(c1), c_L=float(c2),
                gs=gs, gl=gl, gT=gT, ll=ll, lT=lT, svg=svg, svT=svT)


print("Evaluating pretrained…")
pp = params(); pre = build(*pp)
print(f"   pretrained: L0={pp[0]*1e3:.0f} L1={pp[1]*1e3:.0f} Hf(gp)={pp[2]*1e3:.0f} "
      f"Hf(b)={pp[3]*1e3:.0f} meV | eutectic {pre['Te']:.0f} K")
print("Overlaying tuned checkpoint…")
overlay(CKPT)
tp = params(); tun = build(*tp)
print(f"   tuned:      L0={tp[0]*1e3:.0f} L1={tp[1]*1e3:.0f} Hf(gp)={tp[2]*1e3:.0f} "
      f"Hf(b)={tp[3]*1e3:.0f} meV | eutectic {tun['Te']:.0f} K")

fig, ax = plt.subplots(figsize=(6.6, 5.0))
for d, color, ls, lab in [(pre, "0.5", "--", "MACE-MP-0 (pretrained)"),
                          (tun, "C3", "-", "tuned")]:
    ax.plot(d["gs"], d["gT"], ls, color=color, lw=2, label=lab)       # gamma solidus
    ax.plot(d["gl"], d["gT"], ls, color=color, lw=2)                   # gamma liquidus
    ax.plot(d["ll"], d["lT"], ls, color=color, lw=2)                   # L + gamma'
    ax.plot(d["svg"], d["svT"], ls, color=color, lw=2)                # gamma/gamma' solvus
    ax.plot([d["c_gamma"], 0.25], [d["Te"], d["Te"]], ls, color=color, lw=1.1)  # eutectic tie
ax.axvline(0.25, color="0.6", lw=3, alpha=0.35)                        # gamma' line compound
ax.text(0.252, 780, "$\\gamma'$ (Ni$_3$Al)", fontsize=8, color="0.4", rotation=90, va="bottom")
ax.axvline(0.13, ls=":", color="C2", lw=1.1)
ax.text(0.135, 720, "assessed solvus $x_{Al}$≈0.13", color="C2", fontsize=7, rotation=90, va="bottom")
ax.set_xlim(0, 0.32); ax.set_ylim(700, T_M_NI + 60)
ax.set_xlabel("Al content  $x_{Al}$"); ax.set_ylabel("temperature  (K)")
ax.set_title("Ni-Al (Ni-rich): $\\gamma/\\gamma'$ region recreated by training")
ax.legend(fontsize=8, loc="lower left")
ax.annotate("liquid", (0.06, 1680), fontsize=8, color="0.35")
ax.annotate("$\\gamma$ (fcc)", (0.03, 1200), fontsize=8, color="0.35")
fig.tight_layout()
out = OUT / "nial_full_before_after.png"
fig.savefig(out, dpi=150)
json.dump({"pretrained": dict(L0=pp[0], L1=pp[1], Hf_gp=pp[2], Hf_b=pp[3], Te=pre["Te"]),
           "tuned": dict(L0=tp[0], L1=tp[1], Hf_gp=tp[2], Hf_b=tp[3], Te=tun["Te"])},
          open(OUT / "nial_full_before_after.json", "w"), indent=1)
print(f"\nwrote {out}")
