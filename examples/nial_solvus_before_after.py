"""
Ni-Al gamma/gamma' solvus, recreated by training the potential to it.

Companion to phase_diagram_before_after.py (Cu-Ni). The joint tune drove the
Ni-Al gamma' solvus from x_Al = 0.079 to 0.129 (assessed ~0.13). Here we draw
the whole solvus LINE before and after, from MACE-MP-0's own energetics: relax
the reference structures once, evaluate the pretrained and tuned energies on
that frozen geometry (the tune's own frozen-geometry scope), form the gamma
regular-solution interaction and the gamma' line-compound formation energy for
each, and trace the line-compound solvus across temperature with the library's
Newton construction.

Run:  PYTHONPATH=. python examples/nial_solvus_before_after.py [--reps 3]
Out:  examples/output/nial_solvus_before_after.png
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
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "output"; OUT.mkdir(exist_ok=True)
CKPT = OUT / "mace_finetune_joint_all_108_reg0.1_anc5.0.pt"
K_B = 8.617333e-5
T_NIAL = 1000.0          # the tune's target temperature
XAL_ASSESSED = 0.13
X_GP = 0.125             # gamma composition used in the construction

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3)          # 3 -> 108-atom cells (matches the tune)
args = ap.parse_args()
REPS = args.reps; N = 4 * REPS**3
rng = np.random.default_rng(0)

calc = mace_mp(model="small", device="cpu", default_dtype="float64")


def relax(atoms, fmax=0.04):
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=200)
    return atoms


def epa(atoms):                                          # per-atom energy, current weights
    a = atoms.copy(); a.calc = calc
    return a.get_potential_energy() / len(a)


def overlay(ckpt):                                       # copy tuned weights into the model
    state = torch.load(ckpt, map_location="cpu")
    sd = dict(calc.models[0].named_parameters())
    for name, w in state.items():
        if name in sd:
            sd[name].data.copy_(w)


def fcc(el, a):
    return bulk(el, "fcc", a=a, cubic=True).repeat((REPS,) * 3)


def fcc_mix(elA, elB, a, frac):
    at = fcc(elA, a); nums = at.get_atomic_numbers()
    nums[rng.permutation(N)[: int(round(frac * N))]] = bulk(elB, "fcc").numbers[0]
    at.set_atomic_numbers(nums); return at


def solvus_line(om_g, Hf, Ts, c0=0.25, n=150):
    """gamma/gamma' solvus x_Al(T) for the regular-gamma / line-compound-gamma'
    model, by damped Newton in logit space (numpy; plotting only)."""
    out = []
    for T in Ts:
        u = np.log(0.05 / 0.95)
        for _ in range(n):
            s = 1.0 / (1.0 + np.exp(-u)); x = s * c0
            G = om_g * x * (1 - x) + K_B * T * (x * np.log(x) + (1 - x) * np.log(1 - x))
            dG = om_g * (1 - 2 * x) + K_B * T * np.log(x / (1 - x))
            r = G + dG * (c0 - x) - Hf
            d2 = -2 * om_g + K_B * T * (1 / x + 1 / (1 - x))
            dr = d2 * (c0 - x) * (s * (1 - s) * c0)
            u = u - r / dr
        out.append(1.0 / (1.0 + np.exp(-u)) * c0)
    return np.array(out)


print(f"Relaxing Ni-Al reference structures ({N} atoms)…")
S_ni  = relax(fcc("Ni", 3.52))
S_al  = relax(fcc("Al", 4.05))
S_g   = relax(fcc_mix("Ni", "Al", 3.55, X_GP))              # gamma: Ni-12.5%Al fcc
gp = fcc("Ni", 3.57); sc = gp.get_scaled_positions()        # gamma': Ni3Al L12
corner = np.all(np.abs(sc * REPS - np.round(sc * REPS)) < 1e-6, axis=1)
nums = gp.get_atomic_numbers(); nums[np.where(corner)[0]] = 13; gp.set_atomic_numbers(nums)
S_gp  = relax(gp)


def om_hf():
    """gamma interaction Omega_g and gamma' formation Hf at the current weights."""
    e_ni, e_al = epa(S_ni), epa(S_al)
    dH = epa(S_g) - (1 - X_GP) * e_ni - X_GP * e_al
    Hf = epa(S_gp) - 0.75 * e_ni - 0.25 * e_al
    return dH / (X_GP * (1 - X_GP)), Hf


print("Evaluating pretrained energetics…")
om_pre, hf_pre = om_hf()
print(f"   pretrained: Omega_g = {om_pre*1e3:+.0f} meV, Hf(gamma') = {hf_pre*1e3:+.0f} meV")
print(f"Overlaying tuned checkpoint {CKPT.name}…")
overlay(CKPT)
om_tun, hf_tun = om_hf()
print(f"   tuned:      Omega_g = {om_tun*1e3:+.0f} meV, Hf(gamma') = {hf_tun*1e3:+.0f} meV")

Ts = np.linspace(700.0, 1400.0, 60)
xv_pre = solvus_line(om_pre, hf_pre, Ts)
xv_tun = solvus_line(om_tun, hf_tun, Ts)
x_at_target_pre = float(solvus_line(om_pre, hf_pre, [T_NIAL])[0])
x_at_target_tun = float(solvus_line(om_tun, hf_tun, [T_NIAL])[0])
print(f"\n   solvus x_Al at {T_NIAL:.0f} K:  pretrained {x_at_target_pre:.3f}  ->  "
      f"tuned {x_at_target_tun:.3f}  (assessed {XAL_ASSESSED})")

fig, ax = plt.subplots(figsize=(6.0, 4.6))
ax.plot(xv_pre, Ts, "--", color="0.5", lw=2, label=f"MACE-MP-0 (pretrained)")
ax.plot(xv_tun, Ts, "-", color="C0", lw=2, label="tuned (trained to solvus)")
ax.axvline(XAL_ASSESSED, ls=":", color="C2", lw=1.3)
ax.text(XAL_ASSESSED + 0.004, 720, f"assessed $x_{{Al}}\\approx{XAL_ASSESSED}$",
        color="C2", fontsize=8, rotation=90, va="bottom")
ax.plot([x_at_target_pre, x_at_target_tun], [T_NIAL, T_NIAL], "o",
        color="k", ms=4, zorder=5)
ax.set_xlabel("Al content in $\\gamma$,  $x_{Al}$")
ax.set_ylabel("temperature  (K)")
ax.set_xlim(0, 0.2)
ax.set_title("Ni-Al $\\gamma/\\gamma'$ solvus: recreated by training the potential")
ax.legend(fontsize=8, loc="upper left")
ax.annotate("$\\gamma$ (fcc solid solution)", (0.15, 1150), fontsize=8, color="0.35")
fig.tight_layout()
out = OUT / "nial_solvus_before_after.png"
fig.savefig(out, dpi=150)
json.dump({"reps": REPS, "omega_g_pre": om_pre, "omega_g_tuned": om_tun,
           "Hf_pre": hf_pre, "Hf_tuned": hf_tun,
           "xAl_pre": x_at_target_pre, "xAl_tuned": x_at_target_tun},
          open(OUT / "nial_solvus_before_after.json", "w"), indent=1)
print(f"\nwrote {out}")
