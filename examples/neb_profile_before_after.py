"""
Kinetics figure: the vacancy-migration energy profile, before and after tuning.

Runs a climbing-image NEB for Cu vacancy migration on the pretrained potential,
then evaluates the SAME minimum-energy path with the barrier-tuned checkpoint,
so both profiles share a reaction coordinate. Shows MACE-MP-0's too-low barrier
lifted onto the experimental migration energy -- the kinetic counterpart of the
phase-diagram recreations.

Run:  PYTHONPATH=. python examples/neb_profile_before_after.py [--reps 2]
Out:  examples/output/neb_profile_before_after.png
"""
import argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase.build import bulk
from ase.optimize import FIRE
from ase.geometry import find_mic
try:
    from ase.mep import NEB
except ImportError:
    from ase.neb import NEB
from mace.calculators import mace_mp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "output"
CKPT = OUT / "mace_finetune_neb_Cu_all_reg0.1.pt"   # credible anchored barrier tune
A0, EM_EXP = 3.615, 0.71

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--images", type=int, default=7)
args = ap.parse_args()
REPS = args.reps
calc = mace_mp(model="small", device="cpu", default_dtype="float64")


def relax_pos(atoms, fmax=0.03):
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=200)
    return atoms


def overlay(ckpt):
    state = torch.load(ckpt, map_location="cpu")
    sd = dict(calc.models[0].named_parameters())
    for nm, w in state.items():
        if nm in sd: sd[nm].data.copy_(w)


print(f"Building Cu vacancy-migration NEB ({REPS}x{REPS}x{REPS}, {args.images} images)…")
base = bulk("Cu", "fcc", a=A0, cubic=True).repeat((REPS,) * 3)
posA = base.positions[0].copy()
d = base.get_distances(0, range(len(base)), mic=True); d[0] = 1e9
B = int(np.argmin(d)); posB = base.positions[B].copy()
initial = base.copy(); del initial[0]
final = initial.copy(); Bi = B - 1
dvec, _ = find_mic((posA - posB).reshape(1, 3), base.cell, base.pbc)
final.positions[Bi] = posB + dvec[0]
initial = relax_pos(initial); final = relax_pos(final)
images = [initial] + [initial.copy() for _ in range(args.images - 2)] + [final]
for im in images:
    im.calc = mace_mp(model="small", device="cpu", default_dtype="float64")
neb = NEB(images, climb=True, k=0.1); neb.interpolate("idpp")
FIRE(neb, logfile=None).run(fmax=0.05, steps=150)

# reaction coordinate = cumulative path length; energies relative to the endpoint
xyz = [im.get_positions() for im in images]
rc = np.concatenate([[0.0], np.cumsum([np.linalg.norm(xyz[i + 1] - xyz[i])
                                       for i in range(len(images) - 1)])])
rc = rc / rc[-1]


def profile():
    e = np.array([im.get_potential_energy() for im in images])
    return e - e[0]


e_pre = profile(); Em_pre = float(e_pre.max())
print(f"   pretrained barrier E_m = {Em_pre:.3f} eV")
print(f"Overlaying barrier-tuned checkpoint {CKPT.name}…")
overlay(CKPT)
for im in images:                              # re-point images at the tuned model
    im.calc = calc
e_tun = profile(); Em_tun = float(e_tun.max())
print(f"   tuned barrier      E_m = {Em_tun:.3f} eV  (target {EM_EXP})")

fig, ax = plt.subplots(figsize=(6.0, 4.4))
ax.plot(rc, e_pre, "o--", color="0.5", lw=2, label=f"MACE-MP-0: $E_m$={Em_pre:.2f} eV")
ax.plot(rc, e_tun, "o-", color="C3", lw=2, label=f"tuned: $E_m$={Em_tun:.2f} eV")
ax.axhline(EM_EXP, ls=":", color="C2", lw=1.3)
ax.text(0.02, EM_EXP + 0.01, f"experiment $E_m$={EM_EXP} eV", color="C2", fontsize=8)
ax.set_xlabel("reaction coordinate"); ax.set_ylabel("energy above initial (eV)")
ax.set_title("Cu vacancy migration barrier: tuned onto experiment")
ax.legend(fontsize=9, loc="lower center")
fig.tight_layout()
out = OUT / "neb_profile_before_after.png"
fig.savefig(out, dpi=150)
print(f"\nwrote {out}")
