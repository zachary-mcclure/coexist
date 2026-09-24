"""
From a tuned barrier to a diffusion COEFFICIENT: does correcting kinetics move
the rate toward experiment?

The NEB tune (mace_finetune_neb.py) corrected MACE's Cu vacancy migration
barrier. A migration barrier is not yet a rate; vacancy-mediated self-diffusion
is

    D(T) = f a^2 nu* exp( -(E_f + E_m) / kT )                       (Vineyard TST)

with f the fcc correlation factor, a the jump distance, nu* the attempt
frequency, E_f the vacancy formation energy and E_m the migration energy. Only
E_m was tuned; E_f is a MACE prediction here, and nu* is taken from the
literature (the vibrational-entropy prefactor is a separate refinement, noted).
The robust, experiment-comparable quantity is the ACTIVATION ENERGY
Q = E_f + E_m, which the barrier tune directly improves.

D is reverse-mode differentiable through E_m (D ∝ exp(-E_m/kT), and dE_m/dtheta
is the NEB adjoint of mace_finetune_neb.py) -- so this is the same reverse-mode
channel, now expressed as a rate.

Run:  PYTHONPATH=. python examples/tst_diffusivity.py [--reps 2]
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

OUT = Path(__file__).parent / "output"
CKPT = OUT / "mace_finetune_neb_Cu_all_reg0.1.pt"
KB = 8.617333e-5
A0 = 3.615
# Cu experiment: Q ~ 2.04 eV, D0 ~ 0.62 cm^2/s; E_f ~ 1.28, E_m ~ 0.71 eV
Q_EXP, D0_EXP, EF_EXP, EM_EXP = 2.04, 0.62e-4, 1.28, 0.71
NU_STAR = 5.0e12          # attempt frequency, Hz (literature ~few THz; stated estimate)
F_CORR = 0.781            # fcc vacancy correlation factor

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--images", type=int, default=5)
ap.add_argument("--temps", default="1000,1200,1356")
args = ap.parse_args()
REPS = args.reps
TEMPS = [float(t) for t in args.temps.split(",")]
a_jump = A0 / np.sqrt(2) * 1e-10          # NN jump distance, m
calc = mace_mp(model="small", device="cpu", default_dtype="float64")


def relax_pos(atoms, fmax=0.03):
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=200)
    return atoms


def epa_total(atoms):
    a = atoms.copy(); a.calc = calc
    return a.get_potential_energy()


def overlay(ckpt):
    state = torch.load(ckpt, map_location="cpu")
    sd = dict(calc.models[0].named_parameters())
    for nm, w in state.items():
        if nm in sd: sd[nm].data.copy_(w)


# ── build perfect + vacancy cells and the migration path ──────────────────────
print(f"Building Cu cells + migration path ({REPS}x{REPS}x{REPS})…")
perfect = relax_pos(bulk("Cu", "fcc", a=A0, cubic=True).repeat((REPS,) * 3))
Nsite = len(perfect)
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
FIRE(neb, logfile=None).run(fmax=0.05, steps=120)


def kinetics(label):
    e_perf = epa_total(perfect) / Nsite
    E_f = epa_total(initial) - (Nsite - 1) * e_perf
    E = np.array([epa_total(im) for im in images])
    E_m = float(E.max() - E[0])
    Q = E_f + E_m
    D0 = F_CORR * a_jump**2 * NU_STAR
    print(f"\n[{label}]  E_f = {E_f:.3f} eV,  E_m = {E_m:.3f} eV,  "
          f"Q = E_f+E_m = {Q:.3f} eV   (exp: E_f {EF_EXP}, E_m {EM_EXP}, Q {Q_EXP})")
    print(f"          D0 = {D0*1e4:.2e} cm^2/s (nu*={NU_STAR:.0e} Hz, est.)  |  D(T):")
    for T in TEMPS:
        D = D0 * np.exp(-Q / (KB * T))
        Dexp = D0_EXP * np.exp(-Q_EXP / (KB * T))
        print(f"            T={T:6.0f} K   D = {D*1e4:.2e} cm^2/s   "
              f"(exp {Dexp*1e4:.2e},  ratio {D/Dexp:6.2f}x)")
    return E_f, E_m, Q


ef0, em0, q0 = kinetics("MACE-MP-0 pretrained")
if CKPT.exists():
    print(f"\nOverlaying barrier-tuned checkpoint {CKPT.name}…")
    overlay(CKPT)
    ef1, em1, q1 = kinetics("barrier-tuned")
    print(f"\n=== Effect of the barrier tune on the diffusion activation energy ===")
    print(f"    Q: {q0:.3f} -> {q1:.3f} eV   (experiment {Q_EXP:.2f});  "
          f"|Q-Q_exp|: {abs(q0-Q_EXP):.3f} -> {abs(q1-Q_EXP):.3f} eV")
    print(f"    at 1000 K the rate changes by exp((Q0-Q1)/kT) = "
          f"{np.exp((q0-q1)/(KB*1000)):.2f}x")
else:
    print(f"\n(no tuned checkpoint at {CKPT.name}; ran pretrained only)")
