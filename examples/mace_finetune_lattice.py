"""
Tune MACE against a lattice-stability target (the solid-tunable failures).

Some MACE-MP-0 phase-diagram errors are set by a wrong *lattice stability* — a
polymorph energy the model gets wrong — rather than a mixing term:
  * Si fcc-diamond: MACE +424 meV vs DFT ~500; the Si-side reference the
    Al-Si two-lattice eutectic is built on (the eutectic itself is liquid-
    limited, so this is the solid-tunable piece).
  * Zn fcc-hcp: MACE flattens/inverts it (~0 to -9 meV) though hcp is Zn's
    ground state (~+30 meV); feeds the Zn reference in Cu-Zn.

Same reverse-mode idea as the phase-diagram tunes: differentiate a scalar
target (here a per-atom polymorph energy difference) w.r.t. all the readout
weights and step them, with the property benchmark guarding against
whack-a-mole. Static energetics only.

Run:  python examples/mace_finetune_lattice.py --pair Si_fcc_dia --steps 200 --benchmark
      python examples/mace_finetune_lattice.py --pair Zn_fcc_hcp --steps 200 --benchmark
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

OUT = Path(__file__).parent / "output"; OUT.mkdir(exist_ok=True)

# pair -> (allotrope A, allotrope B, target dE(A-B) in eV toward DFT/experiment)
PAIRS = {
    "Si_fcc_dia": (("Si", "fcc", dict(a=3.90, cubic=True)),
                   ("Si", "diamond", dict(a=5.43, cubic=True)), 0.500),
    "Zn_fcc_hcp": (("Zn", "fcc", dict(a=3.90, cubic=True)),
                   ("Zn", "hcp", dict(a=2.66, c=4.95)), 0.030),
}

ap = argparse.ArgumentParser()
ap.add_argument("--pair", choices=list(PAIRS), default="Si_fcc_dia")
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--reg", type=float, default=0.5)
ap.add_argument("--device", default="cpu")
ap.add_argument("--benchmark", action="store_true")
args = ap.parse_args()

calc = mace_mp(model="small", device=args.device, default_dtype="float64")
MODEL = calc.models[0]
(elA, latA, kwA), (elB, latB, kwB), TARGET = PAIRS[args.pair]


def relaxed(el, lat, kw):
    at = bulk(el, lat, **kw).repeat((args.reps,) * 3)
    at.calc = calc
    FIRE(FrechetCellFilter(at), logfile=None).run(fmax=0.03, steps=200)
    return calc._atoms_to_batch(at).to_dict(), len(at)


def energy(batch):
    return MODEL(batch, compute_force=False, training=True)["energy"].sum()


print(f"Relaxing {args.pair} allotropes…")
bA, nA = relaxed(elA, latA, kwA)
bB, nB = relaxed(elB, latB, kwB)


def dE_eV():
    return energy(bA) / nA - energy(bB) / nB


trainable = [(n, p) for n, p in MODEL.named_parameters() if "readout" in n]
for _, p in MODEL.named_parameters():
    p.requires_grad_(False)
for _, p in trainable:
    p.requires_grad_(True)
params = [p for _, p in trainable]
theta0 = [p.detach().clone() for p in params]
print(f"Trainable: {sum(p.numel() for p in params)} readout weights")

d0 = dE_eV().item()
print(f"\nPretrained {args.pair}: dE = {d0*1e3:+.1f} meV  (target {TARGET*1e3:+.0f})")

if args.benchmark:
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    import mace_benchmark as _mb
    print("[benchmark] scoring panel BEFORE…")
    _pre = _mb.evaluate(calc)

opt = torch.optim.Adam(params, lr=args.lr)
print(f"[fine-tune] {args.steps} steps, lr {args.lr}, reg {args.reg}")
for step in range(args.steps):
    opt.zero_grad()
    d = dE_eV()
    reg = sum(((p - t) ** 2).sum() for p, t in zip(params, theta0))
    loss = ((d - TARGET) / max(abs(TARGET), 0.03)) ** 2 + args.reg * reg
    loss.backward(); opt.step()
    if step % max(1, args.steps // 12) == 0 or step == args.steps - 1:
        print(f"   step {step:4d}  dE {d.item()*1e3:+7.1f} meV  reg {float(reg):.2e}")

dF = dE_eV().item()
print(f"\nTuned {args.pair}: dE {d0*1e3:+.1f} -> {dF*1e3:+.1f} meV (target {TARGET*1e3:+.0f})")

ckpt = OUT / f"mace_finetune_lattice_{args.pair}_{args.reps}.pt"
torch.save({n: p.detach().cpu() for (n, _), p in zip(trainable, params)}, ckpt)
json.dump({"pair": args.pair, "dE0_meV": d0*1e3, "dE_tuned_meV": dF*1e3,
           "target_meV": TARGET*1e3, "checkpoint": ckpt.name},
          open(OUT / f"mace_finetune_lattice_{args.pair}_{args.reps}.json", "w"), indent=1)

if args.benchmark:
    print("[benchmark] scoring panel AFTER…")
    for p in params: p.requires_grad_(False)
    _post = _mb.evaluate(calc)
    print("\n[benchmark] whack-a-mole check (pre -> post):")
    _mb.diff_dicts(_pre, _post)
print(f"\nWrote {ckpt.name} + json")
