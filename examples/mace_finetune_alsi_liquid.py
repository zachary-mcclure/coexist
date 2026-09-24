"""
Al-Si LIQUID re-tune: supply, by training, the liquid attraction the gradient
already measured as missing --- and move the eutectic onto experiment.

The forward paper reports that MACE-MP-0's Al-Si eutectic sits ~57 K high
because the liquid is too ideal: dT_e/dOmega_L = +304 K/eV and the diagram
wants ~ -0.19 eV of liquid attraction (Section: Al-Si / Discussion). The Si
lattice-stability tune could not fix this --- the eutectic is liquid-dominated,
not reference-dominated. So here we tune MACE's weights against the LIQUID
interaction itself, closing the loop: measure the deficit, then supply it.

Method (frozen-snapshot scope, using actual MD to build the ensemble):
  1. Melt pure Al, pure Si, and Al50Si50 with MACE MD (Langevin NVT), then
     freeze K snapshots of each.
  2. Omega_L(theta) = 4*(<E_mix>/n - 0.5<E_Al_liq>/n - 0.5<E_Si_liq>/n), an
     average over the frozen snapshots, differentiable w.r.t. the weights
     (the snapshots are fixed; the sampling derivative is out of scope, as for
     the sampled-liquid results elsewhere in the paper).
  3. Adam on the weights toward Omega_L_target (default -0.187 eV -> T_e ~850 K),
     with a stay-near-pretrained regularizer and the property benchmark as a
     guard (pure Al/Si and the solid controls must not regress).

Produces the tuned checkpoint + Omega_L pre/post; draw the recreated Al-Si
eutectic with examples/alsi_diagram_before_after.py.

Run (pod):  python examples/mace_finetune_alsi_liquid.py --device cuda \
                --reps 3 --md-steps 2000 --snapshots 8 --steps 150 \
                --subset all --reg 0.1 --benchmark
"""
import argparse, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase import units
from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import mace_mp

OUT = Path(__file__).parent / "output"; OUT.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3)             # cell size (fcc/dia supercell)
ap.add_argument("--md-steps", type=int, default=2000)      # equilibration steps before sampling
ap.add_argument("--md-temp", type=float, default=2000.0)   # melt temperature (K), above both T_m
ap.add_argument("--md-dt", type=float, default=2.0)        # fs
ap.add_argument("--snapshots", type=int, default=8)        # frozen configs per liquid, sampled after equil
ap.add_argument("--snap-every", type=int, default=100)     # MD steps between snapshots
ap.add_argument("--steps", type=int, default=150)          # tuning steps
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--reg", type=float, default=0.1)          # stay-near-pretrained (the credible knee)
ap.add_argument("--target-omega-l", type=float, default=-0.187,
                help="target liquid interaction, eV (-0.187 -> T_e ~850 K)")
ap.add_argument("--subset", choices=["readout", "all"], default="all")
ap.add_argument("--device", default="cpu")
ap.add_argument("--benchmark", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

REPS, N = args.reps, 4 * args.reps**3
torch.manual_seed(args.seed); np.random.seed(args.seed)
calc = mace_mp(model="small", device=args.device, default_dtype="float64")
MODEL = calc.models[0]
rng = np.random.default_rng(args.seed)


def relax(atoms, fmax=0.05):
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=200)
    return atoms


def substitute(atoms, el_new, frac):
    n = len(atoms); k = int(round(frac * n))
    nums = atoms.get_atomic_numbers()
    nums[rng.permutation(n)[:k]] = bulk(el_new, "fcc").numbers[0]
    atoms.set_atomic_numbers(nums); return atoms


def melt_and_sample(atoms, label):
    """Langevin NVT melt; return `snapshots` frozen MACE batches sampled after
    equilibration."""
    atoms = atoms.copy(); atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.md_temp)
    dyn = Langevin(atoms, args.md_dt * units.fs, temperature_K=args.md_temp,
                   friction=0.02, logfile=None)
    print(f"   [MD] {label}: equilibrating {args.md_steps} steps at {args.md_temp:.0f} K…")
    dyn.run(args.md_steps)
    batches = []
    for s in range(args.snapshots):
        dyn.run(args.snap_every)
        batches.append(calc._atoms_to_batch(atoms.copy()).to_dict())
    print(f"        collected {len(batches)} frozen snapshots ({len(atoms)} atoms)")
    return batches


def E(b):
    return MODEL(b, compute_force=False, training=True)["energy"].sum().cpu()


def mean_epa(batches, n):
    return sum(E(b) for b in batches) / (len(batches) * n)


# ── build the liquid ensembles (actual MD) ───────────────────────────────────
print("Melting Al, Si, and Al50Si50 with MACE MD…")
al_cell = bulk("Al", "fcc", a=4.05, cubic=True).repeat((REPS,) * 3)
si_cell = bulk("Si", "diamond", a=5.43, cubic=True).repeat((max(1, REPS - 1),) * 3)
mix_cell = substitute(bulk("Al", "fcc", a=4.20, cubic=True).repeat((REPS,) * 3), "Si", 0.5)
n_al, n_si, n_mix = len(al_cell), len(si_cell), len(mix_cell)
B_al  = melt_and_sample(al_cell,  "Al liquid")
B_si  = melt_and_sample(si_cell,  "Si liquid")
B_mix = melt_and_sample(mix_cell, "Al50Si50 liquid")


def omega_l():
    """Frozen-snapshot liquid interaction (eV/atom), regular-solution form at x=0.5."""
    return 4.0 * (mean_epa(B_mix, n_mix) - 0.5 * mean_epa(B_al, n_al)
                  - 0.5 * mean_epa(B_si, n_si))


# ── trainable weights ────────────────────────────────────────────────────────
if args.subset == "all":
    trainable = list(MODEL.named_parameters())
else:
    trainable = [(nm, p) for nm, p in MODEL.named_parameters() if "readout" in nm]
for _, p in MODEL.named_parameters(): p.requires_grad_(False)
for _, p in trainable: p.requires_grad_(True)
params = [p for _, p in trainable]; theta0 = [p.detach().clone() for p in params]
print(f"Tuning {sum(p.numel() for p in params)} weights ({args.subset})")

OM_L0 = float(omega_l().detach())
TARGET = args.target_omega_l
print(f"\nPretrained liquid interaction Omega_L = {OM_L0*1e3:+.0f} meV  "
      f"(target {TARGET*1e3:+.0f} meV -> Al-Si eutectic ~850 K)")

if args.benchmark:
    import sys; sys.path.insert(0, str(Path(__file__).parent)); import mace_benchmark as mb
    print("[benchmark] BEFORE…")
    for p in params: p.requires_grad_(False)
    bpre = mb.evaluate(calc, reps=REPS)
    for p in params: p.requires_grad_(True)

opt = torch.optim.Adam(params, lr=args.lr)
print(f"\n[Al-Si liquid tune] {args.steps} steps, lr {args.lr}, reg {args.reg}")
for step in range(args.steps):
    opt.zero_grad()
    oml = omega_l()
    l_liq = ((oml - TARGET) / abs(TARGET))**2
    reg = sum(((p - t)**2).sum() for p, t in zip(params, theta0)).cpu()
    (l_liq + args.reg * reg).backward(); opt.step()
    if step % max(1, args.steps // 12) == 0 or step == args.steps - 1:
        print(f"   step {step:4d}  Omega_L {float(oml)*1e3:+6.0f} meV  loss {float(l_liq):.4f}")

OM_L1 = float(omega_l().detach())
print(f"\nOmega_L: {OM_L0*1e3:+.0f} -> {OM_L1*1e3:+.0f} meV  (target {TARGET*1e3:+.0f})")

tag = f"alsi_liquid_{args.subset}_{N}_reg{args.reg}"
ckpt = OUT / f"mace_finetune_{tag}.pt"
torch.save({nm: p.detach().cpu() for (nm, _), p in zip(trainable, params)}, ckpt)
result = {"omega_l_pre": OM_L0, "omega_l_tuned": OM_L1, "target": TARGET,
          "reps": REPS, "subset": args.subset, "reg": args.reg,
          "checkpoint": ckpt.name}

if args.benchmark:
    print("\n[benchmark] AFTER…")
    for p in params: p.requires_grad_(False)
    bpost = mb.evaluate(calc, reps=REPS)
    print("\n[benchmark] Al-Si liquid tune whack-a-mole check:")
    result["controls_ok"] = bool(mb.diff_dicts(bpre, bpost))
    result["bench_pre"], result["bench_post"] = bpre, bpost

json.dump(result, open(OUT / f"mace_finetune_{tag}.json", "w"), indent=1)
print(f"\nwrote {OUT / f'mace_finetune_{tag}.json'} and {ckpt.name}")
print("Draw the recreated diagram:  python examples/alsi_diagram_before_after.py")
