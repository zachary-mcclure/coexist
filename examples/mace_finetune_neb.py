"""
Kinetics rung: tune a vacancy-MIGRATION BARRIER into the potential.

Reproducing a phase diagram fixes the thermodynamic driving forces (Section:
fine-tune); it does NOT fix the barriers that set rates. This adds the missing
kinetic half with the SAME reverse-mode loss, on a differentiable migration
barrier:

  1. Build an fcc supercell with a vacancy; the initial and final states are the
     vacancy on two adjacent sites (a <110> nearest-neighbor hop).
  2. Get the minimum-energy path + saddle with a climbing-image NEB (pretrained
     potential), then FREEZE the initial and saddle geometries (frozen-geometry
     scope, as elsewhere in the paper; the path's relaxation response to the
     weights is omitted).
  3. E_m(theta) = E(saddle; theta) - E(initial; theta) is then differentiable
     w.r.t. the weights. Adam tunes them toward the target barrier (Cu ~0.71 eV,
     Ni ~1.04 eV experiment), with a stay-near-pretrained regularizer and the
     property benchmark as a guard.

Shows kinetics + thermodynamics in one potential: does correcting the migration
barrier leave the lattice constants, mixing energetics, and phase boundaries
(the benchmark controls) intact?

Run:  python examples/mace_finetune_neb.py --element Cu --target 0.71 \
          --images 5 --steps 150 --subset all --reg 0.1 --device cuda --benchmark
"""
import argparse, json, warnings
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

OUT = Path(__file__).parent / "output"; OUT.mkdir(exist_ok=True)
A0 = {"Cu": 3.615, "Ni": 3.524, "Al": 4.050, "Ag": 4.085}
EXP_EM = {"Cu": 0.71, "Ni": 1.04, "Al": 0.61, "Ag": 0.66}   # migration energies, eV

ap = argparse.ArgumentParser()
ap.add_argument("--element", default="Cu")
ap.add_argument("--reps", type=int, default=3)              # 3 -> 108 sites (107 atoms + vacancy)
ap.add_argument("--images", type=int, default=5)
ap.add_argument("--target", type=float, default=None, help="target E_m (eV); default = experiment")
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--reg", type=float, default=0.1)
ap.add_argument("--anchor", type=float, default=5.0,
                help="weight on holding Ag-Cu Omega (the sensitive control) put")
ap.add_argument("--subset", choices=["readout", "all"], default="all")
ap.add_argument("--device", default="cpu")
ap.add_argument("--benchmark", action="store_true")
args = ap.parse_args()
EL = args.element
TARGET = args.target if args.target is not None else EXP_EM[EL]
REPS = args.reps

calc = mace_mp(model="small", device=args.device, default_dtype="float64")
MODEL = calc.models[0]


def relax_pos(atoms, fmax=0.03):
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=200)
    return atoms


# ── build the vacancy-migration endpoints (A -> B nearest-neighbor hop) ───────
print(f"Building {EL} vacancy-migration path ({REPS}x{REPS}x{REPS} fcc)…")
base = bulk(EL, "fcc", a=A0[EL], cubic=True).repeat((REPS,) * 3)
posA = base.positions[0].copy()
d = base.get_distances(0, range(len(base)), mic=True); d[0] = 1e9
B = int(np.argmin(d)); posB = base.positions[B].copy()
initial = base.copy(); del initial[0]                       # vacancy at A
Bi = B - 1                                                  # B's index after deleting 0
final = initial.copy()
dvec, _ = find_mic((posA - posB).reshape(1, 3), base.cell, base.pbc)
final.positions[Bi] = posB + dvec[0]                       # atom B hops into A -> vacancy at B

initial = relax_pos(initial); final = relax_pos(final)
print(f"   endpoints relaxed; NEB with {args.images} images (climbing)…")
images = [initial] + [initial.copy() for _ in range(args.images - 2)] + [final]
for im in images:                       # NEB requires a separate calculator per image
    im.calc = mace_mp(model="small", device=args.device, default_dtype="float64")
neb = NEB(images, climb=True, k=0.1)
neb.interpolate("idpp")
FIRE(neb, logfile=None).run(fmax=0.05, steps=120)
energies = np.array([im.get_potential_energy() for im in images])
saddle = images[int(np.argmax(energies))]
Em_neb = float(energies.max() - energies[0])
print(f"   NEB barrier (pretrained, relaxed path) E_m = {Em_neb:.3f} eV "
      f"(exp {EXP_EM.get(EL, float('nan')):.2f})")

# freeze the geometries; barrier becomes a function of weights only
b_init = calc._atoms_to_batch(initial.copy()).to_dict()
b_sad = calc._atoms_to_batch(saddle.copy()).to_dict()


def E(b):
    return MODEL(b, compute_force=False, training=True)["energy"].sum().cpu()


def barrier():
    return E(b_sad) - E(b_init)


# Ag-Cu mixing Omega is the hypersensitive control (it broke in the diagram
# tune too); anchor it at its pretrained value, exactly as mace_finetune_joint.
from ase.filters import FrechetCellFilter                       # noqa: E402
_rng = np.random.default_rng(0)
NAN = 4 * REPS ** 3


def _relax_cell(atoms, fmax=0.04):
    a = atoms.copy(); a.calc = calc
    FIRE(FrechetCellFilter(a), logfile=None).run(fmax=fmax, steps=200)
    return a


def _fcc(el, a):
    return bulk(el, "fcc", a=a, cubic=True).repeat((REPS,) * 3)


def _fcc_mix(elA, elB, a, frac):
    at = _fcc(elA, a); nums = at.get_atomic_numbers()
    nums[_rng.permutation(NAN)[: int(round(frac * NAN))]] = bulk(elB, "fcc").numbers[0]
    at.set_atomic_numbers(nums); return at


print("Building Ag-Cu anchor structures…")
b_ag = calc._atoms_to_batch(_relax_cell(_fcc("Ag", 4.09))).to_dict()
b_cu = calc._atoms_to_batch(_relax_cell(_fcc("Cu", 3.615))).to_dict()
b_agcu = calc._atoms_to_batch(_relax_cell(_fcc_mix("Ag", "Cu", 3.85, 0.5))).to_dict()


def omega_agcu():
    return 4.0 * (E(b_agcu) / NAN - 0.5 * E(b_ag) / NAN - 0.5 * E(b_cu) / NAN)


# ── trainable weights ─────────────────────────────────────────────────────────
if args.subset == "all":
    trainable = list(MODEL.named_parameters())
else:
    trainable = [(n, p) for n, p in MODEL.named_parameters() if "readout" in n]
for _, p in MODEL.named_parameters(): p.requires_grad_(False)
for _, p in trainable: p.requires_grad_(True)
params = [p for _, p in trainable]; theta0 = [p.detach().clone() for p in params]
print(f"Tuning {sum(p.numel() for p in params)} weights ({args.subset})")

EM0 = float(barrier().detach())
OM_AGCU0 = omega_agcu().detach()
print(f"\nFrozen-geometry barrier E_m = {EM0:.3f} eV  (target {TARGET:.2f} eV) | "
      f"Ag-Cu Omega {float(OM_AGCU0)*1e3:+.0f} meV (anchored)")

if args.benchmark:
    import sys; sys.path.insert(0, str(Path(__file__).parent)); import mace_benchmark as mb
    print("[benchmark] BEFORE…")
    for p in params: p.requires_grad_(False)
    bpre = mb.evaluate(calc, reps=REPS)
    for p in params: p.requires_grad_(True)

opt = torch.optim.Adam(params, lr=args.lr)
print(f"\n[NEB-barrier tune] {args.steps} steps, lr {args.lr}, reg {args.reg}")
for step in range(args.steps):
    opt.zero_grad()
    em = barrier()
    l_bar = ((em - TARGET) / TARGET) ** 2
    l_anchor = ((omega_agcu() - OM_AGCU0) / OM_AGCU0) ** 2
    reg = sum(((p - t) ** 2).sum() for p, t in zip(params, theta0)).cpu()
    (l_bar + args.anchor * l_anchor + args.reg * reg).backward(); opt.step()
    if step % max(1, args.steps // 12) == 0 or step == args.steps - 1:
        with torch.no_grad():
            print(f"   step {step:4d}  E_m {float(em):.3f} eV  "
                  f"AgCu {float(omega_agcu())*1e3:+.0f} meV  loss {float(l_bar):.4f}")

EM1 = float(barrier().detach())
print(f"\nMigration barrier E_m: {EM0:.3f} -> {EM1:.3f} eV  (target {TARGET:.2f})")

tag = f"neb_{EL}_{args.subset}_reg{args.reg}"
ckpt = OUT / f"mace_finetune_{tag}.pt"
torch.save({n: p.detach().cpu() for (n, _), p in zip(trainable, params)}, ckpt)
result = {"element": EL, "Em_neb_pretrained": Em_neb, "Em_frozen_pre": EM0,
          "Em_frozen_tuned": EM1, "target": TARGET, "subset": args.subset,
          "reg": args.reg, "checkpoint": ckpt.name}

if args.benchmark:
    print("\n[benchmark] AFTER…")
    for p in params: p.requires_grad_(False)
    bpost = mb.evaluate(calc, reps=REPS)
    print("\n[benchmark] barrier-tune whack-a-mole check "
          "(did kinetics tune without breaking thermodynamics?):")
    result["controls_ok"] = bool(mb.diff_dicts(bpre, bpost))
    result["bench_pre"], result["bench_post"] = bpre, bpost

json.dump(result, open(OUT / f"mace_finetune_{tag}.json", "w"), indent=1)
print(f"\nwrote {OUT / f'mace_finetune_{tag}.json'} and {ckpt.name}")
