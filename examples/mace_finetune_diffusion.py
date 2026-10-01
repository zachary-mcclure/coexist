"""
Diffusion rung: tune BOTH halves of the self-diffusion activation energy jointly.

The migration tune (mace_finetune_neb.py) corrected only E_m. But vacancy-mediated
self-diffusion is set by the ACTIVATION ENERGY Q = E_f + E_m (Vineyard TST), and
MACE-MP-0 underpredicts BOTH halves:

    E_f ~ 1.04 eV  vs experiment 1.28 eV
    E_m ~ 0.62 eV  vs experiment 0.71 eV
    Q   ~ 1.66 eV  vs experiment 2.04 eV

This jointly tunes the weights so that both land on experiment and Q -> 2.04 eV,
using the SAME frozen-geometry reverse-mode channel as the NEB tune:

  1. Relax (with the pretrained potential) a perfect fcc supercell, the vacancy
     cell (perfect minus one atom = the migration initial state) and the NEB
     climbing-image saddle; then FREEZE those geometries.
  2. E_f(theta) = E(vac; theta) - (N-1)/N * E(perfect; theta)       (frozen)
     E_m(theta) = E(saddle; theta) - E(vac; theta)                  (frozen)
     Q(theta)   = E_f + E_m
     all differentiable w.r.t. the weights (the geometries' relaxation response
     to the weights is omitted — the frozen-geometry scope used elsewhere).
  3. Adam minimises ((E_m-tgt_em)/tgt_em)^2 + ((E_f-tgt_ef)/tgt_ef)^2, with the
     Ag-Cu mixing Omega anchored (the hypersensitive control that breaks
     otherwise), a stay-near-pretrained regularizer, and the property benchmark
     as a whack-a-mole guard.

Run (fast check): python examples/mace_finetune_diffusion.py --reps 2 --steps 30
Run (real):       python examples/mace_finetune_diffusion.py --reps 3 --steps 150 \
                      --subset all --reg 0.1 --anchor 5 --target-em 0.71 \
                      --target-ef 1.28 --benchmark
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
from ase.filters import FrechetCellFilter
try:
    from ase.mep import NEB
except ImportError:
    from ase.neb import NEB
from mace.calculators import mace_mp

OUT = Path(__file__).parent / "output"; OUT.mkdir(exist_ok=True)
A0 = {"Cu": 3.615, "Ni": 3.524, "Al": 4.050, "Ag": 4.085}
EXP_EM = {"Cu": 0.71, "Ni": 1.04, "Al": 0.61, "Ag": 0.66}   # migration energies, eV
EXP_EF = {"Cu": 1.28, "Ni": 1.79, "Al": 0.68, "Ag": 1.11}   # vacancy formation energies, eV
# diffusion-coefficient constants (match tst_diffusivity.py)
KB = 8.617333e-5
Q_EXP, D0_EXP = 2.04, 0.62e-4          # Cu experiment: Q ~ 2.04 eV, D0 ~ 0.62 cm^2/s
NU_STAR = 5.0e12                        # attempt frequency, Hz (literature ~few THz; est.)
F_CORR = 0.781                         # fcc vacancy correlation factor

ap = argparse.ArgumentParser()
ap.add_argument("--element", default="Cu")
ap.add_argument("--reps", type=int, default=3)             # 3 -> 108 sites (107 atoms + vacancy)
ap.add_argument("--images", type=int, default=5)
ap.add_argument("--target-em", type=float, default=None, help="target E_m (eV); default = experiment")
ap.add_argument("--target-ef", type=float, default=None, help="target E_f (eV); default = experiment")
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--reg", type=float, default=0.1)
ap.add_argument("--anchor", type=float, default=5.0,
                help="weight on holding Ag-Cu Omega (the sensitive control) put")
ap.add_argument("--subset", choices=["readout", "all"], default="all")
ap.add_argument("--device", default="cpu")
ap.add_argument("--temp", type=float, default=1000.0, help="T (K) for the D comparison")
ap.add_argument("--benchmark", action="store_true")
args = ap.parse_args()
EL = args.element
TGT_EM = args.target_em if args.target_em is not None else EXP_EM[EL]
TGT_EF = args.target_ef if args.target_ef is not None else EXP_EF[EL]
REPS = args.reps

calc = mace_mp(model="small", device=args.device, default_dtype="float64")
MODEL = calc.models[0]


def relax_pos(atoms, fmax=0.03):
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=200)
    return atoms


def relax_cell(atoms, fmax=0.03):
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=200)
    return atoms


# ── build perfect cell + vacancy-migration endpoints (A -> B NN hop) ──────────
print(f"Building {EL} perfect + vacancy cells + migration path ({REPS}x{REPS}x{REPS} fcc)…")
perfect = relax_cell(bulk(EL, "fcc", a=A0[EL], cubic=True).repeat((REPS,) * 3))
Nsite = len(perfect)

base = bulk(EL, "fcc", a=A0[EL], cubic=True).repeat((REPS,) * 3)
posA = base.positions[0].copy()
d = base.get_distances(0, range(len(base)), mic=True); d[0] = 1e9
B = int(np.argmin(d)); posB = base.positions[B].copy()
initial = base.copy(); del initial[0]                       # vacancy at A (= the vacancy cell)
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

# pretrained vacancy formation energy from the (relaxed) static cells
e_perf_pa = perfect.get_potential_energy() / Nsite
Ef_relaxed = initial.get_potential_energy() - (Nsite - 1) * e_perf_pa
print(f"   NEB barrier (pretrained) E_m = {Em_neb:.3f} eV (exp {TGT_EM:.2f}) | "
      f"relaxed E_f = {Ef_relaxed:.3f} eV (exp {TGT_EF:.2f})")

# freeze the geometries; E_f, E_m become functions of the weights only
b_perf = calc._atoms_to_batch(perfect.copy()).to_dict()
b_init = calc._atoms_to_batch(initial.copy()).to_dict()     # vacancy cell = migration initial
b_sad = calc._atoms_to_batch(saddle.copy()).to_dict()


def E(b):
    return MODEL(b, compute_force=False, training=True)["energy"].sum().cpu()


def e_form():                                   # vacancy formation energy, eV
    return E(b_init) - (Nsite - 1) / Nsite * E(b_perf)


def barrier():                                  # migration energy, eV
    return E(b_sad) - E(b_init)


# Ag-Cu mixing Omega is the hypersensitive control (it broke in the diagram
# tune too); anchor it at its pretrained value, exactly as mace_finetune_neb.
_rng = np.random.default_rng(0)
NAN = 4 * REPS ** 3


def _fcc(el, a):
    return bulk(el, "fcc", a=a, cubic=True).repeat((REPS,) * 3)


def _fcc_mix(elA, elB, a, frac):
    at = _fcc(elA, a); nums = at.get_atomic_numbers()
    nums[_rng.permutation(NAN)[: int(round(frac * NAN))]] = bulk(elB, "fcc").numbers[0]
    at.set_atomic_numbers(nums); return at


print("Building Ag-Cu anchor structures…")
b_ag = calc._atoms_to_batch(relax_cell(_fcc("Ag", 4.09), fmax=0.04)).to_dict()
b_cu = calc._atoms_to_batch(relax_cell(_fcc("Cu", 3.615), fmax=0.04)).to_dict()
b_agcu = calc._atoms_to_batch(relax_cell(_fcc_mix("Ag", "Cu", 3.85, 0.5), fmax=0.04)).to_dict()


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
from tune_diag import theta0_norm, rel_move, move_stats   # weight-movement tracking
THETA0N = theta0_norm(theta0)
print(f"Tuning {sum(p.numel() for p in params)} weights ({args.subset})")

EM0 = float(barrier().detach())
EF0 = float(e_form().detach())
Q0 = EF0 + EM0
OM_AGCU0 = omega_agcu().detach()
print(f"\nFrozen-geometry pre-tune:  E_f = {EF0:.3f} (tgt {TGT_EF:.2f})  "
      f"E_m = {EM0:.3f} (tgt {TGT_EM:.2f})  Q = {Q0:.3f} eV (exp {Q_EXP:.2f}) | "
      f"Ag-Cu Omega {float(OM_AGCU0)*1e3:+.0f} meV (anchored)")

if args.benchmark:
    import sys; sys.path.insert(0, str(Path(__file__).parent)); import mace_benchmark as mb
    print("[benchmark] BEFORE…")
    for p in params: p.requires_grad_(False)
    bpre = mb.evaluate(calc, reps=REPS)
    for p in params: p.requires_grad_(True)

opt = torch.optim.Adam(params, lr=args.lr)
print(f"\n[diffusion joint tune] {args.steps} steps, lr {args.lr}, reg {args.reg}, anchor {args.anchor}")
for step in range(args.steps):
    opt.zero_grad()
    em = barrier()
    ef = e_form()
    l_em = ((em - TGT_EM) / TGT_EM) ** 2
    l_ef = ((ef - TGT_EF) / TGT_EF) ** 2
    l_anchor = ((omega_agcu() - OM_AGCU0) / OM_AGCU0) ** 2
    reg = sum(((p - t) ** 2).sum() for p, t in zip(params, theta0)).cpu()
    (l_em + l_ef + args.anchor * l_anchor + args.reg * reg).backward(); opt.step()
    if step % max(1, args.steps // 12) == 0 or step == args.steps - 1:
        with torch.no_grad():
            q = float(ef) + float(em)
            print(f"   step {step:4d}  E_m {float(em):.3f}  E_f {float(ef):.3f}  "
                  f"Q {q:.3f} eV  AgCu {float(omega_agcu())*1e3:+.0f} meV  "
                  f"dW {rel_move(params,theta0,THETA0N)*100:.3f}%  "
                  f"loss {float(l_em+l_ef):.4f}")

EM1 = float(barrier().detach())
EF1 = float(e_form().detach())
Q1 = EF1 + EM1
print(f"\nE_f: {EF0:.3f} -> {EF1:.3f} eV  (target {TGT_EF:.2f})")
print(f"E_m: {EM0:.3f} -> {EM1:.3f} eV  (target {TGT_EM:.2f})")
print(f"Q  : {Q0:.3f} -> {Q1:.3f} eV  (experiment {Q_EXP:.2f})  "
      f"|Q-Q_exp|: {abs(Q0-Q_EXP):.3f} -> {abs(Q1-Q_EXP):.3f}")

# ── diffusion coefficient (D ∝ exp(-Q/kT), same prefactor as tst_diffusivity) ──
a_jump = A0[EL] / np.sqrt(2) * 1e-10       # NN jump distance, m
D0 = F_CORR * a_jump ** 2 * NU_STAR        # m^2/s
T = args.temp
D_pre = D0 * np.exp(-Q0 / (KB * T))
D_post = D0 * np.exp(-Q1 / (KB * T))
D_exp = D0_EXP * np.exp(-Q_EXP / (KB * T))
print(f"\nDiffusion at T = {T:.0f} K  (D0 = {D0*1e4:.2e} cm^2/s):")
print(f"   D pre-tune  = {D_pre*1e4:.2e} cm^2/s  (exp {D_exp*1e4:.2e}, ratio {D_pre/D_exp:.2f}x)")
print(f"   D post-tune = {D_post*1e4:.2e} cm^2/s  (exp {D_exp*1e4:.2e}, ratio {D_post/D_exp:.2f}x)")
print(f"   tune changes the rate by {D_post/D_pre:.3e}x "
      f"(= exp((Q0-Q1)/kT)); |D/Dexp-1|: {abs(D_pre/D_exp-1):.2f} -> {abs(D_post/D_exp-1):.2f}")

tag = f"diffusion_{EL}_{args.subset}_reg{args.reg}_anc{args.anchor}"
ckpt = OUT / f"mace_finetune_{tag}.pt"
torch.save({n: p.detach().cpu() for (n, _), p in zip(trainable, params)}, ckpt)
result = {"element": EL, "Em_neb_pretrained": Em_neb, "Ef_relaxed_pretrained": Ef_relaxed,
          "Ef_pre": EF0, "Ef_post": EF1, "target_ef": TGT_EF,
          "Em_pre": EM0, "Em_post": EM1, "target_em": TGT_EM,
          "Q_pre": Q0, "Q_post": Q1, "Q_exp": Q_EXP,
          "temp_K": T, "D0_cm2s": D0 * 1e4,
          "D_pre_cm2s": D_pre * 1e4, "D_post_cm2s": D_post * 1e4, "D_exp_cm2s": D_exp * 1e4,
          "D_rate_change": D_post / D_pre,
          "subset": args.subset, "reg": args.reg, "anchor": args.anchor,
          "checkpoint": ckpt.name}
result.update(move_stats(params, theta0))   # n trained, rel L2, max move, %moved

if args.benchmark:
    print("\n[benchmark] AFTER…")
    for p in params: p.requires_grad_(False)
    bpost = mb.evaluate(calc, reps=REPS)
    print("\n[benchmark] joint-tune whack-a-mole check "
          "(did kinetics+formation tune without breaking thermodynamics?):")
    result["controls_ok"] = bool(mb.diff_dicts(bpre, bpost))
    result["bench_pre"], result["bench_post"] = bpre, bpost

json.dump(result, open(OUT / f"mace_finetune_{tag}.json", "w"), indent=1)
print(f"\nwrote {OUT / f'mace_finetune_{tag}.json'} and {ckpt.name}")
