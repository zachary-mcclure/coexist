"""
CAPSTONE: one MACE-MP-0 potential corrected on THERMODYNAMICS *and* KINETICS at once.

Every prior rung tuned one *kind* of target. The joint tune (mace_finetune_joint.py)
showed four THERMO phase-diagram targets can be satisfied together; the diffusion
tune (mace_finetune_diffusion.py) showed both halves of the Cu self-diffusion
activation energy can be. But those two live in different observables — a static
mixing/ordering energy vs a vacancy-mediated transport barrier — and a TENSION was
already observed: the standalone Cu-diffusion tune dragged the Cu-Ni mixing Omega
from 42 -> 27 meV (T_c 246 -> 159 K), i.e. fixing kinetics quietly broke a
thermodynamic target.

This script puts ALL SIX in one joint loss on a single potential:

  THERMO (from mace_finetune_joint):
    Cu-Ni consolute T_c   (via omega_cuni)            -> 625 K  (Omega -> 2kB*625)
    Ni-Al gamma' solvus   (line-compound solver)      -> x_Al 0.13
    Si fcc-diamond dE                                  -> 500 meV
    Zn fcc-hcp dE                                      -> 30 meV
  KINETIC (from mace_finetune_diffusion, frozen-geometry reverse-mode channel):
    Cu vacancy formation  E_f = E(vac) - (N-1)/N*E(perfect)  -> 1.28 eV
    Cu migration          E_m = E(saddle) - E(vac)  [NEB climb] -> 0.71 eV

Loss = thermo[(Tc,xAl,Si,Zn)] + kinetic[(E_m,E_f)]
       + anchor*agcu_Omega_anchor + reg*||theta-theta0||^2
with the Ag-Cu Omega control anchored and the property benchmark as whack-a-mole guard.

THE QUESTION: does joint optimization satisfy thermo AND kinetics simultaneously,
or do they trade off? Specifically — does the capstone keep Cu-Ni T_c on target
(Omega UP to ~105) while STILL fixing Cu diffusion (which alone pushed Omega DOWN
to 27)? Reported explicitly at the end.

Run (validate): python examples/mace_finetune_capstone.py --reps 2 --steps 40
Run (real):     python examples/mace_finetune_capstone.py --reps 3 --steps 200 \
                    --subset all --reg 0.1 --anchor 5 --benchmark
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
K_B = 8.617333e-5

# ── thermo targets (from mace_finetune_joint) ──────────────────────────────────
TC_CUNI, XAL_NIAL, T_NIAL = 625.0, 0.13, 1000.0
SI_TARGET, ZN_TARGET = 0.500, 0.030          # eV
OM_CUNI_TARGET = 2 * K_B * TC_CUNI           # eV, consolute -> Omega

# ── kinetic targets (from mace_finetune_diffusion), Cu ─────────────────────────
A0_CU = 3.615
TGT_EM, TGT_EF = 0.71, 1.28                  # eV, experiment
Q_EXP, D0_EXP = 2.04, 0.62e-4                # Cu: Q ~ 2.04 eV, D0 ~ 0.62 cm^2/s
NU_STAR = 5.0e12                             # attempt frequency, Hz
F_CORR = 0.781                              # fcc vacancy correlation factor

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--images", type=int, default=5)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--reg", type=float, default=0.1)
ap.add_argument("--anchor", type=float, default=5.0,
                help="weight on holding Ag-Cu Omega (a control) at its pretrained value")
ap.add_argument("--subset", choices=["readout", "all"], default="all",
                help="which weights to tune — 'all' tests whether readout capacity is the bottleneck")
ap.add_argument("--device", default="cpu")
ap.add_argument("--temp", type=float, default=1000.0, help="T (K) for the D comparison")
ap.add_argument("--benchmark", action="store_true")
args = ap.parse_args()
REPS, N = args.reps, 4 * args.reps**3
calc = mace_mp(model="small", device=args.device, default_dtype="float64")
MODEL = calc.models[0]
rng = np.random.default_rng(0)


def relax(atoms, fmax=0.04):
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=250 if REPS == 3 else 150)
    return calc._atoms_to_batch(atoms).to_dict(), len(atoms)


def relax_pos(atoms, fmax=0.03):
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=200)
    return atoms


def relax_cell(atoms, fmax=0.03):
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=200)
    return atoms


def E(b):
    return MODEL(b, compute_force=False, training=True)["energy"].sum().cpu()


def fcc(el, a):
    return bulk(el, "fcc", a=a, cubic=True).repeat((REPS,) * 3)


def fcc_mix(elA, elB, a, frac):
    at = fcc(elA, a); nums = at.get_atomic_numbers()
    nums[rng.permutation(N)[: int(round(frac * N))]] = bulk(elB, "fcc").numbers[0]
    at.set_atomic_numbers(nums); return at


def line_compound_solvus(om_g, Hf, c0, T, n=80):
    def G(x): return om_g * x * (1 - x) + K_B * T * (x * torch.log(x) + (1 - x) * torch.log(1 - x))
    def dG(x): return om_g * (1 - 2 * x) + K_B * T * torch.log(x / (1 - x))
    u = torch.tensor(np.log(0.05 / 0.95))
    for _ in range(n):
        x = torch.sigmoid(u) * c0
        r = G(x) + dG(x) * (c0 - x) - Hf
        d2 = -2 * om_g + K_B * T * (1 / x + 1 / (1 - x)); dr = d2 * (c0 - x) * (torch.sigmoid(u) * (1 - torch.sigmoid(u)) * c0)
        u = u - r / dr
    return torch.sigmoid(u) * c0


# ── THERMO structures (frozen after relaxation) ────────────────────────────────
print(f"Relaxing thermo target structures ({N} atoms)…")
S = {}
S["cu"], _ = relax(fcc("Cu", 3.62));  S["ni"], _ = relax(fcc("Ni", 3.52))
S["al"], _ = relax(fcc("Al", 4.05))
S["cuni"], _ = relax(fcc_mix("Cu", "Ni", 3.57, 0.5))
S["nial_g"], _ = relax(fcc_mix("Ni", "Al", 3.55, 0.125))
gp = fcc("Ni", 3.57); sc = gp.get_scaled_positions()
corner = np.all(np.abs(sc * REPS - np.round(sc * REPS)) < 1e-6, axis=1)
nums = gp.get_atomic_numbers(); nums[np.where(corner)[0]] = 13; gp.set_atomic_numbers(nums)
S["nial_gp"], NGP = relax(gp)
S["si_fcc"], NSF = relax(bulk("Si", "fcc", a=3.9, cubic=True).repeat((REPS,) * 3))
S["si_dia"], NSD = relax(bulk("Si", "diamond", a=5.43, cubic=True).repeat((max(1, REPS - 1),) * 3))
S["zn_fcc"], NZF = relax(bulk("Zn", "fcc", a=3.9, cubic=True).repeat((REPS,) * 3))
S["zn_hcp"], NZH = relax(bulk("Zn", "hcp", a=2.66, c=4.95).repeat((REPS,) * 3))
# Ag-Cu is a CONTROL (MACE gets its eutectic right); anchor it
S["ag"], _ = relax(fcc("Ag", 4.09))
S["agcu"], _ = relax(fcc_mix("Ag", "Cu", 3.85, 0.5))


def omega_agcu(): return 4 * (E(S["agcu"]) / N - 0.5 * E(S["ag"]) / N - 0.5 * E(S["cu"]) / N)
def omega_cuni(): return 4 * (E(S["cuni"]) / N - 0.5 * E(S["cu"]) / N - 0.5 * E(S["ni"]) / N)
def nial_solvus():
    x = 0.125
    dH = E(S["nial_g"]) / N - (1 - x) * E(S["ni"]) / N - x * E(S["al"]) / N
    Hf = E(S["nial_gp"]) / NGP - 0.75 * E(S["ni"]) / N - 0.25 * E(S["al"]) / N
    return line_compound_solvus(dH / (x * (1 - x)), Hf, 0.25, T_NIAL)
def si_dE(): return E(S["si_fcc"]) / NSF - E(S["si_dia"]) / NSD
def zn_dE(): return E(S["zn_fcc"]) / NZF - E(S["zn_hcp"]) / NZH


# ── KINETIC structures: Cu perfect + vacancy + NEB saddle (frozen) ─────────────
print(f"Building Cu perfect + vacancy cells + migration path ({REPS}x{REPS}x{REPS} fcc)…")
perfect = relax_cell(bulk("Cu", "fcc", a=A0_CU, cubic=True).repeat((REPS,) * 3))
Nsite = len(perfect)

base = bulk("Cu", "fcc", a=A0_CU, cubic=True).repeat((REPS,) * 3)
posA = base.positions[0].copy()
d = base.get_distances(0, range(len(base)), mic=True); d[0] = 1e9
Bnn = int(np.argmin(d)); posB = base.positions[Bnn].copy()
initial = base.copy(); del initial[0]                       # vacancy at A (= vacancy cell)
Bi = Bnn - 1                                                # B's index after deleting 0
final = initial.copy()
dvec, _ = find_mic((posA - posB).reshape(1, 3), base.cell, base.pbc)
final.positions[Bi] = posB + dvec[0]                       # atom B hops into A

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
e_perf_pa = perfect.get_potential_energy() / Nsite
Ef_relaxed = initial.get_potential_energy() - (Nsite - 1) * e_perf_pa
print(f"   NEB barrier (pretrained) E_m = {Em_neb:.3f} eV (exp {TGT_EM:.2f}) | "
      f"relaxed E_f = {Ef_relaxed:.3f} eV (exp {TGT_EF:.2f})")

# freeze the geometries; E_f, E_m become functions of the weights only
b_perf = calc._atoms_to_batch(perfect.copy()).to_dict()
b_init = calc._atoms_to_batch(initial.copy()).to_dict()
b_sad = calc._atoms_to_batch(saddle.copy()).to_dict()


def e_form(): return E(b_init) - (Nsite - 1) / Nsite * E(b_perf)
def barrier(): return E(b_sad) - E(b_init)


# ── trainable weights ──────────────────────────────────────────────────────────
if args.subset == "all":
    trainable = list(MODEL.named_parameters())
else:
    trainable = [(n, p) for n, p in MODEL.named_parameters() if "readout" in n]
for _, p in MODEL.named_parameters(): p.requires_grad_(False)
for _, p in trainable: p.requires_grad_(True)
params = [p for _, p in trainable]; theta0 = [p.detach().clone() for p in params]
OM_AGCU0 = omega_agcu().detach()
from tune_diag import theta0_norm, rel_move, move_stats
THETA0N = theta0_norm(theta0)
print(f"Tuning {sum(p.numel() for p in params)} weights ({args.subset})")


def state():
    with torch.no_grad():
        return dict(CuNi_Tc=(omega_cuni() / (2 * K_B)).item(), NiAl_xAl=nial_solvus().item(),
                    Si_dE=si_dE().item() * 1e3, Zn_dE=zn_dE().item() * 1e3,
                    Cu_Ef=e_form().item(), Cu_Em=barrier().item(),
                    CuNi_Omega=omega_cuni().item() * 1e3, AgCu_Omega=omega_agcu().item() * 1e3)


pre = state()
pre["Cu_Q"] = pre["Cu_Ef"] + pre["Cu_Em"]
print("\nPretrained targets:")
print(f"  THERMO : Cu-Ni T_c {pre['CuNi_Tc']:.0f}K (->625)  Ni-Al x_Al {pre['NiAl_xAl']:.3f} (->0.13)  "
      f"Si {pre['Si_dE']:+.0f} (->500)  Zn {pre['Zn_dE']:+.0f}meV (->30)")
print(f"  KINETIC: Cu E_f {pre['Cu_Ef']:.3f} (->1.28)  E_m {pre['Cu_Em']:.3f} (->0.71)  "
      f"Q {pre['Cu_Q']:.3f} eV (exp 2.04)")
print(f"  Cu-Ni Omega {pre['CuNi_Omega']:+.0f} meV | Ag-Cu Omega {pre['AgCu_Omega']:+.0f} meV (anchored)")

if args.benchmark:
    import sys; sys.path.insert(0, str(Path(__file__).parent)); import mace_benchmark as mb
    print("\n[benchmark] BEFORE…")
    for p in params: p.requires_grad_(False)
    bpre = mb.evaluate(calc, reps=REPS)
    for p in params: p.requires_grad_(True)

opt = torch.optim.Adam(params, lr=args.lr)
print(f"\n[capstone fine-tune] 6 targets (4 thermo + 2 kinetic), {args.steps} steps, "
      f"lr {args.lr}, reg {args.reg}, anchor {args.anchor}")
for step in range(args.steps):
    opt.zero_grad()
    # thermo
    l_cuni = ((omega_cuni() - OM_CUNI_TARGET) / OM_CUNI_TARGET) ** 2
    l_nial = ((nial_solvus() - XAL_NIAL) / XAL_NIAL) ** 2
    l_si = ((si_dE() - SI_TARGET) / SI_TARGET) ** 2
    l_zn = ((zn_dE() - ZN_TARGET) / max(ZN_TARGET, 0.03)) ** 2
    # kinetic
    em = barrier(); ef = e_form()
    l_em = ((em - TGT_EM) / TGT_EM) ** 2
    l_ef = ((ef - TGT_EF) / TGT_EF) ** 2
    # control anchor + reg
    l_anchor = ((omega_agcu() - OM_AGCU0) / OM_AGCU0) ** 2
    reg = sum(((p - t) ** 2).sum() for p, t in zip(params, theta0)).cpu()
    loss = (l_cuni + l_nial + l_si + l_zn + l_em + l_ef + args.anchor * l_anchor + args.reg * reg)
    loss.backward(); opt.step()
    if step % max(1, args.steps // 15) == 0 or step == args.steps - 1:
        s = state()
        tgt_loss = float(l_cuni + l_nial + l_si + l_zn + l_em + l_ef)
        print(f"   step {step:4d}  Tc {s['CuNi_Tc']:5.0f}  xAl {s['NiAl_xAl']:.3f}  "
              f"Si {s['Si_dE']:+5.0f}  Zn {s['Zn_dE']:+5.0f}  "
              f"E_f {s['Cu_Ef']:.3f}  E_m {s['Cu_Em']:.3f}  "
              f"CuNi_W {s['CuNi_Omega']:+4.0f}  AgCu {s['AgCu_Omega']:+4.0f}  "
              f"dW {rel_move(params, theta0, THETA0N) * 100:.3f}%  loss {tgt_loss:.3f}")

post = state()
post["Cu_Q"] = post["Cu_Ef"] + post["Cu_Em"]

print("\nCapstone result (pretrained -> tuned, target):")
rows = [("CuNi_Tc", 625, "K"), ("NiAl_xAl", 0.13, ""), ("Si_dE", 500, "meV"),
        ("Zn_dE", 30, "meV"), ("Cu_Ef", 1.28, "eV"), ("Cu_Em", 0.71, "eV")]
for k, tgt, unit in rows:
    moved = "toward" if abs(post[k] - tgt) < abs(pre[k] - tgt) else "AWAY"
    print(f"   {k:9s} {pre[k]:+.3f} -> {post[k]:+.3f}  (target {tgt} {unit})  [{moved}]")
print(f"   Cu self-diffusion Q = E_f+E_m: {pre['Cu_Q']:.3f} -> {post['Cu_Q']:.3f} eV  (experiment {Q_EXP:.2f})  "
      f"|Q-Qexp|: {abs(pre['Cu_Q'] - Q_EXP):.3f} -> {abs(post['Cu_Q'] - Q_EXP):.3f}")

# ── diffusion coefficient (D ∝ exp(-Q/kT)) ─────────────────────────────────────
a_jump = A0_CU / np.sqrt(2) * 1e-10
D0 = F_CORR * a_jump ** 2 * NU_STAR
T = args.temp
D_pre = D0 * np.exp(-pre["Cu_Q"] / (K_B * T))
D_post = D0 * np.exp(-post["Cu_Q"] / (K_B * T))
D_exp = D0_EXP * np.exp(-Q_EXP / (K_B * T))
print(f"\nDiffusion at T = {T:.0f} K  (D0 = {D0 * 1e4:.2e} cm^2/s):")
print(f"   D pre  = {D_pre * 1e4:.2e}  D post = {D_post * 1e4:.2e}  (exp {D_exp * 1e4:.2e} cm^2/s)")

# ── the key thermo<->kinetic tension read-out ──────────────────────────────────
print("\n=== THERMO <-> KINETIC coexistence check ===")
print(f"   Cu-Ni Omega : {pre['CuNi_Omega']:+.0f} -> {post['CuNi_Omega']:+.0f} meV  (target ~{OM_CUNI_TARGET*1e3:.0f})")
print(f"   Cu-Ni T_c   : {pre['CuNi_Tc']:.0f} -> {post['CuNi_Tc']:.0f} K  (target 625)")
print(f"   Cu E_m      : {pre['Cu_Em']:.3f} -> {post['Cu_Em']:.3f} eV  (target 0.71)")
print(f"   Cu E_f      : {pre['Cu_Ef']:.3f} -> {post['Cu_Ef']:.3f} eV  (target 1.28)")
print("   (diffusion-ONLY tune dragged Cu-Ni Omega 42->27 meV / T_c 246->159 K;")
print("    did the JOINT tune keep Cu-Ni on target while fixing Cu diffusion?)")

tag = f"capstone_{args.subset}_{N}_reg{args.reg}_anc{args.anchor}"
ckpt = OUT / f"mace_finetune_{tag}.pt"
torch.save({n: p.detach().cpu() for (n, _), p in zip(trainable, params)}, ckpt)
result = {"subset": args.subset, "N": N, "reg": args.reg, "anchor": args.anchor,
          "targets": {k: {"pre": pre[k], "post": post[k], "target": tgt}
                      for k, tgt, _ in rows},
          "CuNi_Omega_pre": pre["CuNi_Omega"], "CuNi_Omega_post": post["CuNi_Omega"],
          "CuNi_Omega_target_meV": OM_CUNI_TARGET * 1e3,
          "AgCu_Omega_pre": pre["AgCu_Omega"], "AgCu_Omega_post": post["AgCu_Omega"],
          "Cu_Q_pre": pre["Cu_Q"], "Cu_Q_post": post["Cu_Q"], "Q_exp": Q_EXP,
          "Em_neb_pretrained": Em_neb, "Ef_relaxed_pretrained": Ef_relaxed,
          "temp_K": T, "D0_cm2s": D0 * 1e4,
          "D_pre_cm2s": D_pre * 1e4, "D_post_cm2s": D_post * 1e4, "D_exp_cm2s": D_exp * 1e4,
          "pre": pre, "post": post, "checkpoint": ckpt.name}
result.update(move_stats(params, theta0))

if args.benchmark:
    print("\n[benchmark] AFTER…")
    for p in params: p.requires_grad_(False)
    bpost = mb.evaluate(calc, reps=REPS)
    print("\n[benchmark] capstone whack-a-mole check "
          "(did 6-way thermo+kinetic tune hold the controls?):")
    controls_ok = mb.diff_dicts(bpre, bpost)
    result.update(bench_pre=bpre, bench_post=bpost, controls_ok=bool(controls_ok))

json.dump(result, open(OUT / f"mace_finetune_{tag}.json", "w"), indent=1)
print(f"\nwrote {OUT / f'mace_finetune_{tag}.json'} and {ckpt.name}")
