"""
Joint multi-target tune: fix several phase diagrams at once, or do they fight?

Tuning one target against a phase diagram is easy. The real question for
"systematically repairing a foundation-model potential" is whether several
targets can be satisfied simultaneously, or whether their gradients conflict
on the shared readout weights — whack-a-mole between *targets*, not just
target vs control.

We sum four solid-tunable target losses and tune the readout weights jointly:
  Cu-Ni consolute (via Omega),  Ni-Al gamma' solvus (through the line-compound
  solver),  Si fcc-diamond lattice stability,  Zn fcc-hcp ordering.
The property benchmark scores everything pre/post: did ALL targets improve, and
did the controls hold?

Run:  python examples/mace_finetune_joint.py --reps 2 --steps 300 --benchmark
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
K_B = 8.617333e-5
TC_CUNI, XAL_NIAL, T_NIAL = 625.0, 0.13, 1000.0
SI_TARGET, ZN_TARGET = 0.500, 0.030          # eV, toward DFT/experiment
OM_CUNI_TARGET = 2 * K_B * TC_CUNI           # eV, consolute -> Omega

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--reg", type=float, default=0.5)
ap.add_argument("--device", default="cpu")
ap.add_argument("--benchmark", action="store_true")
ap.add_argument("--anchor", type=float, default=5.0,
                help="weight on holding Ag-Cu (a control) at its pretrained value")
ap.add_argument("--subset", choices=["readout", "all"], default="readout",
                help="which weights to tune — 'all' tests whether readout capacity is the bottleneck")
args = ap.parse_args()
REPS, N = args.reps, 4 * args.reps**3
calc = mace_mp(model="small", device=args.device, default_dtype="float64")
MODEL = calc.models[0]
rng = np.random.default_rng(0)


def relax(atoms, fmax=0.04):
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=250 if REPS==3 else 150)
    return calc._atoms_to_batch(atoms).to_dict(), len(atoms)


def E(b):
    return MODEL(b, compute_force=False, training=True)["energy"].sum().cpu()


def fcc(el, a):
    return bulk(el, "fcc", a=a, cubic=True).repeat((REPS,) * 3)


def fcc_mix(elA, elB, a, frac):
    at = fcc(elA, a); nums = at.get_atomic_numbers()
    nums[rng.permutation(N)[: int(round(frac*N))]] = bulk(elB, "fcc").numbers[0]
    at.set_atomic_numbers(nums); return at


def line_compound_solvus(om_g, Hf, c0, T, n=80):
    def G(x): return om_g*x*(1-x) + K_B*T*(x*torch.log(x)+(1-x)*torch.log(1-x))
    def dG(x): return om_g*(1-2*x) + K_B*T*torch.log(x/(1-x))
    u = torch.tensor(np.log(0.05/0.95))
    for _ in range(n):
        x = torch.sigmoid(u)*c0
        r = G(x)+dG(x)*(c0-x)-Hf
        d2 = -2*om_g + K_B*T*(1/x+1/(1-x)); dr = d2*(c0-x)*(torch.sigmoid(u)*(1-torch.sigmoid(u))*c0)
        u = u - r/dr
    return torch.sigmoid(u)*c0


print(f"Relaxing all target structures ({N} atoms)…")
S = {}
S["cu"], _ = relax(fcc("Cu", 3.62));  S["ni"], _ = relax(fcc("Ni", 3.52))
S["al"], _ = relax(fcc("Al", 4.05))
S["cuni"], _ = relax(fcc_mix("Cu", "Ni", 3.57, 0.5))
S["nial_g"], _ = relax(fcc_mix("Ni", "Al", 3.55, 0.125))
gp = fcc("Ni", 3.57); sc = gp.get_scaled_positions()
corner = np.all(np.abs(sc*REPS - np.round(sc*REPS)) < 1e-6, axis=1)
nums = gp.get_atomic_numbers(); nums[np.where(corner)[0]] = 13; gp.set_atomic_numbers(nums)
S["nial_gp"], NGP = relax(gp)
S["si_fcc"], NSF = relax(bulk("Si","fcc",a=3.9,cubic=True).repeat((REPS,)*3))
S["si_dia"], NSD = relax(bulk("Si","diamond",a=5.43,cubic=True).repeat((max(1,REPS-1),)*3))
S["zn_fcc"], NZF = relax(bulk("Zn","fcc",a=3.9,cubic=True).repeat((REPS,)*3))
S["zn_hcp"], NZH = relax(bulk("Zn","hcp",a=2.66,c=4.95).repeat((REPS,)*3))
# Ag-Cu is a CONTROL (MACE gets its eutectic right); build it so we can anchor it
S["ag"], _ = relax(fcc("Ag", 4.09))
S["agcu"], _ = relax(fcc_mix("Ag", "Cu", 3.85, 0.5))


def omega_agcu(): return 4*(E(S["agcu"])/N - 0.5*E(S["ag"])/N - 0.5*E(S["cu"])/N)
def omega_cuni(): return 4*(E(S["cuni"])/N - 0.5*E(S["cu"])/N - 0.5*E(S["ni"])/N)
def nial_solvus():
    x = 0.125
    dH = E(S["nial_g"])/N - (1-x)*E(S["ni"])/N - x*E(S["al"])/N
    Hf = E(S["nial_gp"])/NGP - 0.75*E(S["ni"])/N - 0.25*E(S["al"])/N
    return line_compound_solvus(dH/(x*(1-x)), Hf, 0.25, T_NIAL)
def si_dE(): return E(S["si_fcc"])/NSF - E(S["si_dia"])/NSD
def zn_dE(): return E(S["zn_fcc"])/NZF - E(S["zn_hcp"])/NZH


if args.subset == "all":
    trainable = list(MODEL.named_parameters())
else:
    trainable = [(n, p) for n, p in MODEL.named_parameters() if "readout" in n]
for _, p in MODEL.named_parameters(): p.requires_grad_(False)
for _, p in trainable: p.requires_grad_(True)
params = [p for _, p in trainable]; theta0 = [p.detach().clone() for p in params]
OM_AGCU0 = omega_agcu().detach()   # anchor Ag-Cu (control) at its good pretrained value
from tune_diag import theta0_norm, rel_move, move_stats   # weight-movement tracking
THETA0N = theta0_norm(theta0)
print(f"Tuning {sum(p.numel() for p in params)} weights ({args.subset})")


def state():
    with torch.no_grad():
        return dict(CuNi_Tc=(omega_cuni()/(2*K_B)).item(), NiAl_xAl=nial_solvus().item(),
                    Si_dE=si_dE().item()*1e3, Zn_dE=zn_dE().item()*1e3,
                    AgCu_Omega=omega_agcu().item()*1e3)
pre = state()
print(f"\nPretrained targets: Cu-Ni T_c {pre['CuNi_Tc']:.0f}K (->625)  "
      f"Ni-Al x_Al {pre['NiAl_xAl']:.3f} (->0.13)  Si {pre['Si_dE']:+.0f} (->500)  "
      f"Zn {pre['Zn_dE']:+.0f}meV (->30)  | Ag-Cu Omega {pre['AgCu_Omega']:+.0f} meV (hold)")

if args.benchmark:
    import sys; sys.path.insert(0, str(Path(__file__).parent)); import mace_benchmark as mb
    print("[benchmark] BEFORE…");
    for p in params: p.requires_grad_(False)
    bpre = mb.evaluate(calc, reps=REPS)
    for p in params: p.requires_grad_(True)

opt = torch.optim.Adam(params, lr=args.lr)
print(f"\n[joint fine-tune] 4 targets, {args.steps} steps, lr {args.lr}, reg {args.reg}")
for step in range(args.steps):
    opt.zero_grad()
    l_cuni = ((omega_cuni() - OM_CUNI_TARGET)/OM_CUNI_TARGET)**2
    l_nial = ((nial_solvus() - XAL_NIAL)/XAL_NIAL)**2
    l_si   = ((si_dE() - SI_TARGET)/SI_TARGET)**2
    l_zn   = ((zn_dE() - ZN_TARGET)/max(ZN_TARGET,0.03))**2
    l_anchor = ((omega_agcu() - OM_AGCU0)/OM_AGCU0)**2   # keep the Ag-Cu control put
    reg = sum(((p-t)**2).sum() for p, t in zip(params, theta0)).cpu()
    (l_cuni + l_nial + l_si + l_zn + args.anchor*l_anchor + args.reg*reg).backward(); opt.step()
    if step % max(1, args.steps//12) == 0 or step == args.steps-1:
        s = state()
        print(f"   step {step:4d}  Tc {s['CuNi_Tc']:5.0f}  xAl {s['NiAl_xAl']:.3f}  "
              f"Si {s['Si_dE']:+5.0f}  Zn {s['Zn_dE']:+5.0f}  "
              f"AgCu {s['AgCu_Omega']:+5.0f}  dW {rel_move(params,theta0,THETA0N)*100:.3f}%"
              f"  loss {(l_cuni+l_nial+l_si+l_zn).item():.3f}")

post = state()
print("\nJoint result (pretrained -> tuned, target):")
for k, tgt in [("CuNi_Tc",625),("NiAl_xAl",0.13),("Si_dE",500),("Zn_dE",30)]:
    moved = "toward" if abs(post[k]-tgt) < abs(pre[k]-tgt) else "AWAY"
    print(f"   {k:9s} {pre[k]:+.3f} -> {post[k]:+.3f}  (target {tgt})  [{moved}]")
tag = f"{args.subset}_{N}_reg{args.reg}_anc{args.anchor}"
ckpt = OUT / f"mace_finetune_joint_{tag}.pt"
torch.save({n: p.detach().cpu() for (n, _), p in zip(trainable, params)}, ckpt)
result = {"subset": args.subset, "N": N, "reg": args.reg, "anchor": args.anchor,
          "pre": pre, "post": post, "checkpoint": ckpt.name}
result.update(move_stats(params, theta0))   # n trained, rel L2, max move, %moved

if args.benchmark:
    print("\n[benchmark] AFTER…")
    for p in params: p.requires_grad_(False)
    bpost = mb.evaluate(calc, reps=REPS)
    print("\n[benchmark] joint whack-a-mole check:")
    controls_ok = mb.diff_dicts(bpre, bpost)
    result.update(bench_pre=bpre, bench_post=bpost, controls_ok=bool(controls_ok))

json.dump(result, open(OUT/f"mace_finetune_joint_{tag}.json", "w"), indent=1)
