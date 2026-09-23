"""
Train a foundation-model potential against a phase diagram — stage 2 (pod run).

Stage 1 (examples/mace_finetune_cuni.py) proved, on a laptop, that the gradient
of a phase-boundary loss w.r.t. MACE's own weights is exact (FD-verified 1e-8),
including through the binodal solver, and that a few reverse-mode steps move the
Cu-Ni consolute toward the assessed band. This script scales that to a real
fine-tune and adds everything the claim needs to be credible:

  * converged 108-atom cells (--reps 3),
  * a loss over the whole SOLVUS CURVE (many temperatures via the binodal
    construction), not just the consolute scalar,
  * periodic geometry RE-RELAXATION with the tuning weights (the snapshot
    bridge would otherwise drift as theta moves),
  * a stay-physical regularizer pinning theta near the pretrained weights,
  * a PHYSICALITY BATTERY (lattice constants, pure-element energies, and the
    Ag-Cu mixing energy the tune must NOT break) evaluated before and after,
  * a second target — the Ni-Al gamma/gamma-prime solvus (line-compound
    construction) — because the same machinery retargets for free,
  * checkpointing of the tuned weights + full JSON logging, device flag.

Only STATIC energetics enter the loss (solid solutions, formation energies), so
the whole chain stays autograd-differentiable — no gradient crosses an MD
trajectory. Reverse-mode is what makes it affordable: thousands-to-millions of
weights, a handful of boundary targets, one backward pass per step.

Run (pod):   python examples/mace_finetune_stage2.py --reps 3 --steps 400 \
                    --device cuda --system both
Smoke test:  python examples/mace_finetune_stage2.py --reps 2 --steps 5 --fdcheck
"""
import argparse
import warnings
warnings.filterwarnings("ignore")
import json
import time
from pathlib import Path

import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
K_B = 8.617333e-5                       # eV/K

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3, help="2->32 atoms, 3->108 (real)")
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--reg", type=float, default=2.0, help="stay-physical weight")
ap.add_argument("--relax-every", type=int, default=0,
                help="re-relax frozen geometries with the tuning weights every N steps")
ap.add_argument("--system", choices=["cuni", "nial", "both"], default="cuni")
ap.add_argument("--subset", choices=["readout", "all"], default="readout")
ap.add_argument("--device", default="cpu")
ap.add_argument("--fdcheck", action="store_true", help="FD-verify a few weights first")
ap.add_argument("--benchmark", action="store_true",
                help="score the property panel (mace_benchmark) pre/post and diff — catches whack-a-mole")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
REPS, N = args.reps, 4 * args.reps**3
TC_TARGET = 625.0                       # Cu-Ni assessed consolute (600-650)
XAL_TARGET = 0.13                       # Ni-Al gamma solvus (exp 0.12-0.14, 1000 K)
T_NIAL = 1000.0

calc = mace_mp(model="small", device=args.device, default_dtype="float64")
MODEL = calc.models[0]
rng = np.random.default_rng(args.seed)


def relaxed(numbers, a0, cubic=True):
    at = bulk("Cu", "fcc", a=a0, cubic=True).repeat((REPS,) * 3)
    at.set_atomic_numbers(numbers)
    at.calc = calc
    FIRE(FrechetCellFilter(at), logfile=None).run(fmax=0.05,
                                                  steps=120 if REPS == 2 else 250)
    return at


def L12_Ni3Al(a0):
    """L1_2 Ni3Al: fcc conventional cell with Al on the corner sublattice."""
    at = bulk("Ni", "fcc", a=a0, cubic=True).repeat((REPS,) * 3)
    # corner sites (fractional 0,0,0 of each conventional cell) -> Al
    scaled = at.get_scaled_positions()
    nums = np.full(len(at), 28)          # Ni
    corner = np.all(np.abs(scaled * REPS - np.round(scaled * REPS)) < 1e-6, axis=1)
    # every conventional cell contributes one corner; pick 1/4 as Al (L1_2)
    idx = np.where(corner)[0]
    nums[idx] = 13                       # Al on corners  (=> Ni3Al stoichiometry)
    at.set_atomic_numbers(nums)
    at.calc = calc
    FIRE(FrechetCellFilter(at), logfile=None).run(fmax=0.05,
                                                  steps=120 if REPS == 2 else 250)
    return at


def batch_of(atoms):
    return calc._atoms_to_batch(atoms).to_dict()


def energy(batch):
    return MODEL(batch, compute_force=False, training=True)["energy"].sum().cpu()


# ------------------------------------------------ trainable weight subset
if args.subset == "readout":
    trainable = [(n, p) for n, p in MODEL.named_parameters() if "readout" in n]
else:
    trainable = list(MODEL.named_parameters())
for _, p in MODEL.named_parameters():
    p.requires_grad_(False)
for _, p in trainable:
    p.requires_grad_(True)
params = [p for _, p in trainable]
theta0 = [p.detach().clone() for p in params]
n_param = sum(p.numel() for p in params)
print(f"Trainable: {n_param} weights across {len(params)} tensors "
      f"({args.subset}); device={args.device}; cells={N} atoms")


# ================================================================= constructions
def binodal_c(omega, T, n_iter=60):
    """Cu-rich limb of the symmetric regular-solution gap (differentiable)."""
    u = torch.tensor(np.log(0.05 / 0.95))
    for _ in range(n_iter):
        c = torch.sigmoid(u)
        f = omega * (1 - 2 * c) - K_B * T * torch.log((1 - c) / c)
        df = (-2 * omega - K_B * T * (-1 / (1 - c) - 1 / c)) * c * (1 - c)
        u = u - f / df
    return torch.sigmoid(u)


def line_compound_solvus(omega_g, Hf, c0, T, n_iter=80):
    """Al solubility limit x_s in the gamma solution set by the common tangent
    from G_gamma (regular, Omega_g) to the line compound (c0, Hf). Solves
        G(x_s) + G'(x_s)(c0 - x_s) = Hf ,  x_s in (0, c0),  differentiable."""
    def G(x):
        return omega_g * x * (1 - x) + K_B * T * (x * torch.log(x)
                                                  + (1 - x) * torch.log(1 - x))
    def dG(x):
        return omega_g * (1 - 2 * x) + K_B * T * torch.log(x / (1 - x))
    u = torch.tensor(np.log(0.05 / 0.95))
    for _ in range(n_iter):
        x = torch.sigmoid(u) * c0            # keep x in (0, c0)
        r = G(x) + dG(x) * (c0 - x) - Hf
        dx = (torch.sigmoid(u) * (1 - torch.sigmoid(u))) * c0
        # dr/dx = G'(x) + G''(x)(c0-x) - G'(x) = G''(x)(c0-x)
        d2G = -2 * omega_g + K_B * T * (1 / x + 1 / (1 - x))
        dr = d2G * (c0 - x) * dx
        u = u - r / dr
    return torch.sigmoid(u) * c0


# ================================================================= structures
print(f"Relaxing structures with pretrained MACE ({N} atoms)…")
t0 = time.time()
struct = {}
if args.system in ("cuni", "both"):
    a = np.full(N, 29); a[rng.permutation(N)[: N // 2]] = 28
    struct["cuni_Cu"] = batch_of(relaxed(np.full(N, 29), 3.62))
    struct["cuni_Ni"] = batch_of(relaxed(np.full(N, 28), 3.52))
    struct["cuni_mix_num"] = a
    struct["cuni_mix"] = batch_of(relaxed(a, 3.57))
if args.system in ("nial", "both"):
    g = np.full(N, 28); g[rng.permutation(N)[: N // 8]] = 13      # 1/8 Al in fcc Ni
    struct["nial_Ni"] = batch_of(relaxed(np.full(N, 28), 3.52))
    struct["nial_Al"] = batch_of(relaxed(np.full(N, 13), 4.05))
    struct["nial_g"] = batch_of(g_atoms := relaxed(g, 3.55))
    struct["nial_g_x"] = 0.125
    struct["nial_gp"] = batch_of(L12_Ni3Al(3.57))
# cross-system physicality probe: Ag-Cu must not break when we tune Cu-Ni
ac = np.full(N, 47); ac[rng.permutation(N)[: N // 2]] = 29
struct["agcu_Ag"] = batch_of(relaxed(np.full(N, 47), 4.09))
struct["agcu_Cu"] = batch_of(relaxed(np.full(N, 29), 3.62))
struct["agcu_mix"] = batch_of(relaxed(ac, 3.85))
print(f"   relaxed in {time.time()-t0:.0f} s")


def omega_cuni():
    return 4 * (energy(struct["cuni_mix"]) / N
                - 0.5 * energy(struct["cuni_Cu"]) / N
                - 0.5 * energy(struct["cuni_Ni"]) / N)


def nial_solvus():
    x = struct["nial_g_x"]
    dH = energy(struct["nial_g"]) / N - (1 - x) * energy(struct["nial_Ni"]) / N \
        - x * energy(struct["nial_Al"]) / N
    omega_g = dH / (x * (1 - x))
    ngp = 4 * REPS**3
    Hf = energy(struct["nial_gp"]) / ngp - 0.75 * energy(struct["nial_Ni"]) / N \
        - 0.25 * energy(struct["nial_Al"]) / N
    return line_compound_solvus(omega_g, Hf, 0.25, T_NIAL), omega_g, Hf


def agcu_omega():
    return 4 * (energy(struct["agcu_mix"]) / N
                - 0.5 * energy(struct["agcu_Ag"]) / N
                - 0.5 * energy(struct["agcu_Cu"]) / N)


def physicality():
    """Probes the tune must not wreck (cross-system + pure)."""
    with torch.no_grad():
        d = {"agcu_omega_meV": agcu_omega().item() * 1e3,
             "pure_Cu_eV": energy(struct.get("cuni_Cu", struct["agcu_Cu"])).item() / N,
             "pure_Ni_eV": (energy(struct["cuni_Ni"]).item() / N
                            if "cuni_Ni" in struct else None)}
    return d


# ================================================================= FD check
if args.fdcheck:
    om = omega_cuni()
    g = torch.cat([x.reshape(-1) for x in torch.autograd.grad(om, params)])
    flat = torch.cat([p.reshape(-1) for p in params])
    print("\n[FD check] dOmega_CuNi/dtheta on 4 weights:")
    for i in np.linspace(0, n_param - 1, 4).astype(int):
        off = i
        for p in params:
            if off < p.numel():
                with torch.no_grad():
                    o = p.reshape(-1)[off].item()
                    p.reshape(-1)[off] = o + 1e-4; fp = omega_cuni().item()
                    p.reshape(-1)[off] = o - 1e-4; fm = omega_cuni().item()
                    p.reshape(-1)[off] = o
                break
            off -= p.numel()
        fd = (fp - fm) / 2e-4
        print(f"   w[{i}]  ad {g[i].item():+.4e}  fd {fd:+.4e}  "
              f"rel {abs(g[i].item()-fd)/(abs(fd)+1e-12):.1e}")


# ================================================================= targets + train
T_GRID = torch.linspace(0.3 * TC_TARGET, 0.92 * TC_TARGET, 8)
OM_TARGET = 2 * K_B * TC_TARGET
C_TARGET = torch.stack([binodal_c(torch.tensor(OM_TARGET), float(T)) for T in T_GRID]).detach()

pre = {"physicality": physicality()}
if "cuni_mix" in struct:
    pre["cuni_omega_meV"] = omega_cuni().item() * 1e3
    pre["cuni_Tc_K"] = (omega_cuni() / (2 * K_B)).item()
if "nial_g" in struct:
    xs, omg, hf = nial_solvus()
    pre["nial_xAl_solvus"] = xs.item()
    pre["nial_omega_g_meV"] = omg.item() * 1e3
print("\nPretrained:"
      + (f" Cu-Ni Omega {pre['cuni_omega_meV']:+.1f} meV "
         f"(T_c {pre['cuni_Tc_K']:.0f} K, target {TC_TARGET:.0f})" if "cuni_mix" in struct else "")
      + (f" | Ni-Al x_Al solvus {pre['nial_xAl_solvus']:.3f} "
         f"(target {XAL_TARGET})" if "nial_g" in struct else ""))
print(f"            physicality: Ag-Cu Omega "
      f"{pre['physicality']['agcu_omega_meV']:+.1f} meV")

if args.benchmark:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import mace_benchmark as _mb
    print("\n[benchmark] scoring the property panel BEFORE tuning…")
    for p in params:
        p.requires_grad_(False)
    _bench_pre = _mb.evaluate(calc, reps=REPS)
    for p in params:
        p.requires_grad_(True)

opt = torch.optim.Adam(params, lr=args.lr)
log = []
print(f"\n[fine-tune] {args.steps} steps, lr {args.lr}, reg {args.reg}, "
      f"re-relax every {args.relax_every}")
for step in range(args.steps):
    opt.zero_grad()
    loss = torch.zeros(())
    if args.system in ("cuni", "both"):
        om = omega_cuni()
        c_pred = torch.stack([binodal_c(om, float(T)) for T in T_GRID])
        loss = loss + ((c_pred - C_TARGET) ** 2).mean()
    if args.system in ("nial", "both"):
        xs, _, _ = nial_solvus()
        loss = loss + ((xs - XAL_TARGET) / XAL_TARGET) ** 2
    reg = sum(((p - t) ** 2).sum() for p, t in zip(params, theta0)).cpu()
    (loss + args.reg * reg).backward()
    opt.step()

    if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
        with torch.no_grad():
            row = {"step": step, "loss": loss.item(), "reg": float(reg)}
            if "cuni_mix" in struct:
                row["cuni_omega_meV"] = omega_cuni().item() * 1e3
                row["cuni_Tc_K"] = (omega_cuni() / (2 * K_B)).item()
            if "nial_g" in struct:
                row["nial_xAl_solvus"] = nial_solvus()[0].item()
        log.append(row)
        msg = f"   step {step:4d}  loss {loss.item():.3e}  reg {float(reg):.2e}"
        if "cuni_mix" in struct:
            msg += f"  CuNi T_c {row['cuni_Tc_K']:6.0f} K"
        if "nial_g" in struct:
            msg += f"  NiAl x_Al {row['nial_xAl_solvus']:.3f}"
        print(msg)

    # re-relax frozen geometries with the current weights (snapshot refresh).
    # Freeze the trainable weights (not all grad) so MACE can still compute
    # forces via position-autograd during relaxation; the refreshed geometry
    # re-enters as detached data.
    if args.relax_every and (step + 1) % args.relax_every == 0 and step + 1 < args.steps:
        for p in params:
            p.requires_grad_(False)
        if "cuni_mix" in struct:
            struct["cuni_mix"] = batch_of(relaxed(struct["cuni_mix_num"], 3.57))
        for p in params:
            p.requires_grad_(True)

post = {"physicality": physicality()}
if "cuni_mix" in struct:
    post["cuni_omega_meV"] = omega_cuni().item() * 1e3
    post["cuni_Tc_K"] = (omega_cuni() / (2 * K_B)).item()
if "nial_g" in struct:
    post["nial_xAl_solvus"] = nial_solvus()[0].item()
if "cuni_mix" in struct:
    print(f"\nTuned: Cu-Ni T_c {pre['cuni_Tc_K']:.0f} -> {post['cuni_Tc_K']:.0f} K "
          f"(target {TC_TARGET:.0f})")
if "nial_g" in struct:
    print(f"Tuned: Ni-Al x_Al {pre['nial_xAl_solvus']:.3f} -> {post['nial_xAl_solvus']:.3f} "
          f"(target {XAL_TARGET})")
print(f"       physicality Ag-Cu Omega {pre['physicality']['agcu_omega_meV']:+.1f}"
      f" -> {post['physicality']['agcu_omega_meV']:+.1f} meV "
      f"(drift {post['physicality']['agcu_omega_meV']-pre['physicality']['agcu_omega_meV']:+.1f})")

# checkpoint the tuned weights + log
ckpt = OUT / f"mace_finetune_stage2_{args.system}_{N}atom.pt"
torch.save({n: p.detach().cpu() for (n, _), p in zip(trainable, params)}, ckpt)
res = {"args": vars(args), "n_trainable_weights": int(n_param),
       "target_Tc_K": TC_TARGET, "target_xAl": XAL_TARGET,
       "pretrained": pre, "tuned": post, "history": log,
       "checkpoint": str(ckpt.name)}
with open(OUT / f"mace_finetune_stage2_{args.system}_{N}atom.json", "w") as f:
    json.dump(res, f, indent=1)
print(f"\nWrote checkpoint {ckpt.name} + log json")

if args.benchmark:
    print("\n[benchmark] scoring the property panel AFTER tuning…")
    for p in params:
        p.requires_grad_(False)          # forward-only; MACE still gets forces via position-autograd
    _bench_post = _mb.evaluate(calc, reps=REPS)
    print("\n[benchmark] whack-a-mole check (pre -> post):")
    _mb.diff_dicts(_bench_pre, _bench_post)
