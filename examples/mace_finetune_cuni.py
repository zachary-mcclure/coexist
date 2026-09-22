"""
Train a potential against a phase diagram (stage 1: mechanism + FD check).

This is the reverse-mode showcase: forward simulation gives you properties,
but reverse-mode lets you differentiate a phase-boundary loss with respect to
*all the parameters of the potential itself* and step them to match a target
diagram. For MACE-MP-0's weights (thousands-to-millions of parameters against
a handful of boundary observations) reverse-mode is not a convenience — it is
the only affordable direction. Forward-mode would cost one engine-backward
pass per parameter; reverse-mode costs one, independent of parameter count.

Target system: Cu-Ni. MACE-MP-0 collapses the solid-state miscibility gap —
consolute T_c = Omega/2k_B = 282 K vs. the assessed 600-650 K (the largest
engine-fidelity error in the paper). It is a *solid* miscibility gap, so the
whole chain is static energetics -> Omega/binodal -> loss and stays fully
differentiable: no gradient has to cross an MD trajectory.

Chain (one torch autograd graph):
    theta (MACE weights)
      -> E(fixed relaxed Cu, Ni, Cu50Ni50 cells)     [MACE forward]
      -> dH_mix -> Omega                              [algebra]
      -> consolute T_c and solvus c(T) via a Newton binodal   [construction]
      -> L = (phase-diagram target - prediction)^2
      -> dL/dtheta                                    [ONE backward pass]

Stage 1 deliverable (this script, laptop): prove dL/dtheta is real by matching
it to a central finite difference on individual weights, both for Omega (pure
energetics) and for a solvus loss (reverse *through the binodal solver*), then
show a few optimizer steps move Omega toward the target. Stage 2 (a GPU pod)
runs the real fine-tune to convergence with a stay-physical regularizer and
checks the tuned potential did not break elsewhere.

Geometry is frozen at the pretrained relaxation (snapshot-bridge scope, as
everywhere in the paper: gradients are exact at the relaxed geometry and omit
relaxation response). Small 32-atom cells by default for a fast FD check
(--reps 2); the real run uses the converged 108-atom cells (--reps 3).

Run:  python examples/mace_finetune_cuni.py [--reps 2] [--steps 0]
"""
import argparse
import warnings
warnings.filterwarnings("ignore")
import json
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

K_B = 8.617333e-5           # eV/K
TC_TARGET = 625.0           # K, midpoint of the assessed Cu-Ni 600-650 band

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=2,
                help="cubic-cell repeats/axis: 2->32 atoms (fast FD check), "
                     "3->108 atoms (converged size, the real run)")
ap.add_argument("--steps", type=int, default=25,
                help="gradient-descent steps in the proof-of-concept fine-tune")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
REPS = args.reps
N = 4 * REPS**3

# ---------------------------------------------------------------- MACE engine
calc = mace_mp(model="small", device="cpu", default_dtype="float64")
MODEL = calc.models[0]


def relaxed(numbers, a0):
    """Volume+position relax a cubic fcc cell with pretrained MACE, freeze it."""
    at = bulk("Cu", "fcc", a=a0, cubic=True).repeat((REPS,) * 3)
    at.set_atomic_numbers(numbers)
    at.calc = calc
    FIRE(FrechetCellFilter(at), logfile=None).run(fmax=0.05,
                                                  steps=120 if REPS == 2 else 250)
    return at


def batch_of(atoms):
    return calc._atoms_to_batch(atoms).to_dict()


def energy(batch):
    """Total energy (eV) of a fixed cell as a differentiable function of the
    MACE weights (geometry frozen in `batch`)."""
    return MODEL(batch, compute_force=False, training=True)["energy"].sum()


# ---- frozen relaxed Cu-Ni structures: pure Cu, pure Ni, random 50/50 --------
rng = np.random.default_rng(args.seed)
num_alloy = np.full(N, 29)                    # Cu
num_alloy[rng.permutation(N)[: N // 2]] = 28  # half -> Ni
print(f"Relaxing Cu-Ni cells with pretrained MACE ({N} atoms)…")
B_Cu = batch_of(relaxed(np.full(N, 29), 3.62))
B_Ni = batch_of(relaxed(np.full(N, 28), 3.52))
B_mix = batch_of(relaxed(num_alloy, 3.57))


def omega_of_weights():
    """Omega(theta) = 4 * dH_mix(x=0.5), per atom (eV). Differentiable in theta."""
    e_cu = energy(B_Cu) / N
    e_ni = energy(B_Ni) / N
    e_mix = energy(B_mix) / N
    dH = e_mix - 0.5 * e_cu - 0.5 * e_ni
    return 4.0 * dH


# ---------------------------------------------------- construction (in torch)
def binodal_c(omega, T, n_iter=60):
    """Cu-rich limb of the symmetric regular-solution miscibility gap at T.

    Solves the common-tangent (here horizontal-tangent) condition
        Omega*(1-2c) = k_B*T*ln((1-c)/c)
    in logit space u=ln(c/(1-c)) by damped Newton, seeded on the dilute branch.
    Differentiable in Omega (implicit-function derivative through the solver),
    exactly the JAX common_tangent(G,G,T) the paper uses, kept in one graph.
    """
    u = torch.tensor(np.log(0.05 / 0.95))     # seed c~0.05 (Cu-rich branch)
    for _ in range(n_iter):
        c = torch.sigmoid(u)
        f = omega * (1 - 2 * c) - K_B * T * torch.log((1 - c) / c)
        # df/du via autograd-free chain: dc/du = c(1-c)
        dc = c * (1 - c)
        df = (-2 * omega - K_B * T * (-1 / (1 - c) - 1 / c)) * dc
        u = u - f / df
    return torch.sigmoid(u)


# ------------------------------------------------- trainable weight subset
trainable = [(n, p) for n, p in MODEL.named_parameters() if "readout" in n]
if not trainable:                              # fallback: last two tensors
    trainable = list(MODEL.named_parameters())[-2:]
for _, p in MODEL.named_parameters():
    p.requires_grad_(False)
for _, p in trainable:
    p.requires_grad_(True)
params = [p for _, p in trainable]
n_param = sum(p.numel() for p in params)
theta0 = [p.detach().clone() for p in params]
print(f"Trainable subset: {len(params)} tensors, {n_param} scalar weights "
      f"({', '.join(n for n, _ in trainable)})")

# ------------------------------------------------------------ baseline
om0 = omega_of_weights()
Tc0 = (om0 / (2 * K_B)).item()
print(f"\nPretrained MACE Cu-Ni: Omega = {om0.item()*1e3:+.1f} meV  ->  "
      f"T_c = {Tc0:.0f} K   (target {TC_TARGET:.0f} K; assessed 600-650)")
T_PROBE = 0.6 * Tc0          # safely inside the pretrained gap, any cell size
c_dilute0 = binodal_c(om0.detach(), T_PROBE).item()
print(f"                       solvus at {T_PROBE:.0f} K (0.6 T_c): "
      f"Cu-rich limb c_Ni = {c_dilute0:.4f}")

# ============================================================================
# FD CHECK 1 — dOmega/dtheta (pure energetics, the MACE weight gradient)
# ============================================================================
om = omega_of_weights()
g_om = torch.autograd.grad(om, params, retain_graph=False)
flat_g = torch.cat([g.reshape(-1) for g in g_om])
# probe a handful of individual weights by central finite difference
probe = np.linspace(0, n_param - 1, 6).astype(int)
flat_params = torch.cat([p.reshape(-1) for p in params])
print("\n[FD check 1]  dOmega/dtheta_i : autograd vs central finite difference")
h = 1e-4
rows1 = []
for i in probe:
    # locate (tensor, offset) for flat index i
    off = i
    for p in params:
        if off < p.numel():
            with torch.no_grad():
                orig = p.reshape(-1)[off].item()
                p.reshape(-1)[off] = orig + h
                fp = omega_of_weights().item()
                p.reshape(-1)[off] = orig - h
                fm = omega_of_weights().item()
                p.reshape(-1)[off] = orig
            break
        off -= p.numel()
    fd = (fp - fm) / (2 * h)
    ad = flat_g[i].item()
    rel = abs(ad - fd) / (abs(fd) + 1e-12)
    rows1.append(rel)
    print(f"   w[{i:>6d}]  autograd {ad:+.6e}   FD {fd:+.6e}   rel {rel:.1e}")
print(f"   max rel err over probes: {max(rows1):.1e}  "
      f"({'PASS' if max(rows1) < 1e-4 else 'CHECK'})")

# ============================================================================
# FD CHECK 2 — dL/dtheta THROUGH the binodal solver (reverse through the
# construction): L = (c_solvus(Omega(theta), 400 K) - c_target)^2
# ============================================================================
C_TARGET = 0.08          # a target Cu-rich solvus composition at T_PROBE
def solvus_loss():
    om = omega_of_weights()
    c = binodal_c(om, T_PROBE)
    return (c - C_TARGET) ** 2

L = solvus_loss()
g_L = torch.cat([g.reshape(-1) for g in torch.autograd.grad(L, params)])
print("\n[FD check 2]  dL/dtheta_i THROUGH the binodal Newton solver")
rows2 = []
for i in probe:
    off = i
    for p in params:
        if off < p.numel():
            with torch.no_grad():
                orig = p.reshape(-1)[off].item()
                p.reshape(-1)[off] = orig + h
                fp = solvus_loss().item()
                p.reshape(-1)[off] = orig - h
                fm = solvus_loss().item()
                p.reshape(-1)[off] = orig
            break
        off -= p.numel()
    fd = (fp - fm) / (2 * h)
    ad = g_L[i].item()
    rel = abs(ad - fd) / (abs(fd) + 1e-12)
    rows2.append(rel)
    print(f"   w[{i:>6d}]  autograd {ad:+.6e}   FD {fd:+.6e}   rel {rel:.1e}")
print(f"   max rel err over probes: {max(rows2):.1e}  "
      f"({'PASS' if max(rows2) < 1e-3 else 'CHECK'})")

# ============================================================================
# PROOF-OF-CONCEPT FINE-TUNE — a few reverse-mode steps that move the
# consolute toward the target, with a stay-close-to-pretrained regularizer.
# ============================================================================
history = []
if args.steps > 0:
    opt = torch.optim.Adam(params, lr=1e-3)
    LAMBDA_REG = 1.0
    print(f"\n[fine-tune] {args.steps} Adam steps on (T_c-{TC_TARGET:.0f})^2 "
          f"+ {LAMBDA_REG}||theta-theta0||^2")
    for step in range(args.steps):
        opt.zero_grad()
        om = omega_of_weights()
        Tc = om / (2 * K_B)
        reg = sum(((p - t) ** 2).sum() for p, t in zip(params, theta0))
        loss = ((Tc - TC_TARGET) / TC_TARGET) ** 2 + LAMBDA_REG * reg
        loss.backward()
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"   step {step:3d}: Omega {om.item()*1e3:+7.1f} meV  "
                  f"T_c {Tc.item():6.0f} K  reg {float(reg):.2e}")
            history.append({"step": step, "omega_meV": om.item() * 1e3,
                            "Tc_K": Tc.item(), "reg": float(reg)})
    om_f = omega_of_weights()
    print(f"   final: Omega {om_f.item()*1e3:+.1f} meV -> T_c "
          f"{(om_f/(2*K_B)).item():.0f} K "
          f"(moved {(om_f-om0).item()*1e3:+.1f} meV toward target)")

res = {
    "system": f"Cu-Ni, {N} atoms, seed {args.seed}",
    "n_trainable_weights": int(n_param),
    "trainable_tensors": [n for n, _ in trainable],
    "pretrained_omega_meV": om0.item() * 1e3,
    "pretrained_Tc_K": Tc0,
    "target_Tc_K": TC_TARGET,
    "fd_check1_omega_max_relerr": float(max(rows1)),
    "fd_check2_solvus_through_solver_max_relerr": float(max(rows2)),
    "finetune_history": history,
}
with open(OUT / f"mace_finetune_cuni_{N}atom.json", "w") as f:
    json.dump(res, f, indent=1)
print(f"\nWrote {OUT/('mace_finetune_cuni_%datom.json' % N)}")
