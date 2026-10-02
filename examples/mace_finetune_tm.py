"""
Differentiable LATENT-HEAT melting-point PROXY tune for pure Cu.

Implements the path the recon (examples/melting_point_scope.py) identified as the
only genuinely differentiable, frozen-snapshot-scope route to a melting point:

        T_m_pred = (U_l - U_s) / dS_fus          [latent heat / fusion entropy, P~0]

  U_s(T): solid internal energy from a MACE E(V) scan -> torch Debye-Grüneisen
          chain (eos_fit -> theta_D -> Debye U_vib).  This is a FULL internal
          energy (E0 + zero-point + 3kT·D3), differentiable in the weights.

  U_l(T): liquid internal energy = frozen-snapshot ensemble mean of MACE
          POTENTIAL energies from a short Langevin MD melt, PLUS the classical
          kinetic 3/2·kT per atom.  The +3/2·kT is the CORRECTNESS FIX flagged
          in the recon: MACE snapshot energy is potential-only, but U_s is a
          full internal energy, so without it dH_fus comes out ~0.19 eV/atom too
          low (even negative).  The kinetic term cancels in dH between the phases
          but must be present on BOTH sides, not one.

  dS_fus: HELD CONSTANT at the experimental Cu value 1.134 k_B/atom.  With dS
          fixed, T_m(theta) = dH_fus(theta)/dS_fus is a torch scalar in the
          weights -> dT_m/dtheta by autograd.  (The liquid entropy itself is a
          phase-space-volume property that frozen snapshots do NOT carry; see
          the recon's LIQUID note.  That is the standing caveat.)

BOTH U_s and U_l are computed from FROZEN configurations (the E(V) scan configs
and the MD snapshots are captured once, then re-evaluated with the current
weights each step), exactly like examples/mace_finetune_alsi_liquid.py.  So the
tune can move either phase's energy; the sampling derivative (how the ensemble
would shift) is out of scope, as everywhere else in the paper.

Pipeline:
  1. SOLID: relax fcc Cu, E(V) scan -> store frozen batches.
  2. LIQUID: Langevin melt -> equilibrate at T_ref -> store frozen snapshots.
  3. Compute pretrained T_m_pred (the KEY number) vs exp 1358 K.
  4. PLUMBING CHECK: T_m_pred.backward(); confirm ||dT_m/dtheta|| finite & nonzero.
  5. (optional) TUNE: Adam on loss = ((T_m-1358)/1358)^2 + reg*||theta-theta0||^2
     (+ optional Ag-Cu frozen-anchor), with the mace_benchmark panel as a
     before/after whack-a-mole guard.

Run (BACKGROUND, CPU is slow; keep the cell small and the melt short):
  cd /Users/zmcclure/Research/coexist && PYTHONPATH=/Users/zmcclure/Research/coexist \
    /Users/zmcclure/SimEngine/.venv/bin/python examples/mace_finetune_tm.py \
      --reps 2 --md-steps 1200 --snapshots 6 --t-liq 1500 --steps 0        # plumbing only
  ... add --steps 40 --reg 0.1 --benchmark   to actually tune.
Output: examples/output/mace_finetune_tm.json  (+ checkpoint if a tune ran).
"""
import argparse, json, warnings, time, sys
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
ap.add_argument("--reps", type=int, default=2)             # 2 -> 32-atom fcc
ap.add_argument("--md-steps", type=int, default=1200)      # equilibration steps
ap.add_argument("--melt-temp", type=float, default=2400.0) # melt T (>> T_m)
ap.add_argument("--t-liq", type=float, default=1500.0)     # liquid sampling / reference T
ap.add_argument("--md-dt", type=float, default=2.0)        # fs
ap.add_argument("--snapshots", type=int, default=6)
ap.add_argument("--snap-every", type=int, default=50)
ap.add_argument("--expand", type=float, default=1.05)      # liquid density expansion
ap.add_argument("--subset", choices=["readout", "all"], default="all")
ap.add_argument("--steps", type=int, default=0)            # tuning steps (0 = plumbing only)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--reg", type=float, default=0.1)          # stay-near-pretrained
ap.add_argument("--target-tm", type=float, default=1358.0) # exp Cu T_m (K)
ap.add_argument("--agcu-anchor", type=float, default=0.0,  # weight of the Ag-Cu frozen anchor
                help="if >0, penalize drift of a frozen Ag50Cu50 mixing energy")
ap.add_argument("--benchmark", action="store_true")
ap.add_argument("--device", default="cpu")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

K_B = 8.617333e-5           # eV/K
M_CU = 63.546               # amu
EXP = dict(T_m=1358.0, dH_fus=0.137, dS_fus_kB=1.134, theta_D=343.0, B0_GPa=140.0,
           a_fcc=3.615)
_A3_TO_BOHR3 = 6.74833
_EVA3_TO_KBAR = 1602.176

torch.manual_seed(args.seed); np.random.seed(args.seed)
rng = np.random.default_rng(args.seed)
t0 = time.time()
print(f"Loading MACE-MP-0 (small) on {args.device}…")
calc = mace_mp(model="small", device=args.device, default_dtype="float64")
MODEL = calc.models[0]
# MACE-MP weights load with requires_grad=False (inference calculator). Enable it
# NOW, BEFORE any energy forward, so the autograd graph for U_s/U_l connects to the
# weights; enabling it after the forward (as the recon prototype did) yields zero grad.
for p in MODEL.parameters():
    p.requires_grad_(True)

# Gauss-Legendre nodes for the Debye D3 integral (torch, differentiable).
_glx, _glw = np.polynomial.legendre.leggauss(48)
_GLX = torch.tensor(_glx); _GLW = torch.tensor(_glw)


def to_batch(atoms):
    """Freeze an ASE config into a MACE batch dict (positions etc. are fixed)."""
    return calc._atoms_to_batch(atoms.copy()).to_dict()


def E_of(batch):
    """MACE total energy of a frozen batch as a torch scalar (grad -> weights)."""
    return MODEL(batch, compute_force=False, training=True)["energy"].sum()


# ── torch Debye-Grüneisen (mirrors coexist/core/thermo_vib.py) ───────────────
def torch_eos_fit(volumes, energies):
    """Cubic E(V) fit; differentiable in `energies`. Returns V0,E0,B0,Bprime."""
    V = torch.as_tensor(volumes, dtype=torch.float64)
    E = torch.stack(list(energies)) if isinstance(energies, (list, tuple)) else energies
    Vm = V.mean()
    v = V / Vm - 1.0
    Mmat = torch.stack([torch.ones_like(v), v, v**2, v**3], dim=1)
    coef = torch.linalg.solve(Mmat.T @ Mmat, Mmat.T @ E)
    a, b, c, d = coef
    disc = torch.sqrt(torch.clamp(c**2 - 3.0 * d * b, min=0.0))
    v1 = (-c + disc) / (3.0 * d); v2 = (-c - disc) / (3.0 * d)
    curv1 = 2.0 * c + 6.0 * d * v1
    v0 = torch.where(curv1 > 0.0, v1, v2)
    E0 = a + b * v0 + c * v0**2 + d * v0**3
    Epp = (2.0 * c + 6.0 * d * v0) / Vm**2
    Eppp = 6.0 * d / Vm**3
    V0 = Vm * (1.0 + v0)
    B0 = V0 * Epp
    Bprime = -1.0 - V0 * Eppp / Epp
    return V0, E0, B0, Bprime


def torch_debye_temperature(V0_pa, B0, M_amu, scale=1.0):
    r0 = (3.0 * V0_pa * _A3_TO_BOHR3 / (4.0 * np.pi)) ** (1.0 / 3.0)
    B_kbar = B0 * _EVA3_TO_KBAR
    return scale * 41.63 * torch.sqrt(r0 * B_kbar / M_amu)


def torch_D3(x):
    t = 0.5 * x * (_GLX + 1.0)
    integrand = t**3 / torch.expm1(t)
    return 3.0 * (0.5 * x * (_GLW * integrand).sum()) / x**3


def debye_U_vib(T, theta_D):
    """Per-atom Debye internal energy [eV] (zero-point + thermal)."""
    x = theta_D / T
    return (9.0 / 8.0) * K_B * theta_D + 3.0 * K_B * T * torch_D3(x)


# ── SOLID: relax fcc Cu once, build a FROZEN E(V) scan ───────────────────────
print("\n[SOLID] relaxing fcc Cu and freezing E(V) scan…")
cu = bulk("Cu", "fcc", cubic=True).repeat((args.reps,) * 3)
cu.calc = calc
FIRE(FrechetCellFilter(cu), logfile=None).run(fmax=0.02, steps=200)
n_s = len(cu)
a_relaxed = cu.cell.cellpar()[0] / args.reps
cell0 = cu.get_cell().copy()
scales = np.linspace(0.96, 1.04, 9)
vols_s, solid_batches = [], []
for s in scales:
    c = cu.copy(); c.set_cell(cell0 * s, scale_atoms=True)
    vols_s.append(c.get_volume())
    solid_batches.append(to_batch(c))
VOLS_S = torch.tensor(vols_s)
print(f"   relaxed a = {a_relaxed:.4f} Å (exp {EXP['a_fcc']}),  {n_s} atoms, "
      f"{len(solid_batches)} E(V) points frozen")


def U_solid(T_ref):
    """Full per-atom solid internal energy at T_ref, differentiable in weights."""
    en = [E_of(b) for b in solid_batches]
    V0, E0, B0, Bp = torch_eos_fit(VOLS_S, en)
    # B0 = V0_total * d2E_total/dV_total^2 is already an INTENSIVE bulk modulus
    # (pressure units); do NOT divide by n_s. V0 does need /n_s (per-atom volume
    # feeds the Wigner-Seitz radius in the Debye formula). The recon's B0/n_s was
    # a bug (gave B0~5 GPa, theta_D~59 K instead of ~140 GPa / ~340 K).
    V0_pa, E0_pa = V0 / n_s, E0 / n_s
    theta_D = torch_debye_temperature(V0_pa, B0, M_CU)
    return E0_pa + debye_U_vib(T_ref, theta_D), theta_D, V0_pa, B0, Bp


# ── LIQUID: melt Cu, equilibrate at T_ref, FREEZE snapshots ──────────────────
print(f"\n[LIQUID] melting Cu at {args.melt_temp:.0f} K, sampling at {args.t_liq:.0f} K…")
liq = bulk("Cu", "fcc", cubic=True).repeat((args.reps,) * 3)
liq.set_cell(liq.get_cell() * args.expand, scale_atoms=True)   # liquid density
liq.calc = calc
n_l = len(liq)
MaxwellBoltzmannDistribution(liq, temperature_K=args.melt_temp,
                             rng=np.random.RandomState(args.seed))
dt = args.md_dt * units.fs
Langevin(liq, dt, temperature_K=args.melt_temp, friction=0.02,
         rng=rng).run(args.md_steps // 2)                      # melt hot
dyn = Langevin(liq, dt, temperature_K=args.t_liq, friction=0.02, rng=rng)
dyn.run(args.md_steps)                                         # equilibrate at T_ref
r0 = liq.get_positions().copy()
liq_batches = []
for _ in range(args.snapshots):
    dyn.run(args.snap_every)
    liq_batches.append(to_batch(liq))
msd = float(((liq.get_positions() - r0) ** 2).sum(axis=1).mean())
print(f"   froze {len(liq_batches)} snapshots, MSD(last)={msd:.1f} Å² "
      f"({'liquid' if msd > 2.0 else 'MAYBE CRYSTALLIZED — raise melt-temp/steps'})")


def U_liquid(T_ref):
    """Full per-atom liquid internal energy: <E_pot>/n + 3/2 kT (the recon fix)."""
    U_l_pot = torch.stack([E_of(b) for b in liq_batches]).mean() / n_l
    return U_l_pot + 1.5 * K_B * T_ref, U_l_pot


# ── optional Ag-Cu frozen mixing anchor (keeps the alloy control from drifting)
AGCU = None
if args.agcu_anchor > 0:
    print("\n[Ag-Cu anchor] building one frozen Ag50Cu50 + pure Ag/Cu mixing config…")
    def _relax_batch(atoms):
        a = atoms.copy(); a.calc = calc
        FIRE(FrechetCellFilter(a), logfile=None).run(fmax=0.05, steps=200)
        return to_batch(a), len(a)
    ag = bulk("Ag", "fcc", a=4.085, cubic=True).repeat((args.reps,) * 3)
    cu_a = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((args.reps,) * 3)
    mix = bulk("Ag", "fcc", a=3.85, cubic=True).repeat((args.reps,) * 3)
    nums = mix.get_atomic_numbers()
    nums[rng.permutation(len(mix))[:len(mix) // 2]] = bulk("Cu", "fcc").numbers[0]
    mix.set_atomic_numbers(nums)
    b_ag, n_ag = _relax_batch(ag); b_cu, n_cu = _relax_batch(cu_a)
    b_mix, n_mx = _relax_batch(mix)
    def omega_agcu():
        return 4.0 * (E_of(b_mix) / n_mx - 0.5 * E_of(b_ag) / n_ag
                      - 0.5 * E_of(b_cu) / n_cu)
    AGCU = (omega_agcu, float(omega_agcu().detach()))
    print(f"   pretrained Omega_AgCu(frozen) = {AGCU[1]*1e3:+.0f} meV")


# ── pretrained T_m_pred (THE KEY NUMBER) ─────────────────────────────────────
T_ref = args.t_liq
dS_fixed = EXP["dS_fus_kB"] * K_B           # held constant
print(f"\n[T_m PROXY]  T_m = (U_l - U_s) / dS_fus,  dS_fus = {EXP['dS_fus_kB']} k_B (fixed)")
U_s, theta_D, V0_pa, B0_pa, Bp = U_solid(T_ref)
U_l, U_l_pot = U_liquid(T_ref)
dH_fus = U_l - U_s
Tm = dH_fus / dS_fixed
print(f"   solid:  V0/atom={float(V0_pa):.3f} Å³  B0={float(B0_pa)*160.2176:.0f} GPa "
      f"(exp {EXP['B0_GPa']})  theta_D={float(theta_D):.0f} K (exp {EXP['theta_D']})")
print(f"   U_s({T_ref:.0f}K)={float(U_s):.4f}  U_l_pot={float(U_l_pot):.4f}  "
      f"U_l={float(U_l):.4f} eV/atom")
print(f"   dH_fus = {float(dH_fus)*1e3:+.1f} meV/atom (exp {EXP['dH_fus']*1e3:.0f})")
print(f"   >>> PRETRAINED Cu T_m (proxy) = {float(Tm):.0f} K   "
      f"({float(Tm)-EXP['T_m']:+.0f} K vs exp {EXP['T_m']:.0f}) <<<")
Tm_pretrained = float(Tm)


# ── PLUMBING CHECK: dT_m/dtheta finite & nonzero ─────────────────────────────
print("\n[PLUMBING] T_m_pred.backward() -> ||dT_m/dtheta||…")
for p in MODEL.parameters():
    p.requires_grad_(True)
MODEL.zero_grad(set_to_none=True)
Tm.backward()
grads = [p.grad for p in MODEL.parameters() if p.grad is not None]
gnorm = float(torch.sqrt(sum((g**2).sum() for g in grads))) if grads else 0.0
n_nonzero = sum(1 for g in grads if float(g.abs().max()) > 0)
finite = bool(np.isfinite(gnorm)) and gnorm > 0
print(f"   ||dT_m/dtheta|| = {gnorm:.4e} K/weight-unit over {n_nonzero} tensors "
      f"with nonzero grad")
print(f"   gradient plumbing {'OK — T_m IS differentiable in the weights' if finite else 'FAILED'}")

result = dict(
    args=vars(args), n_solid=n_s, n_liquid=n_l, a_relaxed=a_relaxed, T_ref=T_ref,
    exp=EXP,
    solid=dict(V0_per_atom=float(V0_pa), B0_GPa=float(B0_pa) * 160.2176,
               Bprime=float(Bp), theta_D_K=float(theta_D), U_s_eV=float(U_s)),
    liquid=dict(melt_temp=args.melt_temp, n_snapshots=len(liq_batches),
                msd_last_A2=msd, U_l_pot_eV=float(U_l_pot), U_l_eV=float(U_l)),
    dH_fus_eV=float(dH_fus), dS_fus_kB=EXP["dS_fus_kB"],
    T_m_pretrained_K=Tm_pretrained, T_m_err_vs_exp_K=Tm_pretrained - EXP["T_m"],
    plumbing=dict(grad_norm_dTm_dtheta=gnorm, n_tensors_nonzero_grad=n_nonzero,
                  finite_nonzero=finite),
)


# ── optional TUNE ────────────────────────────────────────────────────────────
if args.steps > 0 and finite:
    sys.path.insert(0, str(Path(__file__).parent))
    from tune_diag import theta0_norm, rel_move, move_stats

    if args.subset == "all":
        trainable = list(MODEL.named_parameters())
    else:
        trainable = [(nm, p) for nm, p in MODEL.named_parameters() if "readout" in nm]
    for _, p in MODEL.named_parameters(): p.requires_grad_(False)
    for _, p in trainable: p.requires_grad_(True)
    params = [p for _, p in trainable]
    theta0 = [p.detach().clone() for p in params]
    THETA0N = theta0_norm(theta0)
    print(f"\n[TUNE] {args.steps} steps, lr {args.lr}, reg {args.reg}, "
          f"subset={args.subset} ({sum(p.numel() for p in params)} weights), "
          f"target T_m {args.target_tm:.0f} K")

    bpre = None
    if args.benchmark:
        import mace_benchmark as mb
        print("[benchmark] BEFORE…")
        for p in params: p.requires_grad_(False)
        bpre = mb.evaluate(calc, reps=args.reps)
        for p in params: p.requires_grad_(True)

    opt = torch.optim.Adam(params, lr=args.lr)
    for step in range(args.steps):
        opt.zero_grad()
        U_s_i, _, _, _, _ = U_solid(T_ref)
        U_l_i, _ = U_liquid(T_ref)
        Tm_i = (U_l_i - U_s_i) / dS_fixed
        l_tm = ((Tm_i - args.target_tm) / args.target_tm) ** 2
        reg = sum(((p - t) ** 2).sum() for p, t in zip(params, theta0))
        loss = l_tm + args.reg * reg
        if AGCU is not None:
            loss = loss + args.agcu_anchor * (AGCU[0]() - AGCU[1]) ** 2
        loss.backward(); opt.step()
        if step % max(1, args.steps // 12) == 0 or step == args.steps - 1:
            print(f"   step {step:4d}  T_m {float(Tm_i):6.0f} K  "
                  f"dW {rel_move(params, theta0, THETA0N)*100:.3f}%  "
                  f"loss {float(l_tm):.4f}")

    with torch.no_grad():
        U_s_f, _, _, _, _ = U_solid(T_ref)
        U_l_f, _ = U_liquid(T_ref)
        Tm_tuned = float((U_l_f - U_s_f) / dS_fixed)
    print(f"\n[TUNE] T_m: {Tm_pretrained:.0f} -> {Tm_tuned:.0f} K  "
          f"(target {args.target_tm:.0f})")

    tag = f"tm_cu_{args.subset}_{n_s}_reg{args.reg}"
    ckpt = OUT / f"mace_finetune_{tag}.pt"
    torch.save({nm: p.detach().cpu() for (nm, _), p in zip(trainable, params)}, ckpt)
    result["tune"] = dict(T_m_tuned_K=Tm_tuned, target_tm=args.target_tm,
                          subset=args.subset, reg=args.reg, lr=args.lr,
                          steps=args.steps, checkpoint=ckpt.name)
    result["tune"].update(move_stats(params, theta0))

    if args.benchmark:
        print("\n[benchmark] AFTER…")
        for p in params: p.requires_grad_(False)
        bpost = mb.evaluate(calc, reps=args.reps)
        print("\n[benchmark] T_m tune whack-a-mole check:")
        result["tune"]["controls_ok"] = bool(mb.diff_dicts(bpre, bpost))
        result["tune"]["bench_pre"], result["tune"]["bench_post"] = bpre, bpost

result["runtime_s"] = time.time() - t0
json.dump(result, open(OUT / "mace_finetune_tm.json", "w"), indent=1)
print(f"\nWrote {OUT/'mace_finetune_tm.json'}   ({result['runtime_s']:.0f} s)")
