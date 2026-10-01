"""
RECON + PROTOTYPE: is a DIFFERENTIABLE melting point T_m a feasible tuning
target for MACE-MP-0 in our frozen-snapshot scope?  (pure Cu)

T_m is where G_solid(T) = G_liquid(T).  Equivalently, at the crossing

        T_m = ΔH_fus / ΔS_fus ,     ΔH_fus = U_liquid − U_solid  (at P≈0)

so the whole question reduces to two pieces:

  SOLID  — easy, and genuinely differentiable in the weights.
     An E(V) scan on fcc Cu → eos_fit → (V0,E0,B0,B′) → Debye θ_D, and the
     Debye internal energy  U_s(T) = E0 + (9/8)kθ_D + 3kT·D3(θ_D/T).
     (thermo_vib.py does exactly this; here mirrored in torch so the whole
     chain back-props to the MACE weights.)

  LIQUID — the crux.  U_liquid(T) is a frozen-snapshot ensemble mean of MACE
     energies → differentiable in the weights, EXACTLY like the Al-Si liquid
     tune.  But G_liquid = U − T·S_liquid needs the liquid ENTROPY, which is a
     phase-space-VOLUME property of the ensemble, not a function of any fixed
     configuration.  Frozen snapshots carry ⟨E⟩ but NOT S.  Getting S right
     needs thermodynamic integration (a λ-path or T-path with the sampling
     derivative), which is out of the frozen-snapshot scope.

So the prototype estimates MACE-MP-0's Cu T_m three ways, each making the
entropy assumption EXPLICIT, and the report states plainly what is and is not
differentiable:

  (A) fusion-anchor         ΔS_fus := experimental 1.134 k_B/atom  (fixed const)
  (B) Richards' rule        ΔS_fus := 1.0   k_B/atom               (fixed const)
  (C) Debye-liquid proxy    ΔS_fus from a Debye entropy of a frozen liquid
                            snapshot + a communal term  (fully MACE-internal,
                            but crude — shows the size of the entropy crux)

Only ΔH_fus = U_l − U_s carries the weights; in (A)/(B) ΔS is a constant, so
T_m(θ) = ΔH_fus(θ)/ΔS_fus IS differentiable — we verify dT_m/dθ by autograd.
In (C) the liquid entropy also moves with θ but through a questionable model.

Experiment: Cu T_m = 1358 K, ΔH_fus = 0.137 eV/atom, θ_D = 343 K, B0 = 140 GPa.

Run (BACKGROUND, CPU is slow):
  cd /Users/zmcclure/Research/coexist && PYTHONPATH=/Users/zmcclure/Research/coexist \
    /Users/zmcclure/SimEngine/.venv/bin/python examples/melting_point_scope.py \
    --reps 2 --md-steps 1500 --snapshots 8 --t-liq 1500
Output: examples/output/melting_point_scope.json
"""
import argparse, json, warnings, time
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
ap.add_argument("--md-steps", type=int, default=1500)      # equilibration steps
ap.add_argument("--melt-temp", type=float, default=2400.0) # melt T (>> T_m)
ap.add_argument("--t-liq", type=float, default=1500.0)     # liquid sampling T (ref for ΔH)
ap.add_argument("--md-dt", type=float, default=2.0)        # fs
ap.add_argument("--snapshots", type=int, default=8)
ap.add_argument("--snap-every", type=int, default=50)
ap.add_argument("--expand", type=float, default=1.05)      # liquid density expansion
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

# Gauss-Legendre nodes for the Debye D3 integral (torch, differentiable).
_glx, _glw = np.polynomial.legendre.leggauss(48)
_GLX = torch.tensor(_glx); _GLW = torch.tensor(_glw)


def E_batch(atoms):
    """MACE total energy of a config as a torch scalar (grad flows to weights)."""
    b = calc._atoms_to_batch(atoms.copy()).to_dict()
    return MODEL(b, compute_force=False, training=True)["energy"].sum()


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


def debye_S_vib(T, theta_D):
    """Per-atom Debye vibrational entropy [eV/K]."""
    x = theta_D / T
    return K_B * (4.0 * torch_D3(x) - 3.0 * torch.log(-torch.expm1(-x)))


# ── SOLID: relax fcc Cu, E(V) scan, Debye θ_D, U_s(T) ────────────────────────
print("\n[SOLID] relaxing fcc Cu and scanning E(V)…")
cu = bulk("Cu", "fcc", cubic=True).repeat((args.reps,) * 3)
cu.calc = calc
FIRE(FrechetCellFilter(cu), logfile=None).run(fmax=0.02, steps=200)
n_s = len(cu)
a_relaxed = cu.cell.cellpar()[0] / args.reps
V0_cell = cu.get_volume()
print(f"   relaxed a = {a_relaxed:.4f} Å (exp {EXP['a_fcc']}),  {n_s} atoms")

scales = np.linspace(0.96, 1.04, 9)
vols, en = [], []
cell0 = cu.get_cell().copy()
for s in scales:
    c = cu.copy()
    c.set_cell(cell0 * s, scale_atoms=True)
    vols.append(c.get_volume())
    en.append(E_batch(c))
vols_t = torch.tensor(vols)
V0, E0, B0, Bp = torch_eos_fit(vols_t, en)
V0_pa = V0 / n_s
E0_pa = E0 / n_s
theta_D = torch_debye_temperature(V0_pa, B0 / n_s, M_CU)
print(f"   E(V) fit: V0/atom={float(V0_pa):.3f} Å³  "
      f"B0={float(B0/n_s)*160.2176:.0f} GPa (exp {EXP['B0_GPa']})  "
      f"B'={float(Bp):.2f}")
print(f"   θ_D = {float(theta_D):.0f} K  (exp {EXP['theta_D']})")

T_ref = args.t_liq
U_s = E0_pa + debye_U_vib(T_ref, theta_D)          # per-atom solid internal E at T_ref
S_s = debye_S_vib(T_ref, theta_D)
print(f"   U_s({T_ref:.0f} K) = {float(U_s):.4f} eV/atom   "
      f"S_s = {float(S_s)/K_B:.2f} k_B/atom")


# ── LIQUID: melt Cu, equilibrate at T_liq, freeze snapshots → U_l(T) ─────────
print(f"\n[LIQUID] melting Cu at {args.melt_temp:.0f} K, sampling at {T_ref:.0f} K…")
liq = bulk("Cu", "fcc", cubic=True).repeat((args.reps,) * 3)
liq.set_cell(liq.get_cell() * args.expand, scale_atoms=True)   # liquid density
liq.calc = calc
n_l = len(liq)
MaxwellBoltzmannDistribution(liq, temperature_K=args.melt_temp, rng=np.random.RandomState(args.seed))
dt = args.md_dt * units.fs
# melt hot, then cool/equilibrate at the sampling temperature
Langevin(liq, dt, temperature_K=args.melt_temp, friction=0.02, rng=rng).run(args.md_steps // 2)
dyn = Langevin(liq, dt, temperature_K=T_ref, friction=0.02, rng=rng)
dyn.run(args.md_steps)
r0 = liq.get_positions().copy()
snaps, e_snap = [], []
for _ in range(args.snapshots):
    dyn.run(args.snap_every)
    snaps.append(liq.copy())
    e_snap.append(E_batch(liq))
# stayed-liquid check (MSD; a crystallized sample plateaus ~ sub-Å²)
msd = float(((liq.get_positions() - r0) ** 2).sum(axis=1).mean())
U_l_pot = torch.stack(e_snap).mean() / n_l          # MACE POTENTIAL energy/atom
# MACE snapshot energy is potential-only; the Debye solid U_s is a FULL internal
# energy (→3kT, i.e. ³⁄₂kT kinetic + ³⁄₂kT potential at high T).  Add the
# classical kinetic ³⁄₂kT to the liquid so both sides are full internal energies
# at the same T (the kinetic term is identical in both phases and cancels in ΔH,
# but it must be present on BOTH sides, not one).
U_l = U_l_pot + 1.5 * K_B * T_ref
print(f"   collected {len(snaps)} snapshots, MSD(last)={msd:.1f} Å² "
      f"({'liquid' if msd > 2.0 else 'MAYBE CRYSTALLIZED'})")
print(f"   U_l_pot = {float(U_l_pot):.4f}  +³⁄₂kT = {float(U_l):.4f} eV/atom")


# ── ΔH_fus and the three T_m estimates ───────────────────────────────────────
dH_fus = U_l - U_s                                  # per atom, P≈0 so H≈U
print(f"\n[ΔH_fus] U_l − U_s = {float(dH_fus)*1e3:+.1f} meV/atom "
      f"(exp {EXP['dH_fus']*1e3:.0f} meV)")

# (C) Debye-liquid proxy entropy: Debye S on a frozen liquid snapshot's E(V)
#     + a communal term k_B (liquids carry ~1 k_B free-volume/communal entropy
#     the harmonic solid lacks). Crude but fully MACE-internal & differentiable.
snap0 = snaps[len(snaps) // 2]
cellL = snap0.get_cell().copy()
vlq, elq = [], []
for s in np.linspace(0.97, 1.03, 7):
    c = snap0.copy(); c.set_cell(cellL * s, scale_atoms=True)
    vlq.append(c.get_volume()); elq.append(E_batch(c))
V0l, E0l, B0l, Bpl = torch_eos_fit(torch.tensor(vlq), elq)
theta_l = torch_debye_temperature((V0l / n_l).abs(), (B0l / n_l).abs(), M_CU)
COMMUNAL_kB = 1.0
S_l_proxy = debye_S_vib(T_ref, theta_l) + COMMUNAL_kB * K_B
dS_proxy = S_l_proxy - S_s
print(f"   Debye-liquid proxy: θ_l={float(theta_l):.0f} K  "
      f"S_l={float(S_l_proxy)/K_B:.2f} k_B  ΔS_proxy={float(dS_proxy)/K_B:.2f} k_B")

estimates = {}
for tag, dS_kB, how in [
        ("fusion_anchor_exp", EXP["dS_fus_kB"], "ΔS = experimental 1.134 k_B (fixed)"),
        ("richards_rule",     1.0,              "ΔS = Richards 1.0 k_B (fixed)"),
        ("debye_liquid_proxy", float(dS_proxy) / K_B, "ΔS = MACE Debye-liquid proxy")]:
    dS = dS_kB * K_B
    Tm = float(dH_fus) / dS if dS > 0 else float("nan")
    err = Tm - EXP["T_m"]
    estimates[tag] = dict(dS_fus_kB=dS_kB, T_m_K=Tm, err_vs_exp_K=err, note=how)
    print(f"   T_m [{tag:18s}] = {Tm:7.0f} K  ({err:+.0f} K vs exp)   [{how}]")


# ── differentiability proof: dT_m/dθ for the fusion-anchor path ──────────────
# T_m = (U_l − U_s)/ΔS_fus ; ΔS_fus const → T_m is a torch scalar in the weights.
print("\n[AUTOGRAD] dT_m/dθ for fusion-anchor path (ΔS fixed)…")
dS_fixed = EXP["dS_fus_kB"] * K_B
Tm_torch = (U_l - U_s) / dS_fixed
params = [p for p in MODEL.parameters() if p.requires_grad or True]
for p in MODEL.parameters():
    p.requires_grad_(True)
MODEL.zero_grad(set_to_none=True)
Tm_torch.backward()
gnorm = float(torch.sqrt(sum((p.grad**2).sum() for p in MODEL.parameters()
                             if p.grad is not None)))
n_with_grad = sum(1 for p in MODEL.parameters() if p.grad is not None
                  and float(p.grad.abs().max()) > 0)
print(f"   ||dT_m/dθ|| = {gnorm:.3e} K/weight-unit over "
      f"{n_with_grad} tensors with nonzero grad")
print("   → T_m IS differentiable in the MACE weights (ΔH_fus carries it).")

result = dict(
    exp=EXP, n_solid=n_s, n_liquid=n_l,
    a_relaxed=a_relaxed,
    solid=dict(V0_per_atom=float(V0_pa), B0_GPa=float(B0 / n_s) * 160.2176,
               Bprime=float(Bp), theta_D_K=float(theta_D),
               U_s_eV=float(U_s), S_s_kB=float(S_s) / K_B, T_ref=T_ref),
    liquid=dict(melt_temp=args.melt_temp, T_ref=T_ref, n_snapshots=len(snaps),
                msd_last_A2=msd, U_l_pot_eV=float(U_l_pot), U_l_eV=float(U_l),
                theta_liquid_proxy_K=float(theta_l),
                S_l_proxy_kB=float(S_l_proxy) / K_B),
    dH_fus_eV=float(dH_fus),
    T_m_estimates=estimates,
    autograd=dict(grad_norm_dTm_dtheta=gnorm, n_tensors_nonzero_grad=n_with_grad),
    runtime_s=time.time() - t0,
)
json.dump(result, open(OUT / "melting_point_scope.json", "w"), indent=1)
print(f"\nWrote {OUT/'melting_point_scope.json'}   ({result['runtime_s']:.0f} s)")
