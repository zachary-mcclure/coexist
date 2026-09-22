"""
A genuine engine-internal derivative: a phase-boundary quantity reverse-mode
differentiated with respect to a MACE-MP-0 alchemical composition coordinate.

Everything else in the paper differentiates engine *outputs* (Redlich-Kister
coefficients, formation energies, sampled liquid interactions), with the
single exception of the eam/alloy linear cross-pair scale (Sec. lambda),
which is a special case. This closes the remaining gap: the derivative reaches
a genuine *internal* input of a foundation-model engine, computed by the
network's own automatic differentiation.

Alchemical coordinate: xi in [0,1] sets the chemical identity of the solute
B-species on the Cu-Ag alloy's B sublattice via a continuous element embedding
(Nam/Peng/Gomez-Bombarelli alchemical MACE): node_attrs_B = (1-xi)Cu + xi Ag.
At xi=1 the B species is Ag (real Cu-Ag). dE/dxi is exact reverse-mode through
the message-passing network -- node_attrs is a differentiable input; the
neighbour graph is held at a fixed, relaxed geometry (the adapter's
snapshot-bridge scope: the derivative omits relaxation response, as elsewhere
in the paper).

Geometry: cells are volume+position relaxed (FrechetCellFilter) once -- pure
Cu, pure Ag, and a random 16/16 Cu-Ag alloy (fixed seed) -- then FROZEN. The
alchemical derivative and its finite-difference check both use these frozen
geometries, so they are consistent by construction.

Pipeline:
  MACE E(xi)  ->  equimolar mixing enthalpy dH(xi)  ->  Omega(xi)=4 dH(xi)
              ->  regular-solution consolute Tc(xi)=Omega/2kB
              ->  implicit-layer binodal c(xi) via common_tangent(G,G,T)

Reported, each verified against a central finite difference that re-runs MACE:
  dOmega/dxi     (pure alchemical, reverse-mode through MACE)
  dTc/dxi        (through the analytic consolute point)
  dc_binodal/dxi (through the actual Newton common-tangent solver; chain rule
                  at the engine interface, dc/dOmega [JAX] * dOmega/dxi [MACE])

Output: examples/output/mace_alchemical_gradient.json (32 atoms, --reps 2)
     or examples/output/mace_alchemical_gradient_<N>atom.json (--reps 3 -> 108).
The paper (Sec. mace_alchemical) headlines the converged 108-atom run and cites
the 32-atom run as its cell-size-robustness check; both come from this script.
Runs on a laptop CPU with MACE-MP-0 small in float64 (32 atoms a few minutes,
108 atoms ~25 s: 3 relaxations either way).
"""
import argparse
import warnings
warnings.filterwarnings("ignore")
import json
from pathlib import Path

import numpy as np
import torch
torch.set_default_dtype(torch.float64)

import jax
jax.config.update("jax_enable_x64", True)

from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp

from coexist.core.phase_diagram import K_B, regular_solution, common_tangent

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

_ap = argparse.ArgumentParser()
_ap.add_argument("--reps", type=int, default=2,
                 help="cubic-cell repeats per axis: 2 -> 32 atoms (paper demo), "
                      "3 -> 108 atoms (the campaigns' converged size)")
REPS = _ap.parse_args().reps
N = 4 * REPS**3                    # atoms in the fcc supercell (must be even)

SEED = 0          # random 50/50 arrangement
XI0 = 0.9         # interior alchemical point (B = 90% Ag / 10% Cu); FD-safe
H = 1e-3          # central-difference step on xi

# ----------------------------------------------------------------------
calc = mace_mp(model="small", device="cpu", default_dtype="float64")
MODEL = calc.models[0]
ZTAB = MODEL.atomic_numbers.tolist()
iCu, iAg = ZTAB.index(29), ZTAB.index(47)
NZ = len(ZTAB)


def _base_cell():
    return bulk("Cu", "fcc", a=3.9, cubic=True).repeat((REPS,) * 3)


def _relax(numbers, a0):
    """Volume+position relax a discrete cell; return frozen ASE Atoms."""
    at = _base_cell()
    at.set_cell(np.eye(3) * a0 * REPS, scale_atoms=True)
    at.set_atomic_numbers(numbers)
    at.calc = calc
    FIRE(FrechetCellFilter(at), logfile=None).run(fmax=0.03,
                                                  steps=120 if REPS == 2 else 300)
    return at


def _batch(atoms):
    return calc._atoms_to_batch(atoms).to_dict()


def _energy(bd, b_mask, xi, want_grad=False):
    """Total energy (eV); B-sites (b_mask) carry (1-xi)Cu+xi Ag, else pure Cu.
    Returns E, or (E, dE/dxi) reverse-mode over the B-sites, at fixed geometry."""
    N = b_mask.shape[0]
    na = torch.zeros((N, NZ), dtype=torch.float64)
    na[:, iCu] = 1.0
    na[b_mask, iCu] = 1.0 - xi
    na[b_mask, iAg] = xi
    if want_grad:
        na.requires_grad_(True)
    d = dict(bd)
    d["node_attrs"] = na
    E = MODEL(d, compute_force=False, training=want_grad)["energy"].sum()
    if want_grad:
        g, = torch.autograd.grad(E, na)
        dEdxi = float((g[b_mask, iAg] - g[b_mask, iCu]).sum())
        return float(E), dEdxi
    return float(E)


# --- frozen geometries: pure Cu, pure Ag, random 50/50 Cu-Ag alloy ---
rng = np.random.default_rng(SEED)
perm = rng.permutation(N)
B_SITES = np.zeros(N, dtype=bool)
B_SITES[perm[:N // 2]] = True      # the alloy B (Ag) sites
b_mask = torch.tensor(B_SITES)
none_mask = torch.zeros(N, dtype=torch.bool)
all_mask = torch.ones(N, dtype=torch.bool)

at_Cu = _relax(np.full(N, 29), 3.62)
at_Ag = _relax(np.full(N, 47), 4.09)
num_alloy = np.full(N, 29); num_alloy[B_SITES] = 47
at_mix = _relax(num_alloy, 3.85)
BD_Cu, BD_Ag, BD_mix = _batch(at_Cu), _batch(at_Ag), _batch(at_mix)

E_Cu = _energy(BD_Cu, none_mask, 0.0)          # xi-independent reference


def omega(xi, want_grad=False):
    """Omega(xi)=4*dH_mix(1/2), fixed frozen geometries. eV [, dOmega/dxi]."""
    if want_grad:
        E_mix, dmix = _energy(BD_mix, b_mask, xi, want_grad=True)
        E_Bp, dBp = _energy(BD_Ag, all_mask, xi, want_grad=True)
        dH = E_mix / N - 0.5 * E_Cu / N - 0.5 * E_Bp / N
        return 4.0 * dH, 4.0 * (dmix / N - 0.5 * dBp / N)
    E_mix = _energy(BD_mix, b_mask, xi)
    E_Bp = _energy(BD_Ag, all_mask, xi)
    return 4.0 * (E_mix / N - 0.5 * E_Cu / N - 0.5 * E_Bp / N)


def binodal_c(om, T):
    """Dilute-limb binodal of the symmetric regular-solution gap, through the
    repo's Newton common_tangent (seeded asymmetrically off the trivial root)."""
    G = regular_solution(om)
    ca, cb = common_tangent(G, G, float(T),
                            c_alpha_guess=0.10, c_beta_guess=0.90)
    return ca


# ---- Omega and its alchemical gradient (reverse-mode through MACE) ----
Om0, dOm_ad = omega(XI0, want_grad=True)
Om_p, Om_m = omega(XI0 + H), omega(XI0 - H)
dOm_fd = (Om_p - Om_m) / (2 * H)

# ---- consolute temperature (analytic boundary point) ----
Tc0 = Om0 / (2 * K_B)
dTc_ad = dOm_ad / (2 * K_B)
dTc_fd = dOm_fd / (2 * K_B)

# ---- implicit-layer binodal composition at T = 0.6 Tc ----
Tbin = 0.6 * Tc0
c0 = float(binodal_c(Om0, Tbin))
dc_dOm = float(jax.grad(lambda om: binodal_c(om, Tbin))(Om0))   # through the solver
dc_ad = dc_dOm * dOm_ad                                          # chain rule at interface
c_p, c_m = float(binodal_c(Om_p, Tbin)), float(binodal_c(Om_m, Tbin))
dc_fd = (c_p - c_m) / (2 * H)                                    # end-to-end (re-runs MACE)


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


res = {
    "engine": "MACE-MP-0 small (float64, CPU)",
    "model_file": "20231210mace128L0_energy_epoch249model",
    "system": "fcc Cu-Ag, %d atoms, random %d/%d (seed %d), volume+position relaxed, frozen"
              % (N, N // 2, N // 2, SEED),
    "alchemical_coordinate": "xi = Ag-character of B sublattice; node_attrs_B=(1-xi)Cu+xi Ag",
    "xi0": XI0, "fd_step_xi": H,
    "Omega_eV": Om0,
    "Omega_meV": Om0 * 1e3,
    "dOmega_dxi_reverse_mode_eV": dOm_ad,
    "dOmega_dxi_FD_eV": dOm_fd,
    "dOmega_dxi_relerr": rel(dOm_ad, dOm_fd),
    "Tc_K": Tc0,
    "dTc_dxi_reverse_mode_K": dTc_ad,
    "dTc_dxi_FD_K": dTc_fd,
    "dTc_dxi_relerr": rel(dTc_ad, dTc_fd),
    "T_binodal_K": Tbin,
    "binodal_c_dilute": c0,
    "dc_dOmega_JAX_through_solver": dc_dOm,
    "dc_binodal_dxi_reverse_mode": dc_ad,
    "dc_binodal_dxi_FD_end_to_end": dc_fd,
    "dc_binodal_dxi_relerr": rel(dc_ad, dc_fd),
    "n_forward_engine_calls_per_gradient": 3,
}

print("\n=== MACE alchemical gradient through the phase-boundary construction ===")
print(f"Omega(xi={XI0})       = {Om0*1e3:+.2f} meV      (Tc = {Tc0:.0f} K)")
print(f"dOmega/dxi  reverse   = {dOm_ad:+.6f} eV")
print(f"dOmega/dxi  FD        = {dOm_fd:+.6f} eV    relerr {rel(dOm_ad,dOm_fd):.2e}")
print(f"dTc/dxi     reverse   = {dTc_ad:+.1f} K")
print(f"dTc/dxi     FD        = {dTc_fd:+.1f} K      relerr {rel(dTc_ad,dTc_fd):.2e}")
print(f"binodal c (T={Tbin:.0f}K)  = {c0:.5f}")
print(f"dc/dxi      reverse   = {dc_ad:+.5f}   (dc/dOmega={dc_dOm:+.4f} through JAX solver)")
print(f"dc/dxi      FD e2e    = {dc_fd:+.5f}   relerr {rel(dc_ad,dc_fd):.2e}")

_name = ("mace_alchemical_gradient.json" if REPS == 2
         else f"mace_alchemical_gradient_{N}atom.json")
with open(OUT / _name, "w") as f:
    json.dump(res, f, indent=1)
print(f"\nWrote {OUT/_name}")
