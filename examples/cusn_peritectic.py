"""
Cu-Sn peritectic: a second attempt at the classifier's peritectic branch,
after Ni-Al and Cu-Zn both established the sought reaction ABSENT rather
than finding one. Cu-rich Cu-Sn has the classic textbook peritectic
L + alpha(fcc) -> beta(bcc, W-type/Cu41Sn11-ish) at 798 C = 1071.15 K,
independently cross-checked (weight-% source vs atomic-% source agree to
~0.1 at.%) against a primary source
(Fuertauer, Li, Cupid & Flandorfer, "The Cu-Sn phase diagram, Part I: New
experimental results", Intermetallics 34:142 (2013), via PMC4819024,
Table 5 DTA results): alpha ~7.7 at% Sn, beta ~13 at% Sn, liquid ~15.4 at% Sn.

New ingredient vs Cu-Zn: pure Sn's REAL ground state is neither fcc nor
hcp but body-centered TETRAGONAL beta-Sn (space group I41/amd, a=5.8318,
c=3.1819 A -- "white tin", stable above 13 C, the right reference this
far above room temperature). This is a genuinely anisotropic cell, so
(unlike every cubic structure elsewhere in this paper) it needs full
cell-SHAPE relaxation, not the isotropic-only `relax_volume` used
everywhere else -- done here with ASE's FrechetCellFilter + BFGS,
verified to lower the energy relative to the unrelaxed experimental-
parameter cell (a real relaxation, not a no-op): a 5.83->6.06 A,
c 3.18->3.20 A, E/atom -3.956->-3.981 eV.

Same three-lattice, engine-promotion-energy construction as Cu-Zn:
    G_fcc(c) = c*dE_Sn(fcc-betaSn) + RK_fcc(c) + kT*ideal
    G_bcc(c) = (1-c)*dE_Cu(bcc-fcc) + c*dE_Sn(bcc-betaSn) + RK_bcc(c) + kT*ideal
    G_L(c)   = ideal + experimental fusion data (no literature interaction)
c = x_Sn; stable pure solids (fcc Cu, beta-Sn) are the references.

Runtime ~15-20 min (MACE CPU relaxations, +cell-shape relax for Sn).
Output: examples/output/cusn_peritectic.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402
from ase.filters import FrechetCellFilter  # noqa: E402
from ase.optimize import BFGS  # noqa: E402

from coexist.adapters.lammps_factory import (  # noqa: E402
    relax_positions, relax_volume)
from coexist.core.mixing import redlich_kister_fit  # noqa: E402
from coexist.core.phase_diagram import (  # noqa: E402
    K_B, classify_invariant, global_tangency_gap, liquid_solution,
    three_phase_equilibrium)

OUT = Path(__file__).parent / "output"
results = {}

DH_FUS_CU, T_M_CU = 0.13742, 1357.77
DH_FUS_SN, T_M_SN = 0.07286, 505.08     # 7.03 kJ/mol, real (beta-Sn) melting point

print("Loading MACE-MP-0 (small, float64, CPU)…")
from mace.calculators import mace_mp  # noqa: E402

_calc = mace_mp(model="small", device="cpu", default_dtype="float64")
fac = lambda: _calc                      # noqa: E731


def relaxed_E(atoms, fmax=0.05, cycles=2):
    """Isotropic volume + internal-coordinate relaxation -- the protocol
    used for every CUBIC structure elsewhere in this paper."""
    work = atoms
    for _ in range(cycles):
        work, _, _ = relax_volume(work, fac, scale_range=0.06, n_points=9)
        work = relax_positions(work, fac, fmax=fmax)
    return float(work.get_potential_energy()) / len(work)


def relaxed_E_full_cell(atoms, fmax=0.01, steps=200):
    """Full cell-SHAPE relaxation (not just isotropic volume) -- needed
    only for beta-Sn's tetragonal reference cell, whose c/a ratio
    `relax_volume`'s single scale factor cannot touch."""
    work = atoms.copy()
    work.calc = fac()
    ecf = FrechetCellFilter(work)
    BFGS(ecf, logfile=None).run(fmax=fmax, steps=steps)
    return float(work.get_potential_energy()) / len(work), work


def substitute(atoms, el_new, n_sub, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(atoms), size=n_sub, replace=False)
    syms = np.array(atoms.get_chemical_symbols(), dtype=object)
    syms[idx] = el_new
    atoms.set_chemical_symbols(list(syms))
    return atoms


def fcc_cell(x_sn, reps=(3, 3, 3)):
    a = (1 - x_sn) * 3.61 + x_sn * 4.30   # Vegard-style guess; relaxed anyway
    return bulk("Cu", "fcc", a=a, cubic=True).repeat(reps)


def bcc_cell(x_sn, reps=(4, 4, 4)):
    a = (1 - x_sn) * 2.88 + x_sn * 3.30
    return bulk("Cu", "bcc", a=a, cubic=True).repeat(reps)


def beta_sn_cell():
    a, c = 5.8318, 3.1819
    frac = [(0, 0, 0), (0, 0.5, 0.25), (0.5, 0.5, 0.5), (0.5, 0, 0.75)]
    return Atoms("Sn4", scaled_positions=frac, cell=[a, a, c, 90, 90, 90],
                pbc=True)


# ═══ 1. End-members and promotion energies ════════════════════════════════════
print("── end-members + lattice stabilities (MACE-MP-0)")
E_fcc_Cu = relaxed_E(fcc_cell(0.0))
E_beta_Sn, _ = relaxed_E_full_cell(beta_sn_cell())
E_fcc_Sn = relaxed_E(bulk("Sn", "fcc", a=4.70, cubic=True).repeat((3, 3, 3)))
E_bcc_Cu = relaxed_E(bcc_cell(0.0))
E_bcc_Sn = relaxed_E(bulk("Sn", "bcc", a=3.70, cubic=True).repeat((4, 4, 4)))

dE_Sn_fcc = E_fcc_Sn - E_beta_Sn
dE_Cu_bcc = E_bcc_Cu - E_fcc_Cu
dE_Sn_bcc = E_bcc_Sn - E_beta_Sn
print(f"   E(beta-Sn, real ground state) = {E_beta_Sn:+.4f} eV/atom")
print(f"   dE_Sn(fcc-betaSn) = {dE_Sn_fcc*1e3:+.0f} meV")
print(f"   dE_Cu(bcc-fcc)    = {dE_Cu_bcc*1e3:+.0f} meV")
print(f"   dE_Sn(bcc-betaSn) = {dE_Sn_bcc*1e3:+.0f} meV")

# ═══ 2. Mixing on each lattice (random supercells, 2 seeds) ═══════════════════
# alpha (fcc): dilute Sn in Cu, bracketing the assessed ~7.7 at% solvus limb.
# beta (bcc): bracketing the assessed ~13 at% peritectic composition.
print("── mixing energetics (random supercells)")
xs_fcc = np.array([3, 8, 14]) / 108.0
dH_fcc = []
for k, x in zip((3, 8, 14), xs_fcc):
    vals = []
    for s in (0, 1):
        E = relaxed_E(substitute(fcc_cell(float(x)), "Sn", k, seed=s))
        vals.append(E - (1 - x) * E_fcc_Cu - x * E_fcc_Sn)
    dH_fcc.append(np.mean(vals))
    print(f"   fcc x_Sn={x:.4f}: dH = {dH_fcc[-1]*1e3:+.0f} meV")
L_fcc = redlich_kister_fit(jnp.array(xs_fcc), jnp.array(dH_fcc), order=1)

xs_bcc = np.array([10, 17, 26]) / 128.0
dH_bcc = []
for k, x in zip((10, 17, 26), xs_bcc):
    vals = []
    for s in (0, 1):
        E = relaxed_E(substitute(bcc_cell(float(x)), "Sn", k, seed=s))
        vals.append(E - (1 - x) * E_bcc_Cu - x * E_bcc_Sn)
    dH_bcc.append(np.mean(vals))
    print(f"   bcc x_Sn={x:.4f}: dH = {dH_bcc[-1]*1e3:+.0f} meV")
L_bcc = redlich_kister_fit(jnp.array(xs_bcc), jnp.array(dH_bcc), order=1)
print(f"   RK(fcc) = [{', '.join(f'{float(v)*1e3:+.0f}' for v in L_fcc)}] meV, "
      f"RK(bcc) = [{', '.join(f'{float(v)*1e3:+.0f}' for v in L_bcc)}] meV")

# ═══ 3. Curves and the invariant hunt ═════════════════════════════════════════
dSf, dCb, dSb = (jnp.float64(dE_Sn_fcc), jnp.float64(dE_Cu_bcc),
                 jnp.float64(dE_Sn_bcc))


def rk_series(L, c):
    return sum(L[k] * (1.0 - 2.0 * c) ** k for k in range(len(L)))


def G_fcc(c, T):
    return (c * dSf + c * (1 - c) * rk_series(L_fcc, c)
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))


def G_bcc(c, T):
    return ((1 - c) * dCb + c * dSb + c * (1 - c) * rk_series(L_bcc, c)
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))


G_L = liquid_solution(0.0, dH_fus_A=DH_FUS_CU, T_m_A=T_M_CU,
                      dH_fus_B=DH_FUS_SN, T_m_B=T_M_SN)

ALL = (G_fcc, G_bcc, G_L)
print("\n── invariant hunt (assessed: peritectic L+alpha->beta at 1071.15 K; "
      "alpha 0.077 / beta 0.13 / L 0.154, Fuertauer et al. Intermetallics 2013)")
c1, c2, c3, T = three_phase_equilibrium(
    G_fcc, G_bcc, G_L, c_guesses=(0.077, 0.13, 0.154), T_guess=1071.15)


def inv_residual(cs, T):
    r = []
    d1 = float(jax.grad(G_fcc, argnums=0)(cs[0], T))
    for Gi, ci in ((G_bcc, cs[1]), (G_L, cs[2])):
        r.append(abs(d1 - float(jax.grad(Gi, argnums=0)(ci, T))))
        r.append(abs(float(Gi(ci, T)) - float(G_fcc(cs[0], T))
                     - d1 * (float(ci) - float(cs[0]))))
    return max(r)


res = inv_residual((c1, c2, c3), float(T))
gap = global_tangency_gap(ALL, (c1, c2, c3), float(T))
kind = classify_invariant(("solid", "solid", "liquid"))
converged = res < 1e-8
if converged and gap > -1e-6:
    tag = "VERIFIED"
elif not converged:
    tag = f"STALLED (residual {res:.0e} — no root; gap {gap:+.1e} alone " \
          f"would have passed it)"
else:
    tag = f"REJECTED (gap {gap:+.1e})"
print(f"   alpha/beta/L: T = {float(T):.1f} K, c = ({float(c1):.4f}, "
      f"{float(c2):.4f}, {float(c3):.4f}) -> {kind} — {tag}")

results["invariant_attempt"] = dict(
    T=float(T), c=[float(c1), float(c2), float(c3)], kind=kind, gap=gap,
    stationarity_residual=res, converged=bool(converged))
results.update({
    "E_beta_Sn_eV": E_beta_Sn,
    "dE_Sn_fcc_minus_betaSn_eV": dE_Sn_fcc,
    "dE_Cu_bcc_minus_fcc_eV": dE_Cu_bcc,
    "dE_Sn_bcc_minus_betaSn_eV": dE_Sn_bcc,
    "L_fcc_eV": np.asarray(L_fcc).tolist(),
    "L_bcc_eV": np.asarray(L_bcc).tolist(),
    "assessed_T_K": 1071.15,
    "assessed_c": [0.077, 0.13, 0.154],
    "assessed_source": "Fuertauer, Li, Cupid & Flandorfer, Intermetallics "
                       "34:142 (2013), via PMC4819024, cross-checked "
                       "wt%->at% against a second at%-reported source",
})
with open(OUT / "cusn_peritectic.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'cusn_peritectic.json'}")
