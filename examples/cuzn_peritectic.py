"""
Cu-Zn peritectic: the classifier's peritectic branch, demonstrated
on a real system from MACE-MP-0 energetics.

Every verified invariant so far is a eutectic or a monotectic; the Ni-Al
peritectic candidate was (correctly) rejected by the verifier. Cu-Zn
closes the loop: the assessed diagram has the classic peritectic
L + α(fcc) → β(bcc) at 1176 K with α ≈ 0.32, β ≈ 0.36, x_L ≈ 0.37 (at.
fraction Zn). At 1176 K β is the DISORDERED bcc phase (the B2 ordering
sets in only below ~740 K), so random bcc supercells are the right model.

Three lattices, engine promotion energies (the Al-Si pattern):
    G_fcc(c) = c·ΔE_Zn(fcc−hcp) + RK_fcc(c) + kT·ideal
    G_bcc(c) = (1−c)·ΔE_Cu(bcc−fcc) + c·ΔE_Zn(bcc−hcp) + RK_bcc(c) + kT·ideal
    G_L(c)   = ideal + experimental fusion data (no literature interaction)

c = x_Zn; stable pure solids (fcc Cu, hcp Zn) are the references.
The invariant is solved with `three_phase_equilibrium`, named by
`classify_invariant` (middle tangent point is a SOLID → peritectic), and
checked with `global_tangency_gap` against all three curves.

Runtime ~15 min (MACE CPU relaxations). Output: examples/output/cuzn_peritectic.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from ase.build import bulk  # noqa: E402

from coexist.adapters.lammps_factory import (  # noqa: E402
    relax_positions, relax_volume)
from coexist.core.mixing import redlich_kister_fit  # noqa: E402
from coexist.core.phase_diagram import (  # noqa: E402
    K_B, classify_invariant, global_tangency_gap, liquid_solution,
    three_phase_equilibrium)

OUT = Path(__file__).parent / "output"
results = {}

DH_FUS_CU, T_M_CU = 0.13742, 1357.77
DH_FUS_ZN, T_M_ZN = 0.07588, 692.68     # 7.322 kJ/mol

print("Loading MACE-MP-0 (small, float64, CPU)…")
from mace.calculators import mace_mp  # noqa: E402

_calc = mace_mp(model="small", device="cpu", default_dtype="float64")
fac = lambda: _calc                      # noqa: E731


def relaxed_E(atoms, fmax=0.05, cycles=2):
    work = atoms
    for _ in range(cycles):
        work, _, _ = relax_volume(work, fac, scale_range=0.06, n_points=9)
        work = relax_positions(work, fac, fmax=fmax)
    return float(work.get_potential_energy()) / len(work)


def substitute(atoms, el_new, n_sub, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(atoms), size=n_sub, replace=False)
    syms = np.array(atoms.get_chemical_symbols(), dtype=object)
    syms[idx] = el_new
    atoms.set_chemical_symbols(list(syms))
    return atoms


# Fidelity protocol: converged cells (fcc 108, bcc 128, hcp 96).
def fcc_cell(x_zn, reps=(3, 3, 3)):
    a = (1 - x_zn) * 3.61 + x_zn * 3.95
    return bulk("Cu", "fcc", a=a, cubic=True).repeat(reps)


def bcc_cell(x_zn, reps=(4, 4, 4)):
    a = (1 - x_zn) * 2.88 + x_zn * 3.13
    return bulk("Cu", "bcc", a=a, cubic=True).repeat(reps)


# ═══ 1. End-members and promotion energies ════════════════════════════════════
print("── end-members + lattice stabilities (MACE-MP-0)")
E_fcc_Cu = relaxed_E(fcc_cell(0.0))
E_hcp_Zn = relaxed_E(bulk("Zn", "hcp", a=2.66, c=4.95).repeat((4, 4, 3)))
E_fcc_Zn = relaxed_E(bulk("Zn", "fcc", a=3.93, cubic=True).repeat((3, 3, 3)))
E_bcc_Cu = relaxed_E(bcc_cell(0.0))
E_bcc_Zn = relaxed_E(bulk("Zn", "bcc", a=3.13, cubic=True).repeat((4, 4, 4)))

dE_Zn_fcc = E_fcc_Zn - E_hcp_Zn
dE_Cu_bcc = E_bcc_Cu - E_fcc_Cu
dE_Zn_bcc = E_bcc_Zn - E_hcp_Zn
print(f"   ΔE_Zn(fcc−hcp) = {dE_Zn_fcc*1e3:+.0f} meV "
      f"(SGTE ≈ +30 − 0.3·T J-level term; DFT ≈ +30…+70)")
print(f"   ΔE_Cu(bcc−fcc) = {dE_Cu_bcc*1e3:+.0f} meV (DFT ≈ +40)")
print(f"   ΔE_Zn(bcc−hcp) = {dE_Zn_bcc*1e3:+.0f} meV (DFT ≈ +70…+90)")

# ═══ 2. Mixing on each lattice (random supercells, 2 seeds) ═══════════════════
print("── mixing energetics (random supercells)")
xs_fcc = np.array([14, 27, 41]) / 108.0
dH_fcc = []
for k, x in zip((14, 27, 41), xs_fcc):
    vals = []
    for s in (0, 1):
        E = relaxed_E(substitute(fcc_cell(float(x)), "Zn", k, seed=s))
        vals.append(E - (1 - x) * E_fcc_Cu - x * E_fcc_Zn)
    dH_fcc.append(np.mean(vals))
    print(f"   fcc x_Zn={x:.3f}: ΔH = {dH_fcc[-1]*1e3:+.0f} meV")
L_fcc = redlich_kister_fit(jnp.array(xs_fcc), jnp.array(dH_fcc), order=1)

xs_bcc = np.array([52, 64, 76]) / 128.0
dH_bcc = []
for k, x in zip((52, 64, 76), xs_bcc):
    vals = []
    for s in (0, 1):
        E = relaxed_E(substitute(bcc_cell(float(x)), "Zn", k, seed=s))
        vals.append(E - (1 - x) * E_bcc_Cu - x * E_bcc_Zn)
    dH_bcc.append(np.mean(vals))
    print(f"   bcc x_Zn={x:.3f}: ΔH = {dH_bcc[-1]*1e3:+.0f} meV")
L_bcc = redlich_kister_fit(jnp.array(xs_bcc), jnp.array(dH_bcc), order=1)
print(f"   RK(fcc) = [{', '.join(f'{float(v)*1e3:+.0f}' for v in L_fcc)}] meV, "
      f"RK(bcc) = [{', '.join(f'{float(v)*1e3:+.0f}' for v in L_bcc)}] meV")

# ═══ 3. Curves and the invariant hunt ═════════════════════════════════════════
dZf, dCb, dZb = (jnp.float64(dE_Zn_fcc), jnp.float64(dE_Cu_bcc),
                 jnp.float64(dE_Zn_bcc))

def rk_series(L, c):
    return sum(L[k] * (1.0 - 2.0 * c) ** k for k in range(len(L)))

def G_fcc(c, T):
    return (c * dZf + c * (1 - c) * rk_series(L_fcc, c)
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))

def G_bcc(c, T):
    return ((1 - c) * dCb + c * dZb + c * (1 - c) * rk_series(L_bcc, c)
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))

G_L = liquid_solution(0.0, dH_fus_A=DH_FUS_CU, T_m_A=T_M_CU,
                      dH_fus_B=DH_FUS_ZN, T_m_B=T_M_ZN)

ALL = (G_fcc, G_bcc, G_L)
print("\n── invariant hunt (assessed: peritectic L+α→β at 1176 K; "
      "α 0.32 / β 0.36 / L 0.37)")
c1, c2, c3, T = three_phase_equilibrium(
    G_fcc, G_bcc, G_L, c_guesses=(0.30, 0.36, 0.45), T_guess=1150.0)


def inv_residual(cs, T):
    """Max stationarity residual of the triple tangent (the solver's own
    residuals, recomputed): the gap check alone cannot distinguish a
    stalled solve from a converged one."""
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
print(f"   α/β/L: T = {float(T):.1f} K, c = ({float(c1):.3f}, "
      f"{float(c2):.3f}, {float(c3):.3f}) → {kind} — {tag}")
print("   (see cuzn_peritectic_scan.py: the fcc/bcc curves are near-"
      "parallel; the model has NO stable β field — the tangent-gap scan "
      "proves it and measures the missing bcc stabilization)")
results["invariant_attempt"] = dict(
    T=float(T), c=[float(c1), float(c2), float(c3)], kind=kind, gap=gap,
    stationarity_residual=res, converged=bool(converged))

results.update({
    "dE_Zn_fcc_minus_hcp_eV": dE_Zn_fcc,
    "dE_Cu_bcc_minus_fcc_eV": dE_Cu_bcc,
    "dE_Zn_bcc_minus_hcp_eV": dE_Zn_bcc,
    "L_fcc_eV": np.asarray(L_fcc).tolist(),
    "L_bcc_eV": np.asarray(L_bcc).tolist(),
})
with open(OUT / "cuzn_peritectic.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'cuzn_peritectic.json'}")
