"""
Al-Si rung: a two-lattice eutectic with DEGENERATE solubility,
entirely from MACE-MP-0 energetics (no EAM exists for Al-Si on NIST).

New ingredient vs Ag-Cu: the two solid phases live on DIFFERENT lattices
(fcc Al vs diamond Si), so each solution curve needs its own end-member
reference states, connected by engine-computed PROMOTION (lattice
stability) energies:

    G_fcc(c) = c·ΔE_Si(fcc−dia) + Ω_fcc c(1−c) + kT·ideal
    G_dia(c) = (1−c)·ΔE_Al(dia−fcc) + Ω_dia c(1−c) + kT·ideal
    G_L(c)   = (1−c)·ΔG_fus(Al) + c·ΔG_fus(Si) + kT·ideal   (Ω_L = 0)

with c = x_Si and the stable solids (fcc Al, diamond Si) as the zero of
each end. Lattice stabilities are a contested quantity between DFT and
CALPHAD practice (DFT fcc-Si ≈ +0.50 eV; the SGTE description
51000 − 21.8·T J/mol gives +0.53 eV at 0 K falling to ≈ +0.34 eV at the
eutectic — Wang et al. Calphad 2004, Grimvall et al. RMP 2012); here
∂T_e/∂ΔE_prom makes that spread a measurable diagram sensitivity.

Liquid is ideal + experimental fusion data — NO literature interaction
parameter anywhere; ∂T_e/∂Ω_L reports what liquid physics is worth.

Degeneracy stress test: Al is ~insoluble in Si (c₃ → 1), the fcc solvus is
~1.6 at% — the float64 + global-tangency machinery is required.

Experimental: T_e = 850 K, x_Si(eutectic) = 0.122, solvus 1.65 at% Si.
Runtime ~5 min (MACE CPU). Output: examples/output/alsi_rung.json
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
from coexist.core.phase_diagram import (  # noqa: E402
    K_B, classify_invariant, global_tangency_gap, liquid_solution,
    three_phase_equilibrium)

OUT = Path(__file__).parent / "output"
results = {}

DH_FUS_AL, T_M_AL = 0.11099, 933.47
DH_FUS_SI, T_M_SI = 0.52040, 1687.0     # 50.21 kJ/mol

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


# ═══ 1. End-members and promotion energies from the engine ═══════════════════
print("── end-members + lattice stabilities (MACE-MP-0)")
# Fidelity protocol: converged cells (fcc 108, diamond 216).
E_fcc_Al = relaxed_E(bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3)))
E_dia_Si = relaxed_E(bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3)))
E_fcc_Si = relaxed_E(bulk("Si", "fcc", a=3.90, cubic=True).repeat((3, 3, 3)))
E_dia_Al = relaxed_E(bulk("Al", "diamond", a=6.00, cubic=True).repeat((3, 3, 3)))

dE_Si = E_fcc_Si - E_dia_Si     # fcc promotion of Si
dE_Al = E_dia_Al - E_fcc_Al     # diamond promotion of Al
print(f"   ΔE_Si(fcc−dia) = {dE_Si*1e3:+.0f} meV/atom "
      f"(DFT lit ≈ +500; SGTE 0 K +529 → ≈+337 at T_e)")
print(f"   ΔE_Al(dia−fcc) = {dE_Al*1e3:+.0f} meV/atom (DFT lit ≈ +380)")

# ═══ 2. Solution interactions on each lattice (dilute, x = 1/16) ══════════════
fcc_mix = substitute(bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3)),
                     "Si", 7, seed=0)                       # Al101Si7
E_fcc_mix = relaxed_E(fcc_mix)
x = 7 / 108
om_fcc = (E_fcc_mix - (1 - x) * E_fcc_Al - x * E_fcc_Si) / (x * (1 - x))

dia_mix = substitute(bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3)),
                     "Al", 14, seed=0)                      # Si202Al14
E_dia_mix = relaxed_E(dia_mix)
xa = 14 / 216
om_dia = (E_dia_mix - (1 - xa) * E_dia_Si - xa * E_dia_Al) / (xa * (1 - xa))
print(f"   Ω_fcc(Al-Si) = {om_fcc*1e3:+.0f} meV, "
      f"Ω_dia(Si-Al) = {om_dia*1e3:+.0f} meV")

# ═══ 3. Curves and the invariant ══════════════════════════════════════════════
dE_Si_j, dE_Al_j = jnp.float64(dE_Si), jnp.float64(dE_Al)

def make_curves(dSi, dAl, om_L):
    def G_fcc(c, T):
        return (c * dSi + om_fcc * c * (1 - c)
                + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))

    def G_dia(c, T):
        return ((1 - c) * dAl + om_dia * c * (1 - c)
                + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))

    G_L = liquid_solution(om_L, dH_fus_A=DH_FUS_AL, T_m_A=T_M_AL,
                          dH_fus_B=DH_FUS_SI, T_m_B=T_M_SI)
    return G_fcc, G_L, G_dia

def invariant(dSi, dAl, om_L):
    G_fcc, G_L, G_dia = make_curves(dSi, dAl, om_L)
    return three_phase_equilibrium(G_fcc, G_L, G_dia,
                                   c_guesses=(0.01, 0.15, 0.995),
                                   T_guess=900.0)

c1, c2, c3, T_e = invariant(dE_Si_j, dE_Al_j, 0.0)
G_fcc, G_L, G_dia = make_curves(dE_Si_j, dE_Al_j, 0.0)
gap = global_tangency_gap((G_fcc, G_L, G_dia), (c1, c2, c3), float(T_e))
kind = classify_invariant(("solid", "liquid", "solid"))

print(f"\n── Al-Si invariant from MACE (exp: T_e = 850 K, x_e = 0.122, "
      f"solvus 0.017)")
print(f"   T_e = {float(T_e):.1f} K, fcc solvus = {float(c1):.4f}, "
      f"x_e = {float(c2):.3f}, Si limb = {float(c3):.6f} → {kind}")
print(f"   global tangency gap = {gap:+.1e} eV (verified)")

# ═══ 4. Sensitivities: lattice stability + liquid physics ═════════════════════
grads = jax.jacobian(lambda a: jnp.stack(invariant(a[0], a[1], a[2])))(
    jnp.array([float(dE_Si), float(dE_Al), 0.0]))
dTe = np.asarray(grads)[3]
print(f"\n── what the diagram is sensitive to (∂T_e/∂·, K/eV)")
print(f"   lattice stability Si: {dTe[0]:+.0f}  |  Al: {dTe[1]:+.0f}  |  "
      f"liquid Ω_L: {dTe[2]:+.0f}")
print(f"   → SGTE-vs-DFT ΔE_Si difference (−260 meV) would move T_e by "
      f"{dTe[0]*(-0.26):+.0f} K; ∂x_e/∂Ω_L = "
      f"{float(np.asarray(grads)[1,2]):+.3f} /eV")

results.update({
    "E_promotion_Si_fcc_minus_dia_eV": dE_Si,
    "E_promotion_Al_dia_minus_fcc_eV": dE_Al,
    "omega_fcc_eV": float(om_fcc), "omega_dia_eV": float(om_dia),
    "T_e_K": float(T_e), "fcc_solvus": float(c1), "x_e": float(c2),
    "si_limb": float(c3), "tangency_gap": gap,
    "dTe_dparams_K_per_eV": dTe.tolist(),
    "dxe_dOmL_per_eV": float(np.asarray(grads)[1, 2]),
})
with open(OUT / "alsi_rung.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'alsi_rung.json'}")
