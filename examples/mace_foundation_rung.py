"""
Engine-fidelity axis, top rung: the SAME pipeline —
fcc_solution → relax → ΔH_mix → Ω → differentiable eutectic — with a
foundation-model potential (MACE-MP-0 small) instead of EAM or EMT.
Zero system-specific fitting anywhere.

The point of the rung: the engine is one constructor line. EMT → EAM →
MACE never touches the thermodynamic construction, and ∂T_e/∂R flows
through whichever engine is plugged in, one engine call per gradient.

Runtime ~5-15 min on CPU (MACE float64 relaxations dominate; single seed,
one volume+position cycle — the EAM campaign quantifies seed scatter at
~4% and cycle effects at ~4 meV for this system).
Output: examples/output/mace_rung.json
"""
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from coexist.adapters.lammps_factory import (  # noqa: E402
    adapter_for_atoms, fcc_solution, relax_positions, relax_volume)
from coexist.core.mixing import omega_regular  # noqa: E402
from coexist.core.phase_diagram import (  # noqa: E402
    K_B, eutectic_point, liquid_solution, regular_solution)

OUT = Path(__file__).parent / "output"
results = {}

print("Loading MACE-MP-0 (small, float64, CPU)…")
from mace.calculators import mace_mp  # noqa: E402

_calc = mace_mp(model="small", device="cpu", default_dtype="float64")
mace_factory = lambda: _calc          # noqa: E731 — cached; MACE is stateless


def relaxed_energy(atoms, fmax=0.05):
    """One volume + one position pass (MACE CPU budget), per-atom energy."""
    work, _, _ = relax_volume(atoms, mace_factory)
    work = relax_positions(work, mace_factory, fmax=fmax)
    return float(work.get_potential_energy()) / len(work), work


t0 = time.time()
print("── Ag-Cu from MACE-MP-0 (exp T_e=1052 K; EAM gave 1054, EMT 1136)")
E_Ag, _ = relaxed_energy(fcc_solution("Ag", "Cu", 0.0, reps=(3, 3, 3), seed=0))
E_Cu, _ = relaxed_energy(fcc_solution("Cu", "Ag", 0.0, reps=(3, 3, 3), seed=0))
E_mix, mix_atoms = relaxed_energy(fcc_solution("Ag", "Cu", 0.5, reps=(3, 3, 3), seed=0))
dH = E_mix - 0.5 * E_Ag - 0.5 * E_Cu
om_mace = dH / 0.25
print(f"   ΔH_mix(0.5) = {dH*1e3:+.1f} meV/atom → Ω = {om_mace*1e3:+.0f} meV "
      f"(EAM +372, EMT +219, CALPHAD ≈ +340)   [{time.time()-t0:.0f} s]")

G_L = liquid_solution(0.1555, dH_fus_A=0.11691, T_m_A=1234.93,
                      dH_fus_B=0.13742, T_m_B=1357.77)
ca, cl, cb, Te = eutectic_point(regular_solution(om_mace), G_L)
print(f"   eutectic: T_e = {float(Te):.1f} K  x_e = {float(cl):.3f}  "
      f"c_α = {float(ca):.3f}")
results["AgCu"] = dict(omega_eV=om_mace, T_e_K=float(Te),
                       x_e=float(cl), c_alpha=float(ca))

# ∂T_e/∂R through the foundation model — identical code path as LAMMPS
ad = adapter_for_atoms(mix_atoms, mace_factory, dtype=jnp.float64,
                       name="MACE-MP-0")
R0 = jnp.array(mix_atoms.get_positions())
E_ref = 0.5 * E_Ag + 0.5 * E_Cu

def Te_of_R(R):
    dH_R = ad(R) / len(mix_atoms) - E_ref
    _, _, _, Te = eutectic_point(regular_solution(omega_regular(dH_R, 0.5)),
                                 G_L)
    return Te

n0 = ad.n_calls
gR = jax.grad(Te_of_R)(R0)
print(f"   ∂T_e/∂R through MACE: max |∂T_e/∂R_i| = "
      f"{float(jnp.abs(gR).max()):.2f} K/Å in {ad.n_calls-n0} engine call(s)")
results["AgCu_dTe_dR_max"] = float(jnp.abs(gR).max())

t1 = time.time()
print("\n── Cu-Ni from MACE-MP-0 (CALPHAD T_c = 600-650 K; EAM gave 600)")
E_Cu2, _ = relaxed_energy(fcc_solution("Cu", "Ni", 0.0, reps=(3, 3, 3), seed=0))
E_Ni, _ = relaxed_energy(fcc_solution("Ni", "Cu", 0.0, reps=(3, 3, 3), seed=0))
E_mx2, _ = relaxed_energy(fcc_solution("Cu", "Ni", 0.5, reps=(3, 3, 3), seed=0))
om_cuni = (E_mx2 - 0.5 * E_Cu2 - 0.5 * E_Ni) / 0.25
print(f"   Ω = {om_cuni*1e3:+.0f} meV → T_c = {om_cuni/(2*K_B):.0f} K   "
      f"[{time.time()-t1:.0f} s]")
results["CuNi"] = dict(omega_eV=om_cuni, T_c_K=float(om_cuni / (2 * K_B)))

with open(OUT / "mace_rung.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'mace_rung.json'}")
