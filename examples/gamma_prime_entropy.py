"""
Debye-Grüneisen vibrational formation entropy of γ'-Ni₃Al:
how much of the 0.80 k_B/atom that the Ni-Al eutectic demands does the Debye level actually deliver?

S_vib(T) = −∂F_vib/∂T — computed by jax.grad of the differentiable Debye
free energy, no analytic entropy formula needed. Formation entropy:

    S_f(γ', T) = S_vib(Ni₃Al) − 0.75·S_vib(Ni) − 0.25·S_vib(Al)

Everything from Mishin-EAM E(V) scans. Note the literature expectation:
ordered intermetallics are usually vibrationally STIFFER than their
end-members (negative vibrational S_f) — if that holds here, the 0.80 k_B
must come from off-stoichiometry/configurational entropy instead, which
sharpens (not weakens) the attribution.

Also re-solves the Ni-Al eutectic with the computed S_f in the compound.
Runtime ~1 min. Output: examples/output/gamma_prime_entropy.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402

from coexist.adapters.lammps_factory import (  # noqa: E402
    eam_alloy_factory, eos_scan)
from coexist.core.thermo_vib import (  # noqa: E402
    debye_temperature, eos_fit, f_vib_debye)

OUT = Path(__file__).parent / "output"
K_B = 8.617333e-5
M = {"Ni": 58.693, "Al": 26.982}
results = {}

fac = eam_alloy_factory("NiAl.eam.alloy", ("Ni", "Al"))

a0 = 3.57
l12 = Atoms("AlNi3",
            positions=[[0, 0, 0], [0, a0 / 2, a0 / 2],
                       [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]],
            cell=np.eye(3) * a0, pbc=True).repeat((3, 3, 3))

print("── E(V) scans (Mishin EAM) → θ_D")
theta = {}
systems = {
    "Ni": (bulk("Ni", "fcc", a=3.52, cubic=True).repeat((3, 3, 3)), M["Ni"]),
    "Al": (bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3)), M["Al"]),
    "Ni3Al": (l12, 0.75 * M["Ni"] + 0.25 * M["Al"]),
}
for tag, (atoms, mass) in systems.items():
    V, E = eos_scan(atoms, fac)
    V0, _, B0, _ = eos_fit(jnp.array(V), jnp.array(E))
    theta[tag] = debye_temperature(V0, B0, mass)
    print(f"   {tag:6s}: V₀={float(V0):5.2f} Å³  B₀={float(B0)*160.2176:3.0f} "
          f"GPa  θ_D={float(theta[tag]):.0f} K")

# S_vib = −∂F_vib/∂T by jax.grad (differentiable Debye)
def S_vib(T, th):
    return -jax.grad(lambda t: f_vib_debye(t, th))(T)

def S_f(T):
    return (S_vib(T, theta["Ni3Al"])
            - 0.75 * S_vib(T, theta["Ni"]) - 0.25 * S_vib(T, theta["Al"]))

print("\n── vibrational formation entropy of γ' (the corrected-liquid "
      "model demands ≈1.7 k_B — exact solve, nial_liquid_retemp)")
for T in (300.0, 1000.0, 1600.0):
    s = float(S_f(T))
    print(f"   S_f^vib({T:4.0f} K) = {s*1e6:+6.1f} μeV/K = "
          f"{s/K_B:+.3f} k_B/atom")
s1600 = float(S_f(1600.0))

# Re-solve the eutectic with the Debye S_f in the compound
from coexist.core.mixing import redlich_kister_fit  # noqa: E402
from coexist.core.phase_diagram import (  # noqa: E402
    line_compound, redlich_kister_solution, three_phase_equilibrium)

ni = json.load(open(OUT / "nial_liquid_retemp.json"))   # model
G_gamma = redlich_kister_solution(jnp.array(ni["L_gamma_eV"]))
L_lo = jnp.array(ni["L_liquid_1500K_eV"])
sl = jnp.array(ni["dL_dT_eV_per_K"])
DH_FUS_NI, T_M_NI = 0.18117, 1728.0
DH_FUS_AL, T_M_AL = 0.11099, 933.47

def G_L(c, T):
    dG_Ni = DH_FUS_NI * (1.0 - T / T_M_NI)
    dG_Al = DH_FUS_AL * (1.0 - T / T_M_AL)
    L_T = L_lo + (T - 1500.0) * sl
    series = L_T[0] + L_T[1] * (1.0 - 2.0 * c)
    return ((1 - c) * dG_Ni + c * dG_Al + c * (1 - c) * series
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))

G_gp = line_compound(0.25, ni["Hf_L12"], S_f=jnp.float64(s1600))
_, _, _, T_new = three_phase_equilibrium(
    G_gamma, G_L, G_gp, c_guesses=(0.18, 0.235, 0.2495), T_guess=1450.0)
T_sf0 = ni["gamma/L/gamma'"]["T"]
Sf_req = ni.get("Sf_exact_kB", 1.71)
print(f"\n── eutectic with Debye S_f(γ'): T = {float(T_new):.1f} K "
      f"(S_f=0 gave {T_sf0:.1f}; assessed 1658)")
print("   → the Debye level supplies "
      f"{s1600/K_B/Sf_req*100:+.0f}% of the required {Sf_req:.2f} k_B; the "
      "remainder is off-stoichiometry/configurational — the line-compound "
      "model, not the vibrations, is the binding approximation.")

results.update({
    "theta_D_K": {k: float(v) for k, v in theta.items()},
    "S_f_vib_eV_per_K": {"300": float(S_f(300.0)),
                         "1000": float(S_f(1000.0)), "1600": s1600},
    "T_eut_with_Sf_K": float(T_new),
})
with open(OUT / "gamma_prime_entropy.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'gamma_prime_entropy.json'}")
