"""
INVERSE CAPSTONE: fit a physical engine parameter to
experimental phase-diagram features by implicit differentiation THROUGH the
eutectic construction.

Parameters θ = (λ, Ω_L):
  λ    — scale on the Ag-Cu cross-pair function φ_AgCu(r) inside the EAM
         file (a real potential parameter; setfl surgery in
         `scaled_cross_potential`). E(λ) is exactly linear in λ at fixed
         geometry, so TWO engine evaluations per structure make the whole
         optimization loop engine-free.
  Ω_L  — liquid interaction parameter (the one CALPHAD literature input of
         the forward rungs — here it is FITTED instead, removing it).

Targets (experimental Ag-Cu eutectic):
  T_e = 1052 K,  c_α = 0.141 (max Cu solubility in Ag at T_e)

Method: 2×2 Newton on R(θ) = [T_e(θ)−1052, c_α(θ)−0.141], Jacobian by
jax.jacobian through the 4×4 implicit triple-tangent solve (a Newton whose
every residual evaluation contains another Newton — all one JAX graph).

The interesting outputs are NOT the fit itself but:
  1. identifiability — the 2×2 sensitivity matrix says which diagram
     feature constrains which parameter;
  2. physicality — does the fitted λ* reproduce the independently assessed
     CALPHAD Ω_s ≈ 340 meV it was never told about?
  3. transferability — x_e and c_β are NOT fitted; do they improve?
  4. the frozen-geometry error of the linear bridge, checked by re-relaxing
     with the λ*-scaled potential file.

float64. Runtime ~2 min. Output: examples/output/inverse_capstone.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from coexist.adapters.lammps_factory import (  # noqa: E402
    eam_alloy_factory, fcc_solution, relax_full, scaled_cross_potential)
from coexist.core.mixing import redlich_kister_fit  # noqa: E402
from coexist.core.phase_diagram import (  # noqa: E402
    eutectic_diagram, eutectic_point, liquid_solution,
    redlich_kister_solution)

OUT = Path(__file__).parent / "output"
SCRATCH = OUT / "scaled_potentials"
SCRATCH.mkdir(parents=True, exist_ok=True)

T_M_AG, DH_AG = 1234.93, 0.11691
T_M_CU, DH_CU = 1357.77, 0.13742
TARGET_TE, TARGET_CA = 1052.0, 0.141
# Fidelity protocol: 108-atom cells (3×3×3), converged per the
# size-scaling study.
REPS = (3, 3, 3)
XS = np.array([14, 27, 41, 54, 67, 81, 94]) / 108.0

results = {}
engine_calls = {"relax": 0, "lambda_pair": 0}

# ═══ 1. Engine work: relax at λ=1, then E(λ=1) and E(λ=0) per structure ═══════
fac1 = eam_alloy_factory("CuAg.eam.alloy", ("Cu", "Ag"))
p0 = scaled_cross_potential("CuAg.eam.alloy", 0.0, out_dir=SCRATCH)
fac0 = eam_alloy_factory(p0, ("Cu", "Ag"))

print("── engine phase: relax (λ=1) + two single points per structure")
E_pure = {}
for el, other in [("Ag", "Cu"), ("Cu", "Ag")]:
    rel, _ = relax_full(fcc_solution(el, other, 0.0, reps=REPS, seed=0), fac1)
    E_pure[el] = float(rel.get_potential_energy()) / len(rel)

structures = {}
E1x, E0x = [], []
for x in XS:
    rel, _ = relax_full(fcc_solution("Ag", "Cu", float(x), reps=REPS, seed=0),
                        fac1)
    structures[float(x)] = rel
    w1 = rel.copy(); w1.calc = fac1(); e1 = float(w1.get_potential_energy())
    w0 = rel.copy(); w0.calc = fac0(); e0 = float(w0.get_potential_energy())
    E1x.append(e1 / len(rel)); E0x.append(e0 / len(rel))
    engine_calls["lambda_pair"] += 2
E1x, E0x = jnp.array(E1x), jnp.array(E0x)
ref = jnp.array([(1 - x) * E_pure["Ag"] + x * E_pure["Cu"] for x in XS])
xs_j = jnp.array(XS)

def L_of_lambda(lam):
    """RK coefficients as an (exactly linear) function of the engine λ."""
    dH = E0x + lam * (E1x - E0x) - ref
    return redlich_kister_fit(xs_j, dH, order=2)

print(f"   Ω_s(x=½): λ=1 → {float(L_of_lambda(1.0)[0])*1e3:.0f} meV(L0); "
      f"λ=0 → {float(L_of_lambda(0.0)[0])*1e3:.0f} meV(L0)")

# ═══ 2. The engine-free inverse loop: 2×2 Newton through the 4×4 solve ════════
def features(theta):
    lam, om_L = theta
    G_s = redlich_kister_solution(L_of_lambda(lam))
    G_L = liquid_solution(om_L, dH_fus_A=DH_AG, T_m_A=T_M_AG,
                          dH_fus_B=DH_CU, T_m_B=T_M_CU)
    ca, cl, cb, Te = eutectic_point(G_s, G_L)
    return jnp.array([Te, ca]), (cl, cb)

def residual(theta):
    """T-residual in K; composition as log(c_α/target) — the solvus responds
    exponentially to interaction changes, so the log form is the
    well-conditioned one (the 108-atom energetics diverge a plain
    Newton on raw c_α)."""
    f, _ = features(theta)
    return jnp.array([f[0] - TARGET_TE, jnp.log(f[1] / TARGET_CA)])

theta = jnp.array([1.0, 0.1555])            # start: raw EAM + CALPHAD liquid
J_fn = jax.jacobian(residual)
W = jnp.array([0.01, 1.0])                  # K → O(1) for the norm

f0, (cl_raw, cb_raw) = features(theta)
x_e_raw, c_b_raw = float(cl_raw), float(cb_raw)
Jphys = jax.jacobian(lambda th: features(th)[0])(theta)
print("\n── identifiability at the start (raw EAM, CALPHAD Ω_L)")
print(f"   features: T_e = {float(f0[0]):.1f} K, c_α = {float(f0[1]):.3f} "
      f"(targets {TARGET_TE:.0f}, {TARGET_CA})")
print(f"   J = ∂(T_e, c_α)/∂(λ, Ω_L) = [[{float(Jphys[0,0]):+8.1f} K/λ, "
      f"{float(Jphys[0,1]):+8.1f} K/eV],")
print(f"                                [{float(Jphys[1,0]):+8.3f} /λ,  "
      f"{float(Jphys[1,1]):+8.3f} /eV ]]  cond = "
      f"{float(jnp.linalg.cond(Jphys)):.1f}")
results["cond_J_physical"] = float(jnp.linalg.cond(Jphys))

print("\n── damped Newton iterations (engine-free)")
n_iters = 0
for it in range(30):
    r = residual(theta)
    if float(jnp.abs(r[0])) < 1e-4 and float(jnp.abs(r[1])) < 1e-6:
        break
    step = jnp.linalg.solve(J_fn(theta), r)
    nr0 = float(jnp.linalg.norm(W * r))
    for frac in (1.0, 0.5, 0.25, 0.1, 0.03):
        trial = theta - frac * step
        if not (0.5 < float(trial[0]) < 2.0
                and -0.5 < float(trial[1]) < 0.5):
            continue                        # keep θ in the physical box
        if float(jnp.linalg.norm(W * residual(trial))) < nr0:
            theta = trial
            break
    else:
        theta = theta - 0.03 * step
    n_iters = it + 1
    f_it, _ = features(theta)
    print(f"   it {it}: λ = {float(theta[0]):.4f}, Ω_L = "
          f"{float(theta[1])*1e3:.1f} meV | T_e = {float(f_it[0]):.2f} K, "
          f"c_α = {float(f_it[1]):.4f}")
results["newton_iterations"] = n_iters

lam_s, omL_s = float(theta[0]), float(theta[1])
f_fin, (cl_f, cb_f) = features(theta)
L_star = L_of_lambda(theta[0])
om_star = float(L_star[0])

chg = 100 * (lam_s - 1)
print("\n── solution and physicality")
print(f"   λ* = {lam_s:.4f} (cross-pair {abs(chg):.1f}% "
      f"{'stronger' if chg > 0 else 'weaker'}), Ω_L* = {omL_s*1e3:.1f} meV")
L0_raw = float(L_of_lambda(1.0)[0]) * 1e3
print(f"   Ω_s(λ*) L0 = {om_star*1e3:.0f} meV vs raw EAM {L0_raw:.0f} and "
      f"0 K CALPHAD ≈ 340: within a T-independent-Ω model, matching the "
      f"1052 K features selects the EFFECTIVE high-T Ω "
      f"(shift {om_star*1e3 - L0_raw:+.0f} meV at T_e; see "
      f"agcu_vibrational.py for the Debye comparison)")
print(f"   Ω_L* = {omL_s*1e3:.1f} meV (CALPHAD literature 155.5)")
results["omega_s_raw_L0_meV"] = L0_raw

print("\n── transferability (NOT fitted): eutectic composition and β limb")
print(f"   x_e = {float(cl_f):.3f} (exp 0.399; raw EAM gave {x_e_raw:.3f})")
print(f"   c_β = {float(cb_f):.3f} (exp 0.951; raw EAM gave {c_b_raw:.3f})")
results["x_e_raw"] = x_e_raw
results["c_beta_raw"] = c_b_raw

# ═══ 3. Frozen-geometry error: re-relax with the λ* potential ═════════════════
p_star = scaled_cross_potential("CuAg.eam.alloy", lam_s, out_dir=SCRATCH)
fac_s = eam_alloy_factory(p_star, ("Cu", "Ag"))
errs = []
for i, x in enumerate(XS):
    rel, _ = relax_full(fcc_solution("Ag", "Cu", float(x), reps=REPS, seed=0),
                        fac_s)
    dH_true = (float(rel.get_potential_energy()) / len(rel)
               - (1 - x) * E_pure["Ag"] - x * E_pure["Cu"])
    dH_lin = float(E0x[i] + lam_s * (E1x[i] - E0x[i]) - ref[i])
    errs.append(abs(dH_true - dH_lin))
print(f"\n── linear-bridge validation: re-relaxed ΔH_mix at λ* differs from "
      f"the frozen-geometry model by max {max(errs)*1e3:.2f} meV/atom "
      f"(scale: ΔH ≈ 90 meV)")

# ═══ 4. Full diagram at θ* for the record ═════════════════════════════════════
G_s_star = redlich_kister_solution(jnp.array(np.asarray(L_star)))
G_L_star = liquid_solution(omL_s, dH_fus_A=DH_AG, T_m_A=T_M_AG,
                           dH_fus_B=DH_CU, T_m_B=T_M_CU)
diag = eutectic_diagram(G_s_star, G_L_star, T_m_A=T_M_AG, T_m_B=T_M_CU,
                        n_T=40)

results.update({
    "lambda_star": lam_s, "omega_L_star_eV": omL_s,
    "L_RK_star_eV": np.asarray(L_star).tolist(),
    "omega_s_star_L0_meV": om_star * 1e3,
    "T_e_K": float(f_fin[0]), "c_alpha": float(f_fin[1]),
    "x_e": float(cl_f), "c_beta": float(cb_f),
    "J_start": np.asarray(Jphys).tolist(),
    "frozen_geometry_err_meV_max": max(errs) * 1e3,
    "engine_calls_lambda_pairs": engine_calls["lambda_pair"],
    "diagram_at_theta_star": {k: (v.tolist() if hasattr(v, "tolist") else v)
                              for k, v in diag.items()},
})
with open(OUT / "inverse_capstone.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'inverse_capstone.json'}")
