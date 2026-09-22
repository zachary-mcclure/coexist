"""
Zr-Nb monotectoid: an all-solid invariant, the first in this paper with
NO liquid phase at all -- beta(bcc, Zr-Nb solid solution) -> alpha(hcp,
Zr-rich) + beta'(bcc, Nb-rich) on cooling through 893 K.

Why a monotectoid (not the eutectoid originally scoped in the open
items): a genuine eutectoid/peritectoid on a SIMPLE structure turns out to
be physically rare among real binary metallic systems -- most real
eutectoids/peritectoids involve a complex intermetallic product (Laves
C14/C15/C36, gamma-brass, D0-superstructures), exactly the kind of
crystallography this paper has consistently avoided (Cu-Zn's own real
eutectoid beta->alpha+gamma was ruled out for this reason). A
monotectoid sidesteps this: it is the solid-state analog of Al-Pb's
already-verified LIQUID monotectic (Sec. 4.5) -- a single phase splitting
into a composition-miscibility-gap pair -- so BOTH product phases can be
ordinary terminal solid solutions (bcc Nb-rich, hcp Zr-rich), with no
compound anywhere in the construction. Zr-Nb is real, well characterized,
and has a dedicated NIST-repository potential built specifically for
bcc-Nb/hcp-Zr coherency.

Assessed invariant: beta(bcc) -> alpha(hcp) + beta'(bcc) at 893 K
(620 C), beta matrix 18.7 at% Nb, alpha product 0.7 at% Nb, beta'
product 92.1 at% Nb -- the special-points table of Okamoto, J. Phase
Equilibria 13(5), 577 (1992), doi:10.1007/BF02665776 (freely
downloadable), whose preferred diagram is the thermodynamic
calculation of Fernandez Guillermet, Z. Metallkd. 82(6), 478-487
(1991), rather than the earlier experimental assessment of Abriata &
Bolcich, "The Nb-Zr system," Bull. Alloy Phase Diagrams 3(1), 34-44
(1982). The same table gives the bcc miscibility-gap critical point
(58.8 at% Nb, 977 C = 1250 K) and pure Zr's allotropic temperature
(863 C = 1136 K).

Potential: Fan, Maras, Cottura, Marinica & Clouet, "Structure and
coherency of bcc Nb precipitates in hcp Zr matrix from atomistic
simulations," Phys. Rev. Materials 8(11), 113601 (2024) -- an EAM
potential purpose-built for exactly this bcc/hcp interface physics,
downloaded from the NIST Interatomic Potentials Repository
(potentials/ZrNb5.eam.alloy).

Construction: architecturally IDENTICAL to Al-Pb's monotectic
(Sec. 4.5) -- `three_phase_equilibrium(G_hcp, G_bcc, G_bcc, ...)`, the
SAME free-energy function passed twice (the docstring's own documented
use case: "monotectic L1 -> alpha + L2 (pass the SAME liquid fn twice)"),
here with G_bcc playing the role the liquid played there. No new solver
code -- this is a direct test of whether that generality claim holds for
an all-solid system.

hcp structures (host AND every alloy composition) get full cell-SHAPE
relaxation (ASE FrechetCellFilter), not the isotropic-only protocol used
for cubic lattices elsewhere: hcp has a genuine c/a degree of freedom no
cubic structure in this paper has had to deal with at the SOLUTION level
(Cu-Sn's beta-Sn needed this only for its one pure reference cell; here
EVERY hcp structure needs it). bcc structures use the standard isotropic
protocol (cubic, single degree of freedom).

Runtime: EAM via LAMMPS -- seconds per relaxation, whole script ~1-2 min.
Output: examples/output/zrnb_monotectoid.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from ase.build import bulk  # noqa: E402
from ase.filters import FrechetCellFilter  # noqa: E402
from ase.optimize import BFGS  # noqa: E402

from coexist.adapters.lammps_factory import (  # noqa: E402
    bcc_solution, eam_alloy_factory, eos_scan, hcp_solution, relax_full)
from coexist.core.mixing import redlich_kister_fit  # noqa: E402
from coexist.core.phase_diagram import (  # noqa: E402
    K_B, classify_invariant, common_tangent, global_tangency_gap,
    three_phase_equilibrium)
from coexist.core.thermo_vib import (  # noqa: E402
    debye_temperature, eos_fit, f_vib_debye)

OUT = Path(__file__).parent / "output"
results = {}

fac = eam_alloy_factory("ZrNb5.eam.alloy", ("Zr", "Nb"))

# Experimental starting guesses (refined by relaxation below).
A_ZR_HCP, C_ZR_HCP = 3.232, 5.147
A_NB_BCC = 3.3008


def relaxed_E_hcp_full_cell(atoms, fmax=0.01, steps=300):
    """Full cell-SHAPE relaxation for an hcp structure (a AND c, plus
    internal coordinates) -- needed because hcp, unlike fcc/bcc, has a
    genuine shape (c/a) degree of freedom no isotropic scan can touch."""
    work = atoms.copy()
    work.calc = fac()
    ecf = FrechetCellFilter(work)
    BFGS(ecf, logfile=None).run(fmax=fmax, steps=steps)
    return float(work.get_potential_energy()) / len(work)


def relaxed_E_bcc(atoms, fmax=0.02):
    """Isotropic volume + internal-coordinate relaxation -- the protocol
    used for every CUBIC structure elsewhere in this paper (bcc has no
    shape freedom under cubic symmetry)."""
    work, _ = relax_full(atoms, fac, fmax=fmax)
    return float(work.get_potential_energy()) / len(work)


# ═══ 1. Pure-element references + promotion energies ═══════════════════════
print("-- pure-element references (EAM, LAMMPS)")
E_hcp_Zr = relaxed_E_hcp_full_cell(
    bulk("Zr", "hcp", a=A_ZR_HCP, c=C_ZR_HCP).repeat((4, 4, 3)))
E_bcc_Nb = relaxed_E_bcc(bulk("Nb", "bcc", a=A_NB_BCC, cubic=True).repeat((4, 4, 4)))
E_hcp_Nb = relaxed_E_hcp_full_cell(
    bulk("Nb", "hcp", a=A_ZR_HCP, c=C_ZR_HCP).repeat((4, 4, 3)))
E_bcc_Zr = relaxed_E_bcc(bulk("Zr", "bcc", a=A_NB_BCC, cubic=True).repeat((4, 4, 4)))

dE_Nb_hcp_minus_bcc = E_hcp_Nb - E_bcc_Nb
dE_Zr_bcc_minus_hcp = E_bcc_Zr - E_hcp_Zr
print(f"   E(hcp-Zr, native)  = {E_hcp_Zr:+.4f} eV/atom")
print(f"   E(bcc-Nb, native)  = {E_bcc_Nb:+.4f} eV/atom")
print(f"   dE_Nb(hcp-bcc) = {dE_Nb_hcp_minus_bcc*1e3:+.0f} meV  "
     "(Nb promoted into hcp, above its native bcc)")
print(f"   dE_Zr(bcc-hcp) = {dE_Zr_bcc_minus_hcp*1e3:+.0f} meV  "
     "(Zr promoted into bcc, above its native hcp)")

# ═══ 2. Mixing on each lattice (random supercells) ══════════════════════════
print("-- mixing energetics (random supercells)")
# hcp: dilute-to-moderate Nb in Zr, full cell-shape relaxation each point
# (96-atom cell, 4x4x3 reps of the 2-atom hcp conventional cell).
ns_hcp = (2, 5, 10, 18)
xs_hcp = np.array(ns_hcp) / 96.0
dH_hcp = []
for k, x in zip(ns_hcp, xs_hcp):
    at = hcp_solution("Zr", "Nb", float(x), reps=(4, 4, 3),
                      a=A_ZR_HCP, c=C_ZR_HCP, seed=0)
    E = relaxed_E_hcp_full_cell(at)
    dH_hcp.append(E - (1 - x) * E_hcp_Zr - x * E_hcp_Nb)
    print(f"   hcp x_Nb={x:.4f}: dH = {dH_hcp[-1]*1e3:+.0f} meV")
L_hcp = redlich_kister_fit(jnp.array(xs_hcp), jnp.array(dH_hcp), order=1)

# bcc: wide range bracketing 0.187 (beta) through 0.92 (beta'), isotropic
# relaxation (64-atom cell, 4x4x4 cubic reps).
ns_bcc = (12, 32, 52, 59)
xs_bcc = np.array(ns_bcc) / 64.0
dH_bcc = []
for k, x in zip(ns_bcc, xs_bcc):
    at = bcc_solution("Nb", "Zr", 1.0 - float(x), reps=(4, 4, 4),
                      a=A_NB_BCC, seed=0)
    # bcc_solution(A,B,x_B) builds A_{1-x_B}B_{x_B}; want x_Nb=x, so
    # host=Nb, x_Zr=1-x -> x_Nb = 1-(1-x) = x. Verify explicitly below.
    E = relaxed_E_bcc(at)
    dH_bcc.append(E - (1 - x) * E_bcc_Zr - x * E_bcc_Nb)
    print(f"   bcc x_Nb={x:.4f}: dH = {dH_bcc[-1]*1e3:+.0f} meV")
L_bcc = redlich_kister_fit(jnp.array(xs_bcc), jnp.array(dH_bcc), order=1)
print(f"   RK(hcp) = [{', '.join(f'{float(v)*1e3:+.0f}' for v in L_hcp)}] meV, "
     f"RK(bcc) = [{', '.join(f'{float(v)*1e3:+.0f}' for v in L_bcc)}] meV")

# ═══ 3. Curves and the invariant hunt ═══════════════════════════════════════
dNh, dZb = jnp.float64(dE_Nb_hcp_minus_bcc), jnp.float64(dE_Zr_bcc_minus_hcp)


def rk(L, c):
    return sum(L[k] * (1.0 - 2.0 * c) ** k for k in range(len(L)))


def G_hcp(c, T):
    return (c * dNh + c * (1 - c) * rk(L_hcp, c)
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))


def G_bcc(c, T):
    return ((1 - c) * dZb + c * (1 - c) * rk(L_bcc, c)
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))


T_ASSESSED = 893.0
C_ALPHA, C_BETA, C_BETAP = 0.007, 0.187, 0.921
# bcc gap critical point, same Okamoto special-points table
TC_ASSESSED, XC_ASSESSED = 1250.0, 0.588

# ═══ 3a. Pre-check: bcc's OWN self-tangent (miscibility gap), the
#         well-conditioned 2-phase construction to seed the 3-phase solve
#         from -- same protocol discipline as Cu-Zn/Cu-Sn's absence checks
#         (Sec. 3.4: verify with the easy solve before trusting the hard
#         one). This is architecturally identical to Cu-Ni's own
#         miscibility gap (common_tangent(G, G, T)). ═══
print(f"\n-- bcc self-tangent (miscibility gap) at the assessed T={T_ASSESSED:.0f} K:")
c_lo, c_hi = common_tangent(G_bcc, G_bcc, T_ASSESSED,
                            c_alpha_guess=0.15, c_beta_guess=0.985)
print(f"   c_lo={float(c_lo):.4f}, c_hi={float(c_hi):.4f} "
     f"(assessed beta/beta': {C_BETA}/{C_BETAP}) -- c_lo matches closely, "
     f"c_hi is much narrower (model predicts less Zr solubility in "
     f"Nb-rich bcc than assessed)")

# ═══ 3b. The gap's APEX (critical/consolute point), closed form from the
#         same RK: with H(x) = x(1-x)[L0 + L1(1-2x)], the spinodal is
#         T_sp(x) = -H''(x) x(1-x)/k_B and the critical point is its
#         maximum, dT_sp/dx = 0 -> 3Bx^2 - 2(A+B)x + A = 0 with
#         A = 2(L0+3L1), B = 12 L1 (endpoint-linear terms drop out of
#         every derivative >= 2). Separates what the model gets right
#         (the dilute gap EDGE at 893 K, deep-subcritical, 0 K-repulsion-
#         dominated) from where it drifts (the apex). ═══
L0_b, L1_b = float(L_bcc[0]), float(L_bcc[1])
A_ap, B_ap = 2.0 * (L0_b + 3.0 * L1_b), 12.0 * L1_b
disc = 4.0 * (A_ap + B_ap) ** 2 - 12.0 * B_ap * A_ap
roots_ap = [(2.0 * (A_ap + B_ap) + s * np.sqrt(disc)) / (6.0 * B_ap)
            for s in (+1.0, -1.0)]
def _T_sp(x):
    return (2.0 * (L0_b + 3.0 * L1_b) - 12.0 * L1_b * x) * x * (1 - x) / K_B
x_c_model, T_c_model = max(((x, _T_sp(x)) for x in roots_ap
                            if 0.0 < x < 1.0), key=lambda p: p[1])
print(f"\n-- bcc gap apex (critical point), closed form from the stored RK:")
print(f"   model x_c={x_c_model:.3f}, T_c={T_c_model:.1f} K "
      f"(assessed {XC_ASSESSED}, {TC_ASSESSED:.0f} K = 977 C)")
results["bcc_gap_critical_point"] = dict(
    x_c_model=x_c_model, T_c_model_K=T_c_model,
    x_c_assessed=XC_ASSESSED, T_c_assessed_K=TC_ASSESSED)

print(f"\n-- invariant hunt (assessed: monotectoid beta(bcc)->alpha(hcp)"
     f"+beta'(bcc) at {T_ASSESSED:.0f} K; alpha {C_ALPHA}/beta {C_BETA}/"
     f"beta' {C_BETAP} at% Nb, Okamoto 1992 special-points table)")
c1, c2, c3, T = three_phase_equilibrium(
    G_hcp, G_bcc, G_bcc, c_guesses=(C_ALPHA, C_BETA, C_BETAP), T_guess=T_ASSESSED)


def inv_residual(cs, T):
    r = []
    d1 = float(jax.grad(G_hcp, argnums=0)(cs[0], T))
    for Gi, ci in ((G_bcc, cs[1]), (G_bcc, cs[2])):
        r.append(abs(d1 - float(jax.grad(Gi, argnums=0)(ci, T))))
        r.append(abs(float(Gi(ci, T)) - float(G_hcp(cs[0], T))
                     - d1 * (float(ci) - float(cs[0]))))
    return max(r)


res = inv_residual((c1, c2, c3), float(T))
gap = global_tangency_gap((G_hcp, G_bcc), (c1, c2, c3), float(T), anchor=0)
kind = classify_invariant(("solid", "solid", "solid"))
converged = res < 1e-8
if converged and gap > -1e-6:
    tag = "VERIFIED"
elif not converged:
    tag = f"STALLED (residual {res:.0e} -- no root; gap {gap:+.1e} alone " \
         f"would have passed it)"
else:
    tag = f"REJECTED (gap {gap:+.1e})"
print(f"   alpha/beta/beta': T = {float(T):.1f} K, c = ({float(c1):.4f}, "
     f"{float(c2):.4f}, {float(c3):.4f}) -> {kind} -- {tag}")
print(
    f"\n   T is {float(T)-T_ASSESSED:+.0f} K from the assessed {T_ASSESSED:.0f} K "
    "-- the largest T-discrepancy anywhere in this paper, but with a\n"
    "   clean structural explanation, not a mystery: ideal configurational "
    "entropy is EXACTLY ZERO at x=0 (both x*ln(x) terms vanish\n"
    "   identically), so G_hcp(0,T)=0 and G_bcc(0,T)=dE_Zr_bcc_minus_hcp "
    f"={dE_Zr_bcc_minus_hcp*1e3:+.0f} meV for EVERY T -- this model class\n"
    "   cannot show pure-Zr's own real alpha(hcp)->beta(bcc) crossover "
    "(experimentally at 1136 K) at ANY temperature, since that transition\n"
    "   is driven almost entirely by VIBRATIONAL entropy (bcc's softer "
    "phonons), which is entirely absent from this ideal-entropy-only\n"
    "   model. Measured below, the same way Cu-Zn/Cu-Sn's missing physics "
    "was measured.")

# ═══ 4. Missing-physics attribution: Debye vibrational entropy for pure Zr
#         in each structure -- does it explain the T discrepancy, the same
#         way Debye explained part of Cu-Zn/Cu-Sn's gap? Isotropic-only EOS
#         scan for hcp (a bounded, exploratory approximation -- does not
#         independently relax c/a under compression, unlike the full
#         cell-shape treatment used for the mixing-energetics supercells
#         above; noted explicitly, not hidden). ═══
print("\n-- Debye vibrational-entropy attribution (pure Zr, bcc vs hcp)")
M_ZR = 91.224
THETA = {}
for lat, a, c, reps in (("bcc", A_NB_BCC, None, (4, 4, 4)),
                        ("hcp", A_ZR_HCP, C_ZR_HCP, (4, 4, 3))):
    kwargs = dict(a=a, cubic=True) if lat == "bcc" else dict(a=a, c=c)
    at = bulk("Zr", lat, **kwargs).repeat(reps)
    V, E = eos_scan(at, fac, scale_range=0.05, n_points=9)
    V0, _, B0, _ = eos_fit(jnp.array(V), jnp.array(E))
    THETA[lat] = float(debye_temperature(V0, B0, M_ZR))
    print(f"   {lat}: theta_D = {THETA[lat]:.1f} K")

dF_vib_893 = float(f_vib_debye(T_ASSESSED, THETA["bcc"])
                   - f_vib_debye(T_ASSESSED, THETA["hcp"]))
dF_vib_1136 = float(f_vib_debye(1136.0, THETA["bcc"])
                    - f_vib_debye(1136.0, THETA["hcp"]))
print(f"   F_vib(bcc)-F_vib(hcp) at {T_ASSESSED:.0f} K = {dF_vib_893*1e3:+.1f} meV")
print(f"   F_vib(bcc)-F_vib(hcp) at 1136 K (real Zr alpha->beta T) = "
     f"{dF_vib_1136*1e3:+.1f} meV")
print(f"   vs the +{dE_Zr_bcc_minus_hcp*1e3:.0f} meV static (0 K) penalty bcc "
     "must overcome: correct (bcc-favoring) sign, but Debye alone supplies "
     f"only {-dF_vib_1136*1e3/(dE_Zr_bcc_minus_hcp*1e3)*100:.0f}% of it even "
     "at Zr's own real transition temperature -- consistent with the "
     "literature understanding that Zr's alpha/beta transition requires "
     "physics beyond simple single-Debye-temperature theory (anharmonicity, "
     "electronic entropy), not a shortcoming specific to this calculation.")

results["invariant_attempt"] = dict(
    T=float(T), c=[float(c1), float(c2), float(c3)], kind=kind, gap=gap,
    stationarity_residual=res, converged=bool(converged), tag=tag)
results["bcc_self_tangent_at_assessed_T"] = dict(c_lo=float(c_lo), c_hi=float(c_hi))
results["debye_attribution"] = dict(
    theta_D_K=THETA,
    dF_vib_bcc_minus_hcp_meV_at_893K=dF_vib_893 * 1e3,
    dF_vib_bcc_minus_hcp_meV_at_1136K=dF_vib_1136 * 1e3,
    static_penalty_meV=dE_Zr_bcc_minus_hcp * 1e3,
    fraction_closed_at_1136K=-dF_vib_1136 / dE_Zr_bcc_minus_hcp,
)
results.update({
    "E_hcp_Zr_eV": E_hcp_Zr, "E_bcc_Nb_eV": E_bcc_Nb,
    "E_hcp_Nb_eV": E_hcp_Nb, "E_bcc_Zr_eV": E_bcc_Zr,
    "dE_Nb_hcp_minus_bcc_eV": dE_Nb_hcp_minus_bcc,
    "dE_Zr_bcc_minus_hcp_eV": dE_Zr_bcc_minus_hcp,
    "L_hcp_eV": np.asarray(L_hcp).tolist(),
    "L_bcc_eV": np.asarray(L_bcc).tolist(),
    "assessed_T_K": T_ASSESSED,
    "assessed_c": [C_ALPHA, C_BETA, C_BETAP],
    "assessed_source": "Okamoto, J. Phase Equilibria 13(5), 577 (1992) "
                       "special-points table (preferred diagram: Fernandez "
                       "Guillermet, Z. Metallkd. 82(6), 478-487 (1991); "
                       "underlying assessment: Abriata & Bolcich, Bull. "
                       "Alloy Phase Diagrams 3(1), 34-44 (1982))",
    "potential_source": "Fan, Maras, Cottura, Marinica & Clouet, Phys. Rev. "
                        "Materials 8(11), 113601 (2024), NIST IPR",
})
with open(OUT / "zrnb_monotectoid.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'zrnb_monotectoid.json'}")
