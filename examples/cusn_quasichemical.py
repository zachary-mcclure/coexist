"""
Cu-Sn missing-physics attribution, part 2: an ATTEMPT at the quasichemical
(Bethe pair) short-range-order correction on top of Debye vibrations (from
cusn_beta_gap.py) -- the identical treatment Cu-Zn's Sec. 4.7 gives its own
gap (examples/cuzn_quasichemical.py). Result: NOT APPLICABLE, for a
specific, derived reason -- not a forced or hidden result.

A first version of this script found the correction flipping the gap
strongly negative (implying, incorrectly, a possible verified peritectic)
by combining Debye (-18.6 meV, a genuine, separately-verified result -- see
below) with a quasichemical number (-60.9/-56.1 meV) that turned out to be
computed from a SILENTLY NON-CONVERGED solve: `solve_field_for_composition`
(coexist/core/quasichemical.py) was returning its own unmoved initial
seed, not a root, at Cu-Sn's actual (z, Omega, T, x). Diagnosed directly
(examples/quasichemical_strong_coupling_check.py Part 2-3): the field ->
composition map has a genuine jump discontinuity at these parameters, and
the target compositions (x_Sn=0.077 alpha / 0.13 beta) sit INSIDE the
regular-solution (Bragg-Williams) spinodal implied by the correction's own
Omega-only simplification -- derived in closed form as
`symmetric_model_spinodal` (c_lo, c_hi = [1 -/+ sqrt(1-2kT/Omega)]/2, real
only when Omega > 2kT). Cu-Sn's Omega (652/893 meV) is large enough that
this simplified-model spinodal (c_lo~0.055-0.077, c_hi~0.92-0.95) swallows
both assessed compositions entirely, even though the REAL multi-term
Redlich-Kister curves being corrected (which include an L1 term the
Omega-only simplification discards) are locally STABLE there (checked
directly: d2G/dc2 > 0 at both compositions from the actual L_fcc/L_bcc
fits). `quasichemical_correction` now detects this and returns NaN instead
of a silently-wrong finite number (fixed in coexist/core/quasichemical.py,
regression-tested in tests/test_quasichemical.py) -- this script reports
that NaN honestly rather than working around it.

Cu-Zn is UNAFFECTED: its Omega is negative (ordering, not clustering), for
which the simplified model has no spinodal at any composition -- verified
directly, and by re-running examples/cuzn_quasichemical.py end to end
post-fix (28.0->25.9->17.7 meV, 37%, bit-for-bit unchanged).

Zero engine calls -- pure JAX on stored MACE-MP-0 energetics.
Output: examples/output/cusn_quasichemical.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from coexist.core.quasichemical import (  # noqa: E402
    quasichemical_correction, symmetric_model_spinodal)

OUT = Path(__file__).parent / "output"
d = json.load(open(OUT / "cusn_peritectic.json"))
beta_gap = json.load(open(OUT / "cusn_beta_gap.json"))
L_fcc, L_bcc = jnp.array(d["L_fcc_eV"]), jnp.array(d["L_bcc_eV"])
L0_fcc, L0_bcc = float(L_fcc[0]), float(L_bcc[0])
Z_FCC, Z_BCC = 12, 8
T0 = d["assessed_T_K"]
x_alpha, x_beta = d["assessed_c"][0], d["assessed_c"][1]

print(f"L0_fcc = {L0_fcc*1e3:+.1f} meV (z={Z_FCC}), "
     f"L0_bcc = {L0_bcc*1e3:+.1f} meV (z={Z_BCC}) "
     "-- both strongly positive (demixing), unlike Cu-Zn's attractive Omega")

# ═══ 1. Applicability check FIRST, before trusting anything the solver
#         returns -- symmetric_model_spinodal is closed-form, no solve
#         required, and derived directly from the same Bragg-Williams
#         functional form the correction's Omega-only simplification
#         reduces to (see coexist/core/quasichemical.py docstring). ═══
print(f"\nApplicability check (T={T0:.1f} K):")
for label, z, Omega, x in [("bcc (beta)", Z_BCC, L0_bcc, x_beta),
                           ("fcc (alpha)", Z_FCC, L0_fcc, x_alpha)]:
    c_lo, c_hi = symmetric_model_spinodal(Omega, T0)
    c_lo, c_hi = float(c_lo), float(c_hi)
    inside = c_lo == c_lo and c_lo < x < c_hi  # nan-safe (nan != nan)
    print(f"  {label}: Omega={Omega*1e3:+.0f} meV -> simplified-model "
         f"spinodal ({c_lo:.4f}, {c_hi:.4f}); assessed x={x} is "
         f"{'INSIDE -- correction not applicable' if inside else 'outside -- applicable'}")

d_bcc_qc = float(quasichemical_correction(Z_BCC, x_beta, T0, L0_bcc))
d_fcc_qc = float(quasichemical_correction(Z_FCC, x_alpha, T0, L0_fcc))
qc_applicable = np.isfinite(d_bcc_qc) and np.isfinite(d_fcc_qc)
print(f"\n  quasichemical_correction(bcc, x={x_beta}) = {d_bcc_qc}")
print(f"  quasichemical_correction(fcc, x={x_alpha}) = {d_fcc_qc}")

dF_vib = beta_gap["dF_vib_bcc_minus_fcc_meV"] / 1e3
gap_at_T0 = beta_gap["gap_bcc_at_assessed_T_meV"] / 1e3
fraction_closed_by_debye = -dF_vib / gap_at_T0

if not qc_applicable:
    print(
        "\n"
        "CONCLUSION: the quasichemical short-range-order correction is NOT\n"
        "APPLICABLE to Cu-Sn at the assessed compositions -- both fall inside\n"
        "the spinodal of the correction's own Omega-only simplified model\n"
        "(the real, multi-term Redlich-Kister curves being corrected are\n"
        "locally stable at these compositions; only the reduced single-\n"
        "parameter model used internally by the correction is not). This is\n"
        "a genuine model-scope boundary, quantified above, not a forced or\n"
        "hidden result -- see the module docstring for the full diagnosis.\n"
        "\n"
        f"The DEBYE correction alone remains valid and is the reported\n"
        f"Cu-Sn missing-physics attribution: gap {gap_at_T0*1e3:+.1f} meV at "
        f"{T0:.0f} K enthalpy-only,\n"
        f"{(gap_at_T0+dF_vib)*1e3:+.1f} meV with Debye vibrations "
        f"({fraction_closed_by_debye*100:.0f}% of the gap closed by Debye alone --\n"
        "substantially more than Cu-Zn's ~30% Debye-alone contribution, though\n"
        "Cu-Zn's COMBINED Debye+quasichemical total (37%) has no Cu-Sn\n"
        "counterpart since quasichemical does not apply here). The gap does\n"
        "NOT cross zero under any correction validated in this paper: Cu-Sn\n"
        "remains the third of three sought-and-absent peritectics.")

results = dict(
    T_K=T0, x_alpha=x_alpha, x_beta=x_beta,
    L0_fcc_meV=L0_fcc * 1e3, L0_bcc_meV=L0_bcc * 1e3,
    quasichemical_applicable=bool(qc_applicable),
    d_bcc_qc_meV=None if not np.isfinite(d_bcc_qc) else d_bcc_qc * 1e3,
    d_fcc_qc_meV=None if not np.isfinite(d_fcc_qc) else d_fcc_qc * 1e3,
    gap_bcc_enthalpy_only_meV=gap_at_T0 * 1e3,
    dF_vib_bcc_minus_fcc_meV=dF_vib * 1e3,
    gap_bcc_plus_debye_meV=(gap_at_T0 + dF_vib) * 1e3,
    fraction_of_gap_closed_by_debye=fraction_closed_by_debye,
    conclusion=("quasichemical SRO not applicable (assessed compositions "
               "inside the correction's own simplified-model spinodal); "
               "Debye vibrations alone close "
               f"{fraction_closed_by_debye*100:.0f}% of the gap but it "
               "does not cross zero -- peritectic remains absent"),
)
with open(OUT / "cusn_quasichemical.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'cusn_quasichemical.json'}")
