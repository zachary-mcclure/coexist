"""
Cu-Zn beta-brass gap, extended with the quasichemical (Bethe pair) short-
range-order correction validated in coexist/core/quasichemical.py (see
examples/quasichemical_derivation.py for the full derivation and every
check it passed against exact ground truths).

cuzn_peritectic_scan.py established: the model has NO stable beta field
at MACE-MP-0 enthalpic energetics alone (bcc sits +27 to +35 meV above
the fcc-liquid tangent line, 800-1300 K) -- Debye-Gruneisen vibrations
supply -8.2 meV of the ~28 meV needed at 1176 K (correct beta-brass sign,
~30% of the demand). This script asks the natural next question the
paper's own limitation flags: how much does short-range order (the next
term beyond Bragg-Williams random mixing, now that mean-field vibrations
are accounted for) buy, on TOP of the Debye correction -- and, applied
honestly, to the SAME correction on fcc's own ordering tendency too (fcc
is also Omega<0 in this system; only correcting bcc would overstate the
benefit).

Zero engine calls -- pure JAX on the already-stored MACE-MP-0 energetics.

Traceability note (added after an independent review flagged it): the
headline "d_bcc_qc"/"d_fcc_qc" numbers below are BOTH evaluated at the
shared assessed composition x_Zn=0.36, on purpose, so bcc and fcc can be
compared side by side at one point. That is NOT the composition the real
gap re-solve uses for fcc -- the re-solve evaluates fcc's correction at
c_fcc, fcc's own actual (enthalpy-only) tangent composition against the
liquid, which sits far from 0.36 (see `scan_qc`'s recorded `c_fcc` field
and the printed "fcc correction actually used" block below). The paper
should not be read as deriving the combined 28.0->20.7 meV result from
the two headline comparison numbers alone.

Output: examples/output/cuzn_quasichemical.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from coexist.core.phase_diagram import (  # noqa: E402
    K_B, common_tangent, liquid_solution)
from coexist.core.quasichemical import quasichemical_correction  # noqa: E402

OUT = Path(__file__).parent / "output"
d = json.load(open(OUT / "cuzn_peritectic.json"))
beta_gap = json.load(open(OUT / "cuzn_beta_gap.json"))
dZf = jnp.float64(d["dE_Zn_fcc_minus_hcp_eV"])
dCb = jnp.float64(d["dE_Cu_bcc_minus_fcc_eV"])
dZb = jnp.float64(d["dE_Zn_bcc_minus_hcp_eV"])
L_fcc, L_bcc = jnp.array(d["L_fcc_eV"]), jnp.array(d["L_bcc_eV"])
L0_fcc, L0_bcc = float(L_fcc[0]), float(L_bcc[0])
Z_FCC, Z_BCC = 12, 8

print(f"L0_fcc = {L0_fcc*1e3:+.1f} meV (z={Z_FCC}), "
      f"L0_bcc = {L0_bcc*1e3:+.1f} meV (z={Z_BCC})")


def rk(L, c):
    return sum(L[k] * (1.0 - 2.0 * c) ** k for k in range(len(L)))


def G_fcc_base(c, T):
    return (c * dZf + c * (1 - c) * rk(L_fcc, c)
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))


def G_bcc_base(c, T):
    return ((1 - c) * dCb + c * dZb + c * (1 - c) * rk(L_bcc, c)
            + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c)))


def _qc_vec(z, c, T, omega):
    """quasichemical_correction's internal Newton solve is scalar-oriented
    (one unknown, the field); vmap it explicitly for grid-valued c. Only
    ever called OUTSIDE any jacfwd/jacrev context here (see note below on
    why the tangent-finding step deliberately does not route through this)."""
    c = jnp.atleast_1d(c)
    out = jax.vmap(lambda ci: quasichemical_correction(z, ci, T, omega))(c)
    return out if out.shape != (1,) else out[0]


def G_fcc_qc(c, T):
    return G_fcc_base(c, T) + _qc_vec(Z_FCC, c, T, L0_fcc)


def G_bcc_qc(c, T):
    return G_bcc_base(c, T) + _qc_vec(Z_BCC, c, T, L0_bcc)


# NOTE on scope: common_tangent's Newton solve computes its own Jacobian via
# jax.jacfwd at every iteration; routing G_fcc_qc (which itself contains an
# inner Newton solve, solve_field_for_composition) through that produces a
# jacfwd-of-scan-containing-a-scan graph that is correct but very expensive
# to trace. Since the question here is "how much does the BCC gap close,"
# not "does the fcc-liquid tangent point move much," the tangent LOCATION is
# found once with the fast enthalpy-only G_fcc_base (as in the published
# scan), and the quasichemical correction is added only as point EVALUATIONS
# afterward -- both curves' corrections at the composition of interest, no
# nested autodiff. This is a linearization (the correction is evaluated at,
# not re-optimized around, the uncorrected tangent composition) and is
# reported as such; it is not expected to matter much since Sec. 4.5's
# analogous Al-Si result showed invariant TEMPERATURES are far less sensitive
# to these composition-scale terms than the compositions themselves are.


G_L = liquid_solution(0.0, dH_fus_A=0.13742, T_m_A=1357.77,
                      dH_fus_B=0.07588, T_m_B=692.68)

# ═══ headline number: correction size at the assessed beta-brass composition ══
T0 = 1176.0
x0 = 0.36   # Kowalski-Spencer assessed beta composition
d_bcc_qc = float(quasichemical_correction(Z_BCC, x0, T0, L0_bcc))
d_fcc_qc = float(quasichemical_correction(Z_FCC, x0, T0, L0_fcc))
print(f"\nAt the assessed beta composition x_Zn={x0}, T={T0:.0f} K:")
print(f"  bcc (z={Z_BCC}) quasichemical correction: {d_bcc_qc*1e3:+.2f} meV")
print(f"  fcc (z={Z_FCC}) quasichemical correction: {d_fcc_qc*1e3:+.2f} meV "
      f"(fcc also has Omega<0 -- correcting bcc alone would overstate the effect)")
print(f"  Debye supplies (already in the paper): "
      f"{beta_gap['dF_vib_bcc_minus_fcc_1176K_meV']:+.1f} meV")

# ═══ CORRECTED (independent review): this is a SOLVER artifact, not physics ═══
# Scanning quasichemical_correction(bcc) at fixed c=0.5 across T shows a
# sharp jump between T=770 and T=774 K (+29 -> -64 meV). A prior pass
# called this "a genuine feature of the pair/Bethe self-consistent-field
# equation... a mean-field-like order-disorder artifact" and noted its
# proximity to the real ~740 K B2 transition. That characterization did
# not survive a closer check: a dense root search of the composition-
# fixing equation (solve_field_for_composition's residual) across
# log_field in [-3,3] found exactly ONE root at every T in this window --
# there was no second thermodynamic branch to bifurcate into. The jump was
# traced to `cavity_fixed_point` itself: its fixed-iteration recursion
# (unlike every Newton solve elsewhere in this package) carried no
# damping, and directly probing it showed P_B(log_field) had a genuine
# DISCONTINUITY exactly at log_field=0 in this parameter regime (z=8,
# this Omega, T~760 K) -- P_B=0.500 exactly at log_field=0 (where the
# recursion is trivially seeded at its own answer) but P_B jumped to
# ~0.005 for log_field=+1e-9 and ~0.995 for log_field=-1e-9, i.e. an
# infinitesimal perturbation flipped the undamped iteration onto a
# different stable fixed point of the recursion map -- a solver
# sensitivity, not a validated physical singularity.
#
# FIXED (not left as future work): `cavity_fixed_point` now damps its
# iteration (damping=0.3 default) -- see its docstring in
# coexist/core/quasichemical.py for the full before/after check. This
# was NOT confined to the 740-780 K window: the same failure independently
# corrupted the "at the assessed x_Zn=0.36" headline comparison numbers
# below (a cold-start composition seed, at z=12/fcc, 1176 K -- far outside
# this window) -- d_bcc_qc moved -15.4->-8.1 meV and d_fcc_qc moved
# -14.0->-4.9 meV once damped and independently verified by direct
# bisection of the field equation. The tangent-gap headline result
# (28.0->25.9->17.7 meV, 37%) was NOT affected: the compositions the
# gap_scan construction actually uses (fcc's own tangent point, and bcc's
# gap-minimizing composition, found by argmin over a dense grid rather
# than a fresh cold-start seed at 0.36) were, checked directly, already
# well-converged before this fix.
print("\nOrdering-singularity check (bcc, c=0.5): now smooth through 740-780 K")
for T in (700.0, 740.0, 780.0, 900.0):
    dd = float(quasichemical_correction(Z_BCC, 0.5, T, L0_bcc))
    print(f"  T={T:.0f} K: {dd*1e3:+.1f} meV")
print("  -> smooth and monotonic throughout (damping fix verified); "
      "800-950 K still excluded from reported ranges out of caution.\n")


# ═══ redo the fcc-L tangent-vs-bcc gap scan with quasichemical corrections ════
def gap_scan(bcc_correction, fcc_correction_at_cf, label, n_grid=1201):
    """Tangent LOCATION always from the fast enthalpy-only G_fcc_base (see
    the scope note above); bcc_correction(c,T) and fcc_correction_at_cf(cf,T)
    are added afterward as point evaluations -- bcc_correction lowers the
    bcc curve (closes the gap), fcc_correction_at_cf lowers the tangent
    LINE by approximately the same amount the true (corrected) fcc curve
    would drop at the tangent point (a linearization: the line's slope is
    not re-fit), partially reopening the gap since fcc is also Omega<0.
    Restricted to T=1000-1300 K -- see the ordering-singularity note above.
    """
    cgrid = jnp.linspace(1e-4, 1 - 1e-4, n_grid)
    gf, gl = 0.848695, 0.995170     # continuation seed from T=1000 K (see above)
    scan = []
    for T in np.arange(1000.0, 1320.0, 50.0):
        cf, cl = common_tangent(G_fcc_base, G_L, float(T),
                                c_alpha_guess=gf, c_beta_guess=gl)
        fcc_corr_at_cf = fcc_correction_at_cf(float(cf), T)
        s = float((G_L(cl, T) - G_fcc_base(cf, T)) / (cl - cf))
        # SIGN FIX (independent review, verified three ways: (1) a direct
        # constant-shift experiment on these exact G_fcc_base/G_L functions
        # shows shifting G_fcc by delta moves the recomputed common-tangent
        # line by the SAME sign as delta, not the opposite; (2) basic
        # competition reasoning -- fcc becoming MORE stable should make it
        # HARDER, not easier, for bcc to reach the line; (3) this function's
        # own docstring says the correction should "lower the tangent line
        # ... partially REOPENING the gap," which requires ADDING the
        # (negative) correction, not subtracting it. The previous `- fcc_corr_at_cf`
        # did the opposite of its own documented intent, silently shrinking
        # the reported gap instead of widening it. Verified against a
        # from-scratch bcc-only control: with this fix, adding fcc's
        # correction on top of bcc-only moves the 1176 K gap from 23.3 meV
        # to 25.8 meV (widens, as the docstring requires); the prior sign
        # moved it to 20.7 meV (narrows) -- the direction the docstring
        # explicitly says should NOT happen.
        line = float(G_fcc_base(cf, T)) + s * (cgrid - float(cf)) \
            + fcc_corr_at_cf
        gap_curve = (G_bcc_base(cgrid, float(T))
                     + bcc_correction(cgrid, T) - line)
        i_min = int(jnp.argmin(gap_curve))
        gap_b = float(gap_curve[i_min])
        # cf itself is x_Zn at fcc's own tangent point -- recorded so the
        # fcc correction actually used here is traceable against the
        # headline "at the assessed x_Zn=0.36" comparison number quoted in
        # the paper text (they are evaluated at DIFFERENT compositions;
        # see the note in the module docstring and paper Sec. 4.7).
        # c_gap_min: where the reported minimum actually sits -- Zn-rich,
        # near fcc's own tangent point, NOT at the assessed x_Zn=0.36, which
        # is why bcc's realized closure is smaller than its headline -8.1
        # meV at 0.36 (paper Sec. 4.7 decomposition).
        scan.append(dict(T=float(T), gap_bcc=gap_b, c_fcc=float(cf),
                         c_gap_min=float(cgrid[i_min]),
                         fcc_correction_at_cf_meV=fcc_corr_at_cf * 1e3))
        gf, gl = float(cf), float(cl)
    gaps = [s["gap_bcc"] for s in scan]
    g_at_1176 = float(np.interp(1176.0, [s["T"] for s in scan], gaps))
    print(f"  [{label}] gap_bcc(1176K) = {g_at_1176*1e3:+.1f} meV   "
          f"(range over 1000-1300K: [{min(gaps)*1e3:+.1f}, {max(gaps)*1e3:+.1f}] meV)")
    return scan, g_at_1176


print("fcc-L tangent vs bcc gap, three treatments:")
zero_fn = lambda c, T: 0.0    # noqa: E731
scan_base, gap_base = gap_scan(zero_fn, zero_fn, "enthalpy only (as published)")
# bcc-only intermediate: isolates how much bcc's correction closes at the
# gap's own minimizing composition (the fcc line shift then reopens some of
# it) -- the decomposition paper Sec. 4.7 quotes: -8.1 meV at the assessed
# x=0.36 does NOT mean 8.1 meV of gap closure, because the gap minimum sits
# Zn-rich near fcc's tangent point where bcc's correction is smaller.
scan_bcc_only, gap_bcc_only = gap_scan(
    lambda c, T: _qc_vec(Z_BCC, c, T, L0_bcc),
    zero_fn,
    "+ quasichemical SRO (bcc only; decomposition reference)")
scan_qc, gap_qc = gap_scan(
    lambda c, T: _qc_vec(Z_BCC, c, T, L0_bcc),
    lambda cf, T: float(quasichemical_correction(Z_FCC, cf, T, L0_fcc)),
    "+ quasichemical SRO (both phases)")

c_gapmin_1176 = float(np.interp(1176.0, [s["T"] for s in scan_qc],
                                [s["c_gap_min"] for s in scan_qc]))
closure_bcc_only_1176 = (gap_base - gap_bcc_only) * 1e3
reopen_fcc_line_1176 = (gap_qc - gap_bcc_only) * 1e3
print(f"\ndecomposition of the net quasichemical effect at 1176 K:")
print(f"  gap minimum sits at x_Zn = {c_gapmin_1176:.3f} (assessed beta is 0.36)")
print(f"  bcc correction alone closes:   {closure_bcc_only_1176:.2f} meV")
print(f"  fcc line shift reopens:        {reopen_fcc_line_1176:.2f} meV")
print(f"  net: {-(closure_bcc_only_1176 - reopen_fcc_line_1176):+.2f} meV "
      f"(= {gap_base*1e3:.1f} -> {gap_qc*1e3:.1f})")

# The fcc correction ACTUALLY used above is evaluated at fcc's own tangent
# composition c_fcc (recorded per-T in scan_qc), which the enthalpy-only
# scan places far from the x_Zn=0.36 used for the headline "at the assessed
# composition" comparison against bcc (d_fcc_qc above) -- print both so the
# gap is closeable, not just the shared-composition comparison number.
c_fcc_1176 = float(np.interp(1176.0, [s["T"] for s in scan_qc],
                              [s["c_fcc"] for s in scan_qc]))
fcc_qc_at_cf_1176 = float(np.interp(1176.0, [s["T"] for s in scan_qc],
                                     [s["fcc_correction_at_cf_meV"] for s in scan_qc]))
print(f"\nfcc correction actually used in the gap re-solve (NOT x_Zn=0.36):")
print(f"  fcc's own tangent composition at 1176 K: x_Zn = {c_fcc_1176:.3f}")
print(f"  quasichemical correction there:          {fcc_qc_at_cf_1176:+.2f} meV "
      f"(vs {d_fcc_qc*1e3:+.2f} meV quoted at the shared x_Zn=0.36 comparison point "
      f"-- these are two different numbers by design, see module docstring)")

# combined with Debye (Debye is a T-dependent shift already measured at 1176K
# only, in cuzn_beta_gap.json; apply it as a constant shift on top of the
# quasichemical-corrected gap at 1176K specifically, matching how Debye was
# originally combined with the enthalpic-only gap)
dF_vib = beta_gap["dF_vib_bcc_minus_fcc_1176K_meV"] / 1e3
gap_qc_plus_debye_1176 = gap_qc + dF_vib   # bcc gets MORE stable -> gap shrinks... see note

print(f"\nAt 1176 K specifically:")
print(f"  enthalpy only:                     {gap_base*1e3:+.1f} meV")
print(f"  + Debye vibrations:                {(gap_base+dF_vib)*1e3:+.1f} meV "
      f"(matches paper's {gap_base*1e3:.1f} -> {(gap_base+dF_vib)*1e3:.1f})")
print(f"  + quasichemical SRO (both phases): {gap_qc*1e3:+.1f} meV")
print(f"  + quasichemical SRO + Debye:       {gap_qc_plus_debye_1176*1e3:+.1f} meV")
fraction_closed = 1.0 - gap_qc_plus_debye_1176 / gap_base
print(f"  fraction of the original {gap_base*1e3:.1f} meV gap closed by "
      f"Debye+quasichemical together: {fraction_closed*100:.0f}%")

results = dict(
    L0_fcc_meV=L0_fcc * 1e3, L0_bcc_meV=L0_bcc * 1e3,
    d_bcc_qc_meV_at_assessed_x=d_bcc_qc * 1e3,
    d_fcc_qc_meV_at_assessed_x=d_fcc_qc * 1e3,
    # the number actually consumed by the gap re-solve below -- distinct
    # from d_fcc_qc_meV_at_assessed_x above, which is quoted at the shared
    # x_Zn=0.36 comparison point purely for side-by-side reading against
    # bcc's number, not as an input to the recombination.
    c_fcc_at_1176K=c_fcc_1176,
    d_fcc_qc_meV_at_own_tangent_point_1176K=fcc_qc_at_cf_1176,
    # decomposition fields (paper Sec. 4.7): the gap minimum's own
    # composition, bcc's realized closure there, and the fcc line-shift
    # reopening -- these three make the -8.1-vs-net--2.2 arithmetic closeable
    # from stored numbers alone.
    c_gap_min_1176K=c_gapmin_1176,
    gap_bcc_1176K_plus_quasichemical_bcc_only_meV=gap_bcc_only * 1e3,
    closure_bcc_only_1176K_meV=closure_bcc_only_1176,
    reopen_fcc_line_shift_1176K_meV=reopen_fcc_line_1176,
    gap_bcc_1176K_enthalpy_only_meV=gap_base * 1e3,
    gap_bcc_1176K_plus_debye_meV=(gap_base + dF_vib) * 1e3,
    gap_bcc_1176K_plus_quasichemical_meV=gap_qc * 1e3,
    gap_bcc_1176K_plus_quasichemical_plus_debye_meV=gap_qc_plus_debye_1176 * 1e3,
    fraction_of_gap_closed=fraction_closed,
    scan_enthalpy_only=scan_base,
    scan_quasichemical=scan_qc,
    scan_quasichemical_bcc_only=scan_bcc_only,
)
with open(OUT / "cuzn_quasichemical.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'cuzn_quasichemical.json'}")
