"""
Quasichemical (Bethe pair-approximation) short-range-order correction to
Bragg-Williams (regular-solution) mixing.

Every solid-solution curve elsewhere in this package (`mixing.py`,
`phase_diagram.py`'s `regular_solution`/`redlich_kister_solution`) assumes
a RANDOM distribution of atoms on the lattice even though the mixing
enthalpy is not zero -- the regular-solution/Bragg-Williams approximation.
The quasichemical (Bethe pair, first-order cluster-variation) approximation
replaces this with a self-consistent PAIR distribution: atoms still sit on
a fixed lattice, but bonds are treated as (nearly) independent objects
whose type distribution is solved self-consistently from a coordination
number z and the same regular-solution interaction Ω, at essentially zero
extra engine cost (it is a post-processing correction to an already-fit Ω,
not a new engine campaign). It is EXACT for a Bethe lattice / Cayley tree
(no loops) and an approximation, that improves monotonically with z, for
real lattices (which have loops) -- both properties are derived and
checked, not assumed, in `examples/quasichemical_derivation.py`, which
this module distills into a permanent, reusable, differentiable form.

Derivation summary (see the derivation script for full validation):
  - cavity recursion for the bulk pair-probability fixed point, checked to
    machine precision against exact finite-tree dynamic programming and,
    at z=2, against the independently-exact 1D transfer matrix;
  - the free energy is obtained by integrating the (validated) internal
    energy from the exact T=infinity reference via the standard
    thermodynamic identity d(F/T)/d(1/T) = U at fixed fugacity field --
    NOT from a remembered closed-form entropy expression, one of which
    was tried, found to disagree with the exact z=2 result by several
    tens of meV, and discarded rather than patched from memory;
  - checked exactly reduces to the ideal/regular solution at Omega=0 for
    every z, and always has entropy <= the ideal-mixing entropy at
    matched composition, for every z tested (2 through 12);
  - the honest approximation error on real (loopy) lattices was measured
    against exact grand-canonical brute force at z=3,4,6,8 and found to
    shrink with z, licensing (not proving exact) its use at z=8 (bcc) and
    z=12 (fcc), where brute force is computationally out of reach.

All routines are fixed-iteration (`lax.scan`/quadrature), so `jax.grad`
through them is well-defined, matching the implicit-layer style used
throughout `phase_diagram.py`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from coexist.core.phase_diagram import _newton_solve

K_B = 8.617333e-5   # eV/K

__all__ = [
    "cavity_fixed_point", "bethe_marginals", "bethe_energy",
    "bethe_free_energy_grand", "solve_field_for_composition",
    "quasichemical_correction", "symmetric_model_spinodal",
]

# fixed Gauss-Legendre quadrature for the free-energy integral (same
# pattern as thermo_vib.py's Debye-function quadrature: a fixed node count
# keeps this JAX-traceable/differentiable, unlike an adaptive integrator).
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_GL_X = jnp.asarray(_GL_X)
_GL_W = jnp.asarray(_GL_W)


def cavity_fixed_point(z, zB_over_zA, b_AA, b_AB, b_BB, n_iter: int = 400,
                       damping: float = 0.3):
    """Bulk cavity ratio r* = V_B/V_A, fixed point of

        r(k) = (zB/zA) * [(b_AB + b_BB r(k-1)) / (b_AA + b_AB r(k-1))]^(z-1)

    Fixed-iteration (not while-loop), so differentiable through the
    converged value. Validated to machine precision against exact
    finite-tree dynamic programming and, for z=2, the exact 1D transfer
    matrix (`examples/quasichemical_derivation.py`, Pieces 3-4) -- both
    checks used sufficiently weak ordering energies that this did not
    surface the issue damping fixes below.

    DAMPED (as of an independent review pass), not the raw map r_new =
    f(r_old): for sufficiently ordering-prone parameters (large |z*w/kT|,
    e.g. the bcc beta-brass case near 750-780 K in `cuzn_quasichemical.py`)
    the undoped map has more than one stable fixed point, and the RAW
    iteration is seed-sensitive at the level of the seed's LAST BIT --
    verified directly: at z=8 and this system's bcc omega, T=760 K,
    P_B(log_field=+1e-9) and P_B(log_field=-1e-9) converged (undamped) to
    0.005 and 0.995 respectively, an actual discontinuity where the true
    P_B(log_field) is continuous and passes through 0.5 (confirmed by a
    dense root search of the underlying composition equation finding
    exactly one root throughout this range -- there is no second
    thermodynamic branch to jump to). Damped iteration
    r_new = (1-damping) r_old + damping f(r_old) resolves this cleanly:
    re-run with damping=0.3, the same two seeds converge to 0.4949 and
    0.5051 -- a smooth, continuous, monotonic P_B(log_field) through the
    problem point, matching the true (undamped-map-blind) root. Checked
    NOT to change any previously-published value: at the well-conditioned
    parameters used elsewhere in this package (T=1176 K for Cu-Zn; every
    T used for the from-scratch validation suite), damped and undamped
    iteration converge to the identical r* to machine precision -- this
    only matters in the regime the prior undamped version got wrong.
    """
    def step(r, _):
        r_new = zB_over_zA * ((b_AB + b_BB * r) / (b_AA + b_AB * r)) ** (z - 1)
        return (1.0 - damping) * r + damping * r_new, None
    r_star, _ = jax.lax.scan(step, zB_over_zA, None, length=n_iter)
    return r_star


def bethe_marginals(z, zA, zB, b_AA, b_AB, b_BB, r_star):
    """Bulk site marginal (P_A, P_B) and bond marginal (P_AA, P_AB, P_BB)
    from the converged cavity ratio. See the derivation script (Piece 3)
    for the factoring argument and its validation against exact tree DP
    (site/edge marginals, geometric convergence with tree depth) across
    z in {2,3,4,6} and asymmetric zA != zB fields.
    """
    VA_site = zA * (b_AA + b_AB * r_star) ** z
    VB_site = zB * (b_AB + b_BB * r_star) ** z
    P_A = VA_site / (VA_site + VB_site)
    P_B = 1.0 - P_A
    wAA = (zA * (b_AA + b_AB * r_star) ** (z - 1)) ** 2 * b_AA
    wAB = (zA * (b_AA + b_AB * r_star) ** (z - 1)) \
        * (zB * (b_AB + b_BB * r_star) ** (z - 1)) * b_AB
    wBB = (zB * (b_AB + b_BB * r_star) ** (z - 1)) ** 2 * b_BB
    Z_edge = wAA + 2 * wAB + wBB
    return dict(P_A=P_A, P_B=P_B, P_AA=wAA / Z_edge,
               P_AB=2 * wAB / Z_edge, P_BB=wBB / Z_edge)


def bethe_energy(z, P_AA, P_AB, P_BB, e_AA, e_AB, e_BB):
    """<U>/N (per-atom internal energy) from the (validated) bond marginals."""
    return (z / 2.0) * (P_AA * e_AA + P_AB * e_AB + P_BB * e_BB)


def _U_at_beta_inv(z, zA, zB, e_AA, e_AB, e_BB, T):
    kT = K_B * T
    b_AA, b_AB, b_BB = jnp.exp(-e_AA / kT), jnp.exp(-e_AB / kT), jnp.exp(-e_BB / kT)
    r_star = cavity_fixed_point(z, zB / zA, b_AA, b_AB, b_BB)
    bm = bethe_marginals(z, zA, zB, b_AA, b_AB, b_BB, r_star)
    return bethe_energy(z, bm["P_AA"], bm["P_AB"], bm["P_BB"], e_AA, e_AB, e_BB)


def bethe_free_energy_grand(z, zA, zB, e_AA, e_AB, e_BB, T):
    """Grand potential per site, Omega_grand/N, via the exact thermodynamic
    identity d(F/T)/d(1/T) = U at fixed field (zA, zB), integrated from the
    EXACT T=infinity reference Omega_grand/T -> -k ln(zA+zB) (bonds carry
    no weight at T=infinity, so sites are exactly independent).

    Fixed 64-node Gauss-Legendre quadrature on x=1/T in [0, 1/T] (mapped
    from [-1,1]), matching the fixed-quadrature pattern used for the Debye
    function in thermo_vib.py -- keeps this JAX-differentiable. Validated
    against the exact 1D (z=2) transfer-matrix free energy to ~1e-10 eV
    (derivation script, Piece 5); NOT equal to the canonical (fixed-x)
    Helmholtz free energy -- see `solve_field_for_composition` /
    `quasichemical_correction` for the Legendre transform to fixed x.
    """
    x_hi = 1.0 / T
    x = 0.5 * x_hi * (_GL_X + 1.0)
    x = jnp.where(x <= 0.0, 1e-12, x)   # avoid T=inf exactly at the node
    Ts = 1.0 / x
    Us = jax.vmap(lambda Ti: _U_at_beta_inv(z, zA, zB, e_AA, e_AB, e_BB, Ti))(Ts)
    integral = 0.5 * x_hi * jnp.sum(_GL_W * Us)
    return (-K_B * jnp.log(zA + zB) + integral) * T


def solve_field_for_composition(z, x_B_target, e_AA, e_AB, e_BB, T,
                                n_iter: int = 80):
    """Find the fugacity ratio zB/zA (zA fixed at 1) whose resulting bulk
    composition (P_B from bethe_marginals) equals `x_B_target`.

    Reuses `phase_diagram._newton_solve` (damped, backtracking) rather
    than a hand-rolled plain Newton: an earlier plain-Newton version of
    this function diverged to nan for several ordinary (z, Omega, T)
    combinations -- exactly the failure mode `_newton_solve`'s
    step-halving exists to prevent, and it is already validated
    elsewhere in this package (including the exact-fixed-point-seed
    gradient fix). residual_fn is 1-D (wrapped as a length-1 vector).
    """
    kT = K_B * T
    b_AA, b_AB, b_BB = jnp.exp(-e_AA / kT), jnp.exp(-e_AB / kT), jnp.exp(-e_BB / kT)

    def resid(log_field_vec):
        zB_over_zA = jnp.exp(log_field_vec[0])
        r_star = cavity_fixed_point(z, zB_over_zA, b_AA, b_AB, b_BB)
        bm = bethe_marginals(z, 1.0, zB_over_zA, b_AA, b_AB, b_BB, r_star)
        return jnp.stack([bm["P_B"] - x_B_target])

    x0 = jnp.array([jnp.log(x_B_target / (1.0 - x_B_target))])  # ideal-solution guess
    log_field_star = _newton_solve(resid, x0, n_iter=n_iter, damping=1.0)
    return jnp.exp(log_field_star[0])


def symmetric_model_spinodal(Omega, T):
    """Regular-solution (Bragg-Williams) spinodal compositions (c_lo, c_hi)
    of G_mix(c) = Omega*c*(1-c) + kT*[c ln c + (1-c) ln(1-c)] -- the exact
    functional form `quasichemical_correction`'s own e_AA=e_BB=0,
    e_AB=Omega/z simplification reduces to at the Bragg-Williams (pre-Bethe)
    level. Derived directly (two lines of calculus), not cited: setting
    d2G/dc2 = -2*Omega + kT/[c(1-c)] = 0 gives c(1-c) = kT/(2*Omega), a
    quadratic in c with roots c = [1 +/- sqrt(1 - 2*kT/Omega)] / 2 -- real
    (a genuine two-phase/spinodal region exists) only when Omega > 2*kT,
    i.e. the regular-solution consolute temperature T_c = Omega/(2*k_B)
    exceeds T. Returns (nan, nan) when there is no gap (Omega <= 2*kT,
    including every Omega <= 0 / ordering case): every composition in
    (0,1) is then locally stable and `quasichemical_correction` is safe
    to call anywhere.

    Why this matters: for x_B strictly between c_lo and c_hi, NO
    homogeneous field reproduces that composition even in the simplified
    model `solve_field_for_composition` actually solves -- its Newton
    iteration has no root to find there (found empirically for Cu-Sn,
    Omega_bcc=652/Omega_fcc=893 meV at 1071 K: the solve silently returns
    its own unmoved initial seed, a residual of ~0.11-0.13 in composition,
    not a converged answer near the target). This is a property of the
    Omega-only simplification, not of the real (multi-term
    Redlich-Kister) curve being corrected, which can easily be locally
    stable at the same composition -- see `examples/cusn_quasichemical.py`
    for the case where this was caught.
    """
    kT = K_B * T
    Omega_safe = jnp.where(Omega > 0, Omega, 1.0)
    disc = 1.0 - 2.0 * kT / Omega_safe
    has_gap = (Omega > 2.0 * kT) & (disc > 0.0)
    sq = jnp.sqrt(jnp.where(disc > 0.0, disc, 0.0))
    c_lo = jnp.where(has_gap, 0.5 * (1.0 - sq), jnp.nan)
    c_hi = jnp.where(has_gap, 0.5 * (1.0 + sq), jnp.nan)
    return c_lo, c_hi


def quasichemical_correction(z, x_B, T, Omega, kappa_shape: float = 1.0):
    """The short-range-order correction to a regular-solution curve at
    fixed composition x_B and temperature T:

        Delta G_qc(x_B, T) = G_quasichemical(x_B, T) - G_Bragg-Williams(x_B, T)

    to be ADDED to an existing `regular_solution`/`redlich_kister_solution`
    free-energy curve (which is Bragg-Williams by construction). Uses the
    common simplification e_AA = e_BB = 0, e_AB = w = Omega/z (only the
    ordering energy w matters for a regular-solution-level Omega; `kappa_shape`
    is a placeholder hook for a future non-regular parameterization and
    defaults to a no-op).

    Bragg-Williams reference at the SAME x_B, T: G_BW = Omega*x_B*(1-x_B) +
    kT*[x_B ln x_B + (1-x_B) ln(1-x_B)] (the ideal + regular-solution terms
    already present in `regular_solution`), so Delta G_qc is exactly what
    should be added on top without double-counting either term.

    Returns NaN (not a silently-wrong finite number) when x_B falls inside
    `symmetric_model_spinodal(Omega, T)` -- see that function's docstring.
    This was added after an independent check found the field-solve
    silently returning ~0 progress from its own seed (not a converged
    root close to the target) for a strongly-clustering system evaluated
    at a dilute composition; the fix is to recognize the composition is
    outside this simplified model's domain BEFORE calling the solver, not
    to patch the solver into finding a root that provably does not exist
    there.
    """
    del kappa_shape
    w = Omega / z
    e_AA, e_AB, e_BB = 0.0, w, 0.0
    zB_over_zA = solve_field_for_composition(z, x_B, e_AA, e_AB, e_BB, T)
    Omega_grand = bethe_free_energy_grand(z, 1.0, zB_over_zA, e_AA, e_AB, e_BB, T)
    kT = K_B * T
    mu_A, mu_B = kT * jnp.log(1.0), kT * jnp.log(zB_over_zA)
    G_qc = Omega_grand + x_B * mu_B + (1.0 - x_B) * mu_A
    G_bw = Omega * x_B * (1.0 - x_B) + kT * (
        x_B * jnp.log(x_B) + (1.0 - x_B) * jnp.log(1.0 - x_B))

    c_lo, c_hi = symmetric_model_spinodal(Omega, T)
    inside_gap = (x_B > c_lo) & (x_B < c_hi)   # False whenever c_lo/c_hi are nan
    return jnp.where(inside_gap, jnp.nan, G_qc - G_bw)
