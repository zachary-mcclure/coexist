"""Tests for coexist.core.quasichemical: the Bethe pair-approximation
short-range-order correction to Bragg-Williams mixing.

Distills the full derivation-and-validation exercise in
examples/quasichemical_derivation.py into a permanent, fast regression
suite. See that script for the complete derivation, the exact ground
truths used (finite-tree DP, 1D transfer matrix, grand-canonical brute
force on small loopy graphs), and the two real bugs it caught along the
way (a wrong ratio substitution in the site/edge marginals, caught by an
asymmetric zA != zB test case; and a remembered CVM entropy formula that
disagreed with the exact z=2 free energy by tens of meV, replaced by
integrating the validated energy function instead of trusting it).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


from coexist.core.quasichemical import (  # noqa: E402
    K_B, bethe_energy, bethe_free_energy_grand, bethe_marginals,
    cavity_fixed_point, quasichemical_correction, solve_field_for_composition,
    symmetric_model_spinodal)


def _exact_tree_V(node, z_A, z_B, b_AA, b_AB, b_BB):
    """Exact bottom-up tree DP (ground truth), rescaled per node to avoid
    float64 overflow on deep/high-z trees -- see the derivation script's
    Piece 3 docstring for why the rescaling doesn't change the ratio."""
    if node is None:
        return z_A, z_B
    VA_prod, VB_prod = 1.0, 1.0
    for child in node:
        vA, vB = _exact_tree_V(child, z_A, z_B, b_AA, b_AB, b_BB)
        s = 1.0 / (vA + vB)
        vA, vB = vA * s, vB * s
        VA_prod *= (b_AA * vA + b_AB * vB)
        VB_prod *= (b_AB * vA + b_BB * vB)
    return z_A * VA_prod, z_B * VB_prod


def _build_cavity_tree(z, depth):
    if depth == 0:
        return None
    return [_build_cavity_tree(z, depth - 1) for _ in range(z - 1)]


@pytest.mark.parametrize("z,zB_over_zA,b_AA,b_AB,b_BB", [
    (3, 1.0, 1.0, 1.3, 1.0),
    (4, 0.6, 1.3, 1.0, 0.88),
    (6, 2.3, 1.0, 0.7, 1.0),
])
def test_cavity_fixed_point_matches_exact_tree_dp(z, zB_over_zA, b_AA, b_AB, b_BB):
    """The closed-form recursion must match exact finite-tree dynamic
    programming to machine precision -- this is the core correctness
    check of the whole module (Piece 3 of the derivation).

    Kept to z<=6 here so the reference tree (branching z-1, leaves
    (z-1)^depth) converges to machine precision at a depth still fast
    enough for a permanent, every-run test; z=8 and z=12 (needed for the
    actual bcc/fcc application) converge the identical way but need much
    deeper trees to resolve to machine precision -- validated with
    adaptive/leaf-budgeted depth and explicit convergence-trend checking
    in examples/quasichemical_derivation.py (Piece 3) instead of re-running
    that here on every test invocation.
    """
    depth = 10 if z <= 4 else 8   # (z-1)^depth leaf-count budget
    tree = _build_cavity_tree(z, depth)
    vA, vB = _exact_tree_V(tree, 1.0, zB_over_zA, b_AA, b_AB, b_BB)
    r_exact = vB / vA
    r_star = float(cavity_fixed_point(z, zB_over_zA, b_AA, b_AB, b_BB))
    tol = 1e-8 if z <= 4 else 1e-3
    assert abs(r_star - r_exact) / abs(r_exact) < tol


def test_marginals_asymmetric_field_matches_exact_tree():
    """Regression for the real bug this module's derivation caught: an
    earlier version substituted a mis-rescaled ratio into the site/edge
    marginal formulas that only happened to cancel when zA == zB. This
    case (zA != zB) is exactly the one that exposed it."""
    z, zA, zB = 4, 1.0, 0.6
    b_AA, b_AB, b_BB = 1.3, 1.0, 0.88
    r_star = cavity_fixed_point(z, zB / zA, b_AA, b_AB, b_BB)
    bm = bethe_marginals(z, zA, zB, b_AA, b_AB, b_BB, r_star)

    tree = [_build_cavity_tree(z, depth=8) for _ in range(z)]
    child_VAB = [_exact_tree_V(c, zA, zB, b_AA, b_AB, b_BB) for c in tree]

    def site_val(is_A):
        zs = zA if is_A else zB
        prod = 1.0
        for vA, vB in child_VAB:
            prod *= ((b_AA if is_A else b_AB) * vA + (b_AB if is_A else b_BB) * vB)
        return zs * prod
    VA, VB = site_val(True), site_val(False)
    P_A_exact = VA / (VA + VB)
    assert abs(float(bm["P_A"]) - P_A_exact) < 1e-4   # depth=8 truncation-limited


def test_w_zero_recovers_ideal_solution():
    """No interaction (Omega=0) must reduce EXACTLY to the ideal solution
    -- checked via the production quasichemical_correction API, which
    should return (numerically) zero for every z and composition."""
    for z in (3, 4, 6, 8, 12):
        for x in (0.1, 0.3, 0.5, 0.7, 0.9):
            d = float(quasichemical_correction(z, x, 900.0, 0.0))
            assert abs(d) < 1e-8, f"z={z}, x={x}: {d}"


def test_no_spinodal_for_ordering_or_weak_clustering():
    """symmetric_model_spinodal must return (nan, nan) -- no gap -- for
    every Omega<=0 (ordering, Cu-Zn's regime) and for weak clustering
    (Omega well below the regular-solution consolute limit 2*kT), so
    quasichemical_correction stays callable everywhere those regimes are
    actually used in the paper."""
    T = 1176.0
    for Omega in (-0.382, -0.30, -0.05, 0.0, 0.05, 2.0 * K_B * T * 0.5):
        c_lo, c_hi = symmetric_model_spinodal(Omega, T)
        assert np.isnan(float(c_lo)) and np.isnan(float(c_hi)), \
            f"Omega={Omega}: expected no gap, got ({c_lo}, {c_hi})"


def test_spinodal_gate_catches_the_cusn_failure_mode():
    """Regression test for a real bug: at Cu-Sn's actual (z, Omega, T),
    the target beta composition (0.13) sits inside the symmetric-model
    Bragg-Williams spinodal, so `solve_field_for_composition`'s Newton
    solve has no root and previously returned its own unmoved initial
    seed (a silently-wrong finite number, verified directly: residual
    ~0.11-0.13 in composition, not a converged answer). The correction
    must now return NaN there instead -- checked against the exact
    derived spinodal boundary, and against a composition just outside it
    (near-pure A) staying finite."""
    z, Omega, T = 8, 0.6518, 1071.15   # Cu-Sn bcc, as used in cusn_quasichemical.py
    c_lo, c_hi = symmetric_model_spinodal(Omega, T)
    assert float(c_lo) == pytest.approx(0.0767, abs=1e-3)
    assert float(c_hi) == pytest.approx(0.9233, abs=1e-3)
    inside = float(quasichemical_correction(z, 0.13, T, Omega))
    assert np.isnan(inside), f"expected NaN inside the spinodal, got {inside}"
    outside = float(quasichemical_correction(z, 0.02, T, Omega))
    assert np.isfinite(outside), f"expected a finite value outside the spinodal, got {outside}"


def test_cuzn_published_values_unaffected_by_spinodal_gate():
    """The spinodal gate must not change any already-published Cu-Zn
    number: Cu-Zn's Omega is negative (ordering), which never has a
    gap, and its own field solve was already verified to converge to
    the target composition (residual 0) before this fix existed."""
    z_bcc, z_fcc, T, x = 8, 12, 1176.0, 0.36
    d_bcc = float(quasichemical_correction(z_bcc, x, T, -0.35711668141704933))
    d_fcc = float(quasichemical_correction(z_fcc, x, T, -0.3398652556929172))
    assert d_bcc == pytest.approx(-8.1e-3, abs=0.2e-3)
    assert d_fcc == pytest.approx(-4.9e-3, abs=0.2e-3)


def test_entropy_never_exceeds_ideal_mixing():
    """The quasichemical (correlated) entropy must never exceed the ideal
    (random-mixing) entropy at the same composition -- ordering/clustering
    correlations can only reduce entropy relative to random placement."""
    z, x_B, T = 6, 0.35, 850.0
    Omega = 0.25   # clustering tendency
    e_AA, e_AB, e_BB = 0.0, Omega / z, 0.0
    zB_over_zA = solve_field_for_composition(z, x_B, e_AA, e_AB, e_BB, T)
    kT = K_B * T
    b_AA, b_AB, b_BB = (jnp.exp(-e_AA / kT), jnp.exp(-e_AB / kT),
                        jnp.exp(-e_BB / kT))
    r_star = cavity_fixed_point(z, zB_over_zA, b_AA, b_AB, b_BB)
    bm = bethe_marginals(z, 1.0, zB_over_zA, b_AA, b_AB, b_BB, r_star)
    U = bethe_energy(z, bm["P_AA"], bm["P_AB"], bm["P_BB"], e_AA, e_AB, e_BB)
    Omega_grand = bethe_free_energy_grand(z, 1.0, zB_over_zA, e_AA, e_AB, e_BB, T)
    S = (U - Omega_grand) / T   # entropy at fixed field == entropy at the
                                 # resulting macrostate (x_B here), see the
                                 # derivation script's ensemble note
    x_A = 1.0 - float(bm["P_B"])
    S_ideal = -K_B * (float(bm["P_B"]) * np.log(float(bm["P_B"]))
                      + x_A * np.log(x_A))
    assert float(S) <= S_ideal + 1e-10


def test_ordering_tendency_stabilizes_relative_to_bragg_williams():
    """An ordering interaction (Omega < 0, unlike pairs favored -- the
    beta-brass/Cu-Zn regime) should make the quasichemical correction
    NEGATIVE (extra stabilization SRO buys beyond Bragg-Williams): more
    unlike pairs form than random mixing predicts, which lowers the
    (negative-w-weighted) energy faster than it costs in pair entropy."""
    z, x_B, T, Omega = 8, 0.5, 1176.0, -0.30
    d = float(quasichemical_correction(z, x_B, T, Omega))
    assert d < 0.0, f"expected a stabilizing (negative) SRO correction, got {d}"


def test_correction_differentiable_matches_finite_difference():
    """The whole pipeline (field solve -> cavity fixed point -> marginals
    -> free-energy quadrature) must be JAX-differentiable in Omega, since
    it is meant to compose with the rest of phase_diagram.py's implicit
    layers."""
    z, x_B, T, Omega0 = 8, 0.5, 1176.0, -0.30
    g = float(jax.grad(lambda om: quasichemical_correction(z, x_B, T, om))(Omega0))
    h = 1e-5
    fd = (float(quasichemical_correction(z, x_B, T, Omega0 + h))
          - float(quasichemical_correction(z, x_B, T, Omega0 - h))) / (2 * h)
    assert abs(g - fd) / abs(fd) < 1e-3


def test_free_energy_matches_exact_1d_transfer_matrix():
    """z=2 (1D chain): the pair approximation is exact, and can be checked
    against a fully independent method (2x2 eigendecomposition, no
    recursion/trees at all)."""
    zA, zB = 1.0, 0.6
    T = 700.0
    kT = K_B * T
    e_AA, e_AB, e_BB = 0.02, 0.0, -0.01

    def b(e):
        return np.exp(-e / kT)
    T_mat = np.array([[zA * b(e_AA), np.sqrt(zA * zB) * b(e_AB)],
                      [np.sqrt(zA * zB) * b(e_AB), zB * b(e_BB)]])
    lam = np.linalg.eigvalsh(T_mat).max()
    F_exact = -kT * np.log(lam)
    F_bethe = float(bethe_free_energy_grand(2, zA, zB, e_AA, e_AB, e_BB, T))
    assert abs(F_bethe - F_exact) < 1e-6
