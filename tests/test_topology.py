"""Tests for the topology extensions of coexist.core.phase_diagram:
three_phase_equilibrium, classify_invariant, line_compound, tangent_sweep."""
import jax
import jax.numpy as jnp
import numpy as np

from coexist.core.phase_diagram import (
    K_B, classify_invariant, common_tangent, eutectic_point, line_compound,
    liquid_solution, regular_solution, tangent_sweep, three_phase_equilibrium)

# a solid + liquid pair with a well-formed eutectic (Ag-Cu-like numbers)
OM_S, OM_L = 0.35, 0.155
FUS = dict(dH_fus_A=0.117, T_m_A=1235.0, dH_fus_B=0.137, T_m_B=1358.0)
G_S = regular_solution(OM_S)
G_L = liquid_solution(OM_L, **FUS)


def test_three_phase_reduces_to_eutectic_point():
    """General three-curve solver on (solid, liquid, solid) must agree with
    the dedicated eutectic solver."""
    ca_e, cl_e, cb_e, T_e = eutectic_point(G_S, G_L)
    c1, c2, c3, T = three_phase_equilibrium(
        G_S, G_L, G_S, c_guesses=(0.1, 0.4, 0.9), T_guess=1000.0)
    np.testing.assert_allclose(float(T), float(T_e), rtol=2e-4)
    np.testing.assert_allclose(float(c1), float(ca_e), atol=2e-3)
    np.testing.assert_allclose(float(c2), float(cl_e), atol=2e-3)
    np.testing.assert_allclose(float(c3), float(cb_e), atol=2e-3)


def test_monotectic_found_and_ordered():
    """Demixing liquid (Ω_L > 2kT_mono) + near-insoluble solid → the solver
    finds solid < L1 < L2 with T between the melting points.

    Runs in float64 (the solid tangent point sits at c₁ ~ 1e-5). Newton
    stationarity has SPURIOUS roots here — from seed (0.005, 0.2, 0.97)
    it converges to a stationary-but-not-tangent configuration 100 K above
    the true monotectic. The global_tangency_gap verifier is what tells
    them apart; this test locks in both the true root and the rejection
    of the spurious one.
    """
    from coexist.core.phase_diagram import global_tangency_gap

    jax.config.update("jax_enable_x64", True)
    try:
        G_s = regular_solution(0.9)
        G_l = liquid_solution(0.27, dH_fus_A=0.111, T_m_A=933.5,
                              dH_fus_B=0.049, T_m_B=600.6)
        c1, c2, c3, T = three_phase_equilibrium(
            G_s, G_l, G_l, c_guesses=(0.001, 0.05, 0.95), T_guess=900.0)
        assert float(c1) < 0.01 < float(c2) < float(c3)
        assert 850.0 < float(T) < 950.0
        # true common tangent: no curve dips below the invariant line
        gap = global_tangency_gap((G_s, G_l), (c1, c2, c3), float(T))
        assert gap > -1e-6
        # equal chemical potentials at the two liquid points
        dG = jax.grad(G_l, argnums=0)
        np.testing.assert_allclose(float(dG(c2, T)), float(dG(c3, T)),
                                   atol=1e-6)

        # the documented spurious root: stationary, but NOT a tangent
        s1, s2, s3, Ts = three_phase_equilibrium(
            G_s, G_l, G_l, c_guesses=(0.005, 0.2, 0.97), T_guess=950.0)
        bad_gap = global_tangency_gap((G_s, G_l), (s1, s2, s3), float(Ts))
        assert bad_gap < -1e-3          # verifier rejects it
    finally:
        jax.config.update("jax_enable_x64", False)


def test_classify_invariant_table():
    assert classify_invariant(("solid", "liquid", "solid")) == "eutectic"
    assert classify_invariant(("solid", "solid", "liquid")) == "peritectic"
    assert classify_invariant(("liquid", "solid", "solid")) == "peritectic"
    assert classify_invariant(("solid", "liquid", "liquid")) == "monotectic"
    assert classify_invariant(("liquid", "solid", "liquid")) == "syntectic"
    assert "solid" in classify_invariant(("solid", "solid", "solid"))


def test_line_compound_tangent_stays_near_stoichiometry():
    """Tangent point on a κ=300 parabola sits within ~0.5 at.% of c0."""
    G_sol = regular_solution(-2.0)          # strongly ordering solution
    G_cmp = line_compound(0.25, -0.45)
    c_g, c_p = common_tangent(G_sol, G_cmp, 1000.0,
                              c_alpha_guess=0.08, c_beta_guess=0.249)
    assert abs(float(c_p) - 0.25) < 5e-3
    assert 0.0 < float(c_g) < 0.25


def test_line_compound_solvus_gradient():
    """∂(solvus)/∂H_f is finite and positive: stabilising the compound
    (more negative H_f) pulls solute out of the solution."""
    G_sol = regular_solution(-2.0)

    def solvus(Hf):
        c_g, _ = common_tangent(G_sol, line_compound(0.25, Hf), 1000.0,
                                c_alpha_guess=0.08, c_beta_guess=0.249)
        return c_g

    g = float(jax.grad(solvus)(jnp.asarray(-0.45)))
    assert np.isfinite(g) and g > 0.0


def test_tangent_sweep_matches_loop():
    """vmapped sweep equals per-T host loop on the solvus."""
    Ts = jnp.array([600.0, 700.0, 800.0])
    ca_loop, cb_loop = [], []
    for T in Ts:
        ca, cb = common_tangent(G_S, G_S, float(T))
        ca_loop.append(float(ca)); cb_loop.append(float(cb))
    ca_v, cb_v = tangent_sweep(G_S, G_S, Ts,
                               jnp.full(3, 0.05), jnp.full(3, 0.95))
    np.testing.assert_allclose(np.asarray(ca_v), ca_loop, atol=1e-5)
    np.testing.assert_allclose(np.asarray(cb_v), cb_loop, atol=1e-5)


def test_tangent_sweep_whole_boundary_jacobian():
    """One jax.jacobian call yields ∂(entire solvus)/∂Ω."""
    Ts = jnp.array([600.0, 700.0, 800.0])

    def solvus_of_omega(om):
        ca, _ = tangent_sweep(regular_solution(om), regular_solution(om),
                              Ts, jnp.full(3, 0.05), jnp.full(3, 0.95))
        return ca

    J = jax.jacfwd(solvus_of_omega)(0.35)
    assert J.shape == (3,)
    assert np.all(np.isfinite(np.asarray(J)))
    assert np.all(np.asarray(J) < 0.0)   # larger Ω → less solubility


def test_newton_solve_gradient_at_exact_fixed_point_seed():
    """Regression: seeding a Newton solve EXACTLY at its own converged root
    (bit-identical) must not zero the reverse- or forward-mode gradient.

    This is precisely the pattern `tangent_sweep` creates when seeded from a
    continuation sweep already converged at the same parameter value (as in
    the whole-boundary UQ Jacobian, Sec. 4.4 / Fig. 1): before the 1e-9 seed
    nudge in `_newton_solve`, dx was exactly 0 at every one of the 60
    unrolled iterations whenever x0 was already the root, and the unrolled
    reverse-mode (and forward-mode) gradient collapsed to exactly 0 even
    though the true implicit-function-theorem derivative is nonzero and
    matches finite differences the instant the seed differs from the root.
    Found via an AD-vs-FD audit of tangent_sweep; ~5/30 T-points in the
    published Fig. 1 band were silently zero before this fix.
    """
    # This is a float64-precision effect (a 1e-9 seed nudge underflows in
    # float32); force x64 on for this test regardless of global state left
    # by other tests in this file, matching the paper's own "float64 is not
    # optional" stance (Sec. 3.4(iii)).
    jax.config.update("jax_enable_x64", True)
    try:
        om0 = 0.16
        T = 1134.4

        def ca_of_om(om, ca_seed, cb_seed):
            ca, _ = common_tangent(liquid_solution(om, dH_fus_A=0.1109,
                                                    T_m_A=1234.93, dH_fus_B=0.1314,
                                                    T_m_B=1357.77),
                                   G_S, T, c_alpha_guess=ca_seed,
                                   c_beta_guess=cb_seed)
            return ca

        # Converge once with a generic seed, then re-seed EXACTLY at that
        # root — the degenerate case tangent_sweep constructs by design.
        ca_root, cb_root = common_tangent(
            liquid_solution(om0, dH_fus_A=0.1109, T_m_A=1234.93,
                            dH_fus_B=0.1314, T_m_B=1357.77), G_S, T,
            c_alpha_guess=0.6, c_beta_guess=0.98)

        g_rev = float(jax.grad(ca_of_om)(om0, float(ca_root), float(cb_root)))
        g_fwd = float(jax.jacfwd(ca_of_om)(om0, float(ca_root), float(cb_root)))
        h = 1e-5
        fd = (ca_of_om(om0 + h, float(ca_root), float(cb_root))
              - ca_of_om(om0 - h, float(ca_root), float(cb_root))) / (2 * h)

        assert abs(g_rev) > 1e-3, "reverse-mode gradient collapsed to ~0 at an exact-fixed-point seed"
        assert abs(g_rev - fd) / abs(fd) < 1e-3
        assert abs(g_fwd - fd) / abs(fd) < 1e-3
    finally:
        jax.config.update("jax_enable_x64", False)
