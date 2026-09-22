"""Tests for coexist.core.ternary — tangent planes, tie-lines,
tie-triangles, and the simplex verifier. All in float64 (degenerate
corners are float64-only, as in the binary case)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


from coexist.core.ternary import (  # noqa: E402
    chemical_potentials, plane_tangency_gap, section_sweep, softmax_comp,
    ternary_regular_solution, tie_line, tie_triangle)


def test_softmax_stays_interior():
    for u in ([0.0, 0.0], [30.0, -30.0], [-20.0, -20.0]):
        xB, xC = softmax_comp(jnp.array(u))
        xA = 1.0 - float(xB) - float(xC)
        assert 0.0 < float(xB) < 1.0 and 0.0 < float(xC) < 1.0
        assert 0.0 < xA < 1.0


def test_chemical_potentials_gibbs_duhem():
    """G = Σ x_i μ_i must hold identically."""
    G = ternary_regular_solution(0.3, 0.2, 0.1, L_ABC=0.05)
    xB, xC, T = jnp.float64(0.3), jnp.float64(0.25), jnp.float64(900.0)
    muA, muB, muC = chemical_potentials(G, xB, xC, T)
    xA = 1.0 - xB - xC
    np.testing.assert_allclose(
        float(xA * muA + xB * muB + xC * muC), float(G(xB, xC, T)),
        rtol=1e-12)


def test_tie_line_reduces_to_binary():
    """With no C interactions and x_C → 0, the tie-line endpoints match
    the binary common tangent."""
    from coexist.core.phase_diagram import common_tangent, regular_solution

    O, T = 0.35, 900.0
    G = ternary_regular_solution(O, 0.0, 0.0)
    xa, xb, f = tie_line(G, G, T, x_overall=(0.5, 1e-5),
                         x_alpha_guess=(0.1, 1e-5),
                         x_beta_guess=(0.9, 1e-5))
    ca, cb = common_tangent(regular_solution(O), regular_solution(O), T)
    np.testing.assert_allclose(float(xa[0]), float(ca), atol=2e-5)
    np.testing.assert_allclose(float(xb[0]), float(cb), atol=2e-5)
    assert 0.4 < float(f) < 0.6      # symmetric overall → f ≈ ½


def test_symmetric_triangle_verified_and_impostor_rejected():
    """Symmetric demixer: corner seeds give the permutation-symmetric
    triangle (verified); mid seeds converge to the anti-corner stationary
    impostor, which the simplex verifier rejects — the binary impostor
    phenomenon generalizes."""
    G = ternary_regular_solution(0.4, 0.4, 0.4)
    T = 600.0

    v1, v2, v3 = tie_triangle(G, G, G, T,
                              x_guesses=((0.005, 0.005), (0.98, 0.005),
                                         (0.005, 0.98)))
    xA1 = 1 - float(v1[0]) - float(v1[1])
    assert xA1 > 0.99                          # A-rich corner
    np.testing.assert_allclose(float(v2[0]), xA1, atol=1e-4)  # symmetry
    np.testing.assert_allclose(float(v3[1]), xA1, atol=1e-4)
    gap = plane_tangency_gap((G,), v1, G, T, n_grid=50)
    assert gap > -1e-6

    vi = tie_triangle(G, G, G, T,
                      x_guesses=((0.05, 0.05), (0.9, 0.05), (0.05, 0.9)))
    gap_i = plane_tangency_gap((G,), vi[0], G, T, n_grid=50)
    assert gap_i < -1e-3                       # rejected


def test_triangle_gradient_matches_fd():
    def vertex(O_AB):
        G = ternary_regular_solution(O_AB, 0.4, 0.4)
        v = tie_triangle(G, G, G, 600.0,
                         x_guesses=((0.005, 0.005), (0.98, 0.005),
                                    (0.005, 0.98)))
        return v[1][0]

    g = float(jax.grad(vertex)(jnp.float64(0.4)))
    h = 1e-4
    fd = (float(vertex(jnp.float64(0.4 + h)))
          - float(vertex(jnp.float64(0.4 - h)))) / (2 * h)
    assert np.isfinite(g)
    np.testing.assert_allclose(g, fd, rtol=1e-3)


def test_triangle_collapse_diagnoses_missing_field():
    """Above the consolute point of the B-C pair, two vertices collapse —
    the solver diagnoses 'no three-phase field' instead of fabricating one."""
    from coexist.core.phase_diagram import K_B

    G = ternary_regular_solution(0.5, 0.5, 0.1)
    T = 0.1 / (2 * K_B) * 1.5                  # 50% above T_c of the BC pair
    v1, v2, v3 = tie_triangle(G, G, G, T,
                              x_guesses=((0.005, 0.005), (0.95, 0.02),
                                         (0.02, 0.95)))
    sep = float(jnp.hypot(v2[0] - v3[0], v2[1] - v3[1]))
    assert sep < 1e-6


def test_section_sweep_continuity():
    """A short pinned sweep produces monotone β motion and finite α."""
    G = ternary_regular_solution(0.4, 0.6, 0.05)
    al, be = section_sweep(G, G, 700.0, "beta_C",
                           np.linspace(1e-4, 0.1, 5),
                           x_alpha_seed=(0.01, 1e-4),
                           x_beta_seed=(0.95, 1e-4))
    assert np.all(np.isfinite(al)) and np.all(np.isfinite(be))
    assert np.all(np.diff(be[:, 1]) > 0)       # pinned coordinate advances
    gap = plane_tangency_gap((G,), tuple(al[2]), G, 700.0, n_grid=40)
    assert gap > -1e-6
