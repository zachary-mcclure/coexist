"""
Differentiable TERNARY phase equilibria: common tangent planes, tie-lines,
tie-triangles, and a simplex verifier — the Paper 2 §6 formulation made real.

Geometry: free energies are surfaces G(x, T) over the Gibbs simplex
x = (x_A, x_B, x_C), Σx = 1. With (x_B, x_C) independent and A eliminated,
chemical potentials are

    μ_A = G − x_B·∂G/∂x_B − x_C·∂G/∂x_C
    μ_B = μ_A + ∂G/∂x_B,   μ_C = μ_A + ∂G/∂x_C

and phase equilibrium is equality of (μ_A, μ_B, μ_C) across phases —
equivalently of (∂G/∂x_B, ∂G/∂x_C, μ_A), which is the residual set used
here. The common tangent LINE of the binary becomes a tangent PLANE:

    plane(x) = μ_A + μ_B'·x_B + μ_C'·x_C   (with μ' = ∂G/∂x at the point)

  * two phases  → 3 equations, 4 unknowns → a one-parameter FAMILY of
    tie-lines (a two-phase field); `tie_line` closes the system with
    either a lever rule through an overall composition or a component
    pin, `section_sweep` walks the family by continuation;
  * three phases → 6 equations, 6 unknowns → isolated TIE-TRIANGLES
    at fixed T (`tie_triangle`);
  * four phases + T → the ternary invariant (future work).

Parameterization: the binary logit generalizes to a two-component softmax
u = (u_B, u_C) → x = softmax(0, u_B, u_C), keeping compositions strictly
interior with no gradient-breaking clips. All solvers reuse the
backtracking-Newton `_newton_solve` of `phase_diagram` and are
fixed-iteration, so jax.grad through them yields implicit-function-theorem
derivatives of tie-line endpoints and triangle vertices w.r.t. anything
the surfaces close over (engine Ω's, ternary excess terms, T).

Verification: `plane_tangency_gap` is the simplex generalization of
`global_tangency_gap` — stationarity admits impostor roots in the binary
(measured) and the population can only grow with dimension, so
every reported ternary equilibrium must pass it. Degenerate corners
(x ~ 1e-6 for strong demixers) require float64, as in the binary case.

All energies per-atom eV; temperatures K.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from .phase_diagram import _newton_solve

K_B = 8.617333e-5

# A ternary free-energy surface: G(xB, xC, T) -> scalar, with
# x_A = 1 - xB - xC implied. Differentiable in all args and closures.
TernaryG = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]

__all__ = [
    "ternary_regular_solution", "softmax_comp", "chemical_potentials",
    "tie_line", "tie_triangle", "plane_tangency_gap", "section_sweep",
]


# ── free-energy constructor ───────────────────────────────────────────────────

def ternary_regular_solution(
    O_AB, O_AC, O_BC, L_ABC=0.0,
) -> TernaryG:
    """Ternary regular solution from the three binary interactions plus an
    optional ternary excess term:

        G = Ω_AB x_A x_B + Ω_AC x_A x_C + Ω_BC x_B x_C
            + L_ABC x_A x_B x_C + kT Σ_i x_i ln x_i

    Every parameter may be a traced JAX scalar (engine outputs) — gradients
    flow through. Reduces exactly to `regular_solution(Ω_AB)` on the
    x_C → 0 edge.
    """
    def G(xB, xC, T):
        xA = 1.0 - xB - xC
        ent = (xA * jnp.log(xA) + xB * jnp.log(xB) + xC * jnp.log(xC))
        return (O_AB * xA * xB + O_AC * xA * xC + O_BC * xB * xC
                + L_ABC * xA * xB * xC + K_B * T * ent)
    return G


# ── simplex parameterization and potentials ──────────────────────────────────

def softmax_comp(u):
    """u = (u_B, u_C) → (x_B, x_C) strictly inside the simplex
    (x_A = 1 − x_B − x_C > 0 automatically; gauge u_A = 0)."""
    e = jnp.exp(jnp.concatenate([jnp.zeros(1), u]))
    x = e / jnp.sum(e)
    return x[1], x[2]


def _inv_softmax(xB, xC):
    xA = 1.0 - xB - xC
    return jnp.array([jnp.log(xB / xA), jnp.log(xC / xA)])


def chemical_potentials(G: TernaryG, xB, xC, T):
    """(μ_A, μ_B, μ_C) at a point on one surface."""
    gB = jax.grad(G, argnums=0)(xB, xC, T)
    gC = jax.grad(G, argnums=1)(xB, xC, T)
    muA = G(xB, xC, T) - xB * gB - xC * gC
    return muA, muA + gB, muA + gC


def _equil_residuals(G_1: TernaryG, G_2: TernaryG, x1, x2, T):
    """The three tangent-plane conditions between two surfaces."""
    m1 = chemical_potentials(G_1, x1[0], x1[1], T)
    m2 = chemical_potentials(G_2, x2[0], x2[1], T)
    return jnp.stack([m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2]])


# ── two phases: tie-lines ─────────────────────────────────────────────────────

def tie_line(
    G_alpha: TernaryG,
    G_beta: TernaryG,
    T,
    x_overall=None,
    pin=None,
    x_alpha_guess=(0.1, 0.1),
    x_beta_guess=(0.6, 0.2),
    n_iter: int = 80,
    damping: float = 0.7,
):
    """One tie-line of a two-phase field.

    The 3 equilibrium conditions leave a one-parameter family; close the
    system with EITHER:
      * `x_overall=(xB0, xC0)`: lever rule — the tie-line passes through
        this overall composition (adds phase fraction f as unknown;
        5 equations, 5 unknowns), OR
      * `pin=("beta_C", value)`: pin one endpoint component
        ("alpha_B" | "alpha_C" | "beta_B" | "beta_C"; 4×4) — the natural
        closure for continuation sweeps.

    Returns ((xB_α, xC_α), (xB_β, xC_β), f) with f the α phase fraction
    (f = nan when `pin` is used). Differentiable in T and closures.
    """
    if (x_overall is None) == (pin is None):
        raise ValueError("provide exactly one of x_overall / pin")

    u_a0 = _inv_softmax(jnp.asarray(x_alpha_guess[0]),
                        jnp.asarray(x_alpha_guess[1]))
    u_b0 = _inv_softmax(jnp.asarray(x_beta_guess[0]),
                        jnp.asarray(x_beta_guess[1]))

    if x_overall is not None:
        xB0, xC0 = x_overall

        def residual(z):
            xa = softmax_comp(z[0:2])
            xb = softmax_comp(z[2:4])
            f = jax.nn.sigmoid(z[4])
            r_eq = _equil_residuals(G_alpha, G_beta, xa, xb, T)
            r_lev = jnp.stack([
                f * xa[0] + (1 - f) * xb[0] - xB0,
                f * xa[1] + (1 - f) * xb[1] - xC0,
            ])
            return jnp.concatenate([r_eq, r_lev])

        z0 = jnp.concatenate([u_a0, u_b0, jnp.zeros(1)])
        z = _newton_solve(residual, z0, n_iter=n_iter, damping=damping)
        xa = softmax_comp(z[0:2]); xb = softmax_comp(z[2:4])
        return xa, xb, jax.nn.sigmoid(z[4])

    which, val = pin
    idx = {"alpha_B": (0, 0), "alpha_C": (0, 1),
           "beta_B": (1, 0), "beta_C": (1, 1)}[which]

    def residual(z):
        xa = softmax_comp(z[0:2])
        xb = softmax_comp(z[2:4])
        r_eq = _equil_residuals(G_alpha, G_beta, xa, xb, T)
        pt = (xa, xb)[idx[0]][idx[1]]
        return jnp.concatenate([r_eq, jnp.stack([pt - val])])

    z0 = jnp.concatenate([u_a0, u_b0])
    z = _newton_solve(residual, z0, n_iter=n_iter, damping=damping)
    xa = softmax_comp(z[0:2]); xb = softmax_comp(z[2:4])
    return xa, xb, jnp.nan


# ── three phases: tie-triangles ───────────────────────────────────────────────

def tie_triangle(
    G_1: TernaryG,
    G_2: TernaryG,
    G_3: TernaryG,
    T,
    x_guesses=((0.05, 0.05), (0.8, 0.1), (0.1, 0.8)),
    n_iter: int = 100,
    damping: float = 0.7,
):
    """Three-phase coexistence at fixed T: one tangent plane touching three
    surfaces. Six unknowns (two softmax coordinates per vertex), six
    residuals (equal μ across phases 1-2 and 1-3). Returns the three
    vertices ((xB, xC) each), differentiable in T and closures.

    Verify every result with `plane_tangency_gap` — stationary impostors
    exist in the binary and generalize.
    """
    u0 = jnp.concatenate([
        _inv_softmax(jnp.asarray(g[0]), jnp.asarray(g[1]))
        for g in x_guesses])

    def residual(z):
        x1 = softmax_comp(z[0:2])
        x2 = softmax_comp(z[2:4])
        x3 = softmax_comp(z[4:6])
        r12 = _equil_residuals(G_1, G_2, x1, x2, T)
        r13 = _equil_residuals(G_1, G_3, x1, x3, T)
        return jnp.concatenate([r12, r13])

    z = _newton_solve(residual, u0, n_iter=n_iter, damping=damping)
    return (softmax_comp(z[0:2]), softmax_comp(z[2:4]),
            softmax_comp(z[4:6]))


# ── verification: the simplex tangency gap ────────────────────────────────────

def plane_tangency_gap(
    surfaces,
    x_point,
    G_at: TernaryG,
    T,
    n_grid: int = 120,
):
    """Simplex generalization of `global_tangency_gap`: anchor the tangent
    plane at `x_point` on surface `G_at`, and return the worst (most
    negative) signed gap min_x [G_φ(x) − plane(x)] over a barycentric grid
    of the simplex interior, across ALL `surfaces`.

    A true equilibrium has gap ≥ −tol for every phase; stationary
    impostors cut a surface by orders of magnitude more. Host-side
    diagnostic (not differentiable) — use for verification, not loss.
    """
    import numpy as np

    xB0, xC0 = (jnp.asarray(x_point[0]), jnp.asarray(x_point[1]))
    muA, muB, muC = chemical_potentials(G_at, xB0, xC0, jnp.asarray(T))
    muA, muB, muC = float(muA), float(muB), float(muC)

    eps = 1e-4
    worst = np.inf
    ts = np.linspace(eps, 1.0 - eps, n_grid)
    for G in surfaces:
        for xb in ts:
            xcs = ts[ts < 1.0 - xb - eps]
            if len(xcs) == 0:
                continue
            vals = np.array([
                float(G(jnp.asarray(xb), jnp.asarray(xc), jnp.asarray(T)))
                for xc in xcs])
            plane = muA + (muB - muA) * xb + (muC - muA) * xcs
            worst = min(worst, float((vals - plane).min()))
    return worst


# ── continuation sweep of a two-phase field (isothermal section) ─────────────

def section_sweep(
    G_alpha: TernaryG,
    G_beta: TernaryG,
    T,
    pin_which: str,
    pin_values,
    x_alpha_seed=(0.1, 0.01),
    x_beta_seed=(0.7, 0.01),
):
    """Walk a tie-line family by continuation: solve `tie_line` with the
    pinned component stepped through `pin_values`, each solve seeded by
    the previous endpoints. Returns (alphas, betas) as (n, 2) numpy
    arrays — the two binodal branches of the isothermal section, with
    tie-lines connecting row i to row i.

    Host-side (like `eutectic_diagram`); for differentiable whole-section
    Jacobians, re-solve the converged rows in a vmapped batch seeded by
    this sweep (the binary `tangent_sweep` pattern).
    """
    import numpy as np

    ga, gb = tuple(x_alpha_seed), tuple(x_beta_seed)
    alphas, betas = [], []
    for v in pin_values:
        xa, xb, _ = tie_line(G_alpha, G_beta, T, pin=(pin_which, float(v)),
                             x_alpha_guess=ga, x_beta_guess=gb)
        ga = (float(xa[0]), float(xa[1]))
        gb = (float(xb[0]), float(xb[1]))
        alphas.append(ga); betas.append(gb)
    return np.array(alphas), np.array(betas)
