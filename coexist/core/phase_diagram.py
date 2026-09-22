"""
Differentiable binary phase diagrams: common tangents, solvus/liquidus/solidus,
and the eutectic point — all as JAX root-solves with implicit gradients.

Classical construction, made differentiable:
  1. Each phase φ has a molar Gibbs curve G_φ(c, T).
  2. Two-phase equilibrium = common tangent between two curves:
         G'_α(c_α) = G'_β(c_β)                     (equal chemical potential)
         G_β(c_β) − G_α(c_α) = G'_α(c_α)(c_β − c_α) (equal grand potential)
     Solved by damped Newton in logit space u = ln(c/(1−c)) so compositions
     stay in (0,1) with no gradient-killing clips.
  3. Eutectic point = ONE straight line tangent to three curves (α, L, β)
     simultaneously — four residuals in four unknowns (c_α, c_L, c_β, T_e).

Because the solvers are fixed-iteration `lax.scan` Newton loops, jax.grad
through them converges to the implicit-function-theorem derivative at the
root. Everything downstream and upstream stays differentiable: if Ω comes
from an external engine via SimulatorAdapter, ∂T_eutectic/∂(atom positions)
is one backward pass.

All energies are per-atom eV; temperatures in K.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

K_B = 8.617333e-5   # eV/K

# A free-energy curve is any callable G(c, T) -> scalar, differentiable in
# both args (and in anything it closes over).
FreeEnergyFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


# ── Reference free-energy constructors ────────────────────────────────────────

def regular_solution(omega: jnp.ndarray | float) -> FreeEnergyFn:
    """G(c,T) = Ω·c(1−c) + kT[c·ln c + (1−c)·ln(1−c)]  (solid reference state).

    `omega` may be a traced JAX scalar (e.g. produced by a SimulatorAdapter
    chain) — gradients flow through it.
    """
    def G(c, T):
        return (omega * c * (1.0 - c)
                + K_B * T * (c * jnp.log(c) + (1.0 - c) * jnp.log(1.0 - c)))
    return G


def liquid_solution(
    omega_liq: jnp.ndarray | float,
    dH_fus_A: float, T_m_A: float,
    dH_fus_B: float, T_m_B: float,
) -> FreeEnergyFn:
    """Liquid Gibbs curve relative to the SOLID reference states.

    Pure-component ends carry the fusion free energy
    ΔG_m,i(T) = ΔH_m,i·(1 − T/T_m,i)  (linear/Richard approximation), so the
    liquid drops below the solid above T_m and rises below it.
    """
    def G(c, T):
        dG_A = dH_fus_A * (1.0 - T / T_m_A)
        dG_B = dH_fus_B * (1.0 - T / T_m_B)
        return ((1.0 - c) * dG_A + c * dG_B
                + omega_liq * c * (1.0 - c)
                + K_B * T * (c * jnp.log(c) + (1.0 - c) * jnp.log(1.0 - c)))
    return G


def subregular_solution(
    omega0: jnp.ndarray | float,
    omega1: jnp.ndarray | float,
) -> FreeEnergyFn:
    """Redlich-Kister first-order (asymmetric) solid solution:
    G = c(1−c)·[Ω₀ + Ω₁(1−2c)] + kT·[ideal entropy]."""
    def G(c, T):
        return (c * (1.0 - c) * (omega0 + omega1 * (1.0 - 2.0 * c))
                + K_B * T * (c * jnp.log(c) + (1.0 - c) * jnp.log(1.0 - c)))
    return G


def redlich_kister_solution(L: jnp.ndarray) -> FreeEnergyFn:
    """Arbitrary-order Redlich-Kister solid solution from coefficient vector
    L = (L₀, L₁, …): G = c(1−c)·Σ_k L_k(1−2c)^k + kT·[ideal entropy].

    `L` is typically the output of `core.mixing.redlich_kister_fit` on
    engine-computed ΔH_mix(x) — gradients flow through the fit into the
    engine. L of length 1 reduces to `regular_solution`, length 2 to
    `subregular_solution`.
    """
    L = jnp.atleast_1d(L)
    ks = jnp.arange(L.shape[0])

    def G(c, T):
        series = jnp.sum(L * (1.0 - 2.0 * c) ** ks)
        return (c * (1.0 - c) * series
                + K_B * T * (c * jnp.log(c) + (1.0 - c) * jnp.log(1.0 - c)))
    return G


def redlich_kister_liquid(
    L_liq: jnp.ndarray,
    dH_fus_A: float, T_m_A: float,
    dH_fus_B: float, T_m_B: float,
) -> FreeEnergyFn:
    """Liquid with Redlich-Kister excess, relative to solid reference states
    (fusion terms as in `liquid_solution`)."""
    L_liq = jnp.atleast_1d(L_liq)
    ks = jnp.arange(L_liq.shape[0])

    def G(c, T):
        dG_A = dH_fus_A * (1.0 - T / T_m_A)
        dG_B = dH_fus_B * (1.0 - T / T_m_B)
        series = jnp.sum(L_liq * (1.0 - 2.0 * c) ** ks)
        return ((1.0 - c) * dG_A + c * dG_B
                + c * (1.0 - c) * series
                + K_B * T * (c * jnp.log(c) + (1.0 - c) * jnp.log(1.0 - c)))
    return G


def line_compound(
    c0: float,
    H_f: jnp.ndarray | float,
    S_f: jnp.ndarray | float = 0.0,
    kappa: float = 300.0,
) -> FreeEnergyFn:
    """Stoichiometric (line) compound as a narrow parabola at composition c0:

        G(c, T) = H_f − T·S_f + κ·(c − c0)²

    H_f is the formation energy per atom RELATIVE TO THE SOLID REFERENCE
    STATES of the diagram (same convention as every other curve here) —
    e.g. from an engine single point on the ordered structure. κ (eV) sets
    the compound's composition tolerance: against eV-scale solution curves,
    κ=300 holds the tangent point within ~0.3 at.% of c0 (κ=20 lets it
    drift ~5 at.%, visibly softening the compound — checked on Ni₃Al).
    v1 simplification: real sublattice models later.
    """
    def G(c, T):
        return H_f - T * S_f + kappa * (c - c0) ** 2
    return G


# ── Newton machinery (fixed iterations, damped, logit-space) ─────────────────

def _sig(u):
    return jax.nn.sigmoid(u)


def _newton_solve(residual_fn, x0, n_iter=60, damping=1.0):
    """Backtracking Newton with dense Jacobian via jacfwd; fixed lax.scan length.

    Each iteration tries step fractions {1, ½, ¼, ⅒}·damping·Δx and takes the
    largest one whose residual is finite and non-increasing — full Newton where
    it works, automatic damping where it would overshoot (e.g. near-singular
    Jacobians just above T_c). Near the root the full step is always accepted,
    so differentiating the unrolled iteration w.r.t. any closed-over parameter
    still yields the implicit-function-theorem derivative.

    x0 is nudged by 1e-9 before the scan starts. This matters only in a
    degenerate corner case (found via a vmap/AD audit of the whole-boundary
    Jacobian, Sec. 4.4): if x0 is seeded EXACTLY at the residual's root
    (bit-identical — the situation `tangent_sweep` creates by construction,
    reusing a converged continuation solution as the seed for a fresh solve
    at the same parameter value), dx is exactly 0 at every iteration and the
    reverse-mode gradient of the unrolled scan collapses to exactly zero,
    even though the true implicit-function-theorem derivative is nonzero and
    matches finite differences as soon as the seed differs from the root by
    even 1e-10. The nudge is far below any reported precision in this module
    (residuals are checked to 1e-8..1e-14) and is wiped out well within the
    60-iteration budget by Newton's quadratic convergence; it changes no
    converged value, only prevents this exact-fixed-point degeneracy.
    """
    x0 = x0 + 1e-9
    jac = jax.jacfwd(residual_fn)
    fractions = jnp.array([1.0, 0.5, 0.25, 0.1])

    def step(x, _):
        r = residual_fn(x)
        rnorm = jnp.linalg.norm(r)
        J = jac(x)
        dx = jnp.linalg.solve(J, r)

        cands = x[None, :] - (damping * fractions)[:, None] * dx[None, :]
        cand_norms = jax.vmap(lambda xc: jnp.linalg.norm(residual_fn(xc)))(cands)
        # score: prefer largest fraction with finite, non-increasing residual
        ok = jnp.isfinite(cand_norms) & (cand_norms <= rnorm * (1.0 + 1e-8))
        score = jnp.where(ok, cand_norms, jnp.inf)
        i_best = jnp.argmin(score)
        x_new = jnp.where(jnp.isinf(score[i_best]), x, cands[i_best])
        return x_new, None

    x_star, _ = jax.lax.scan(step, x0, None, length=n_iter)
    return x_star


# ── Two-point common tangent (asymmetric — the general construction) ─────────

def common_tangent(
    G_alpha: FreeEnergyFn,
    G_beta: FreeEnergyFn,
    T,
    c_alpha_guess: float = 0.05,
    c_beta_guess: float = 0.95,
    n_iter: int = 60,
    damping: float = 1.0,
):
    """Common tangent between two (generally different) free-energy curves.

    Solves, in logit variables u = ln(c/(1−c)):
        r₁ = G'_α(c_α) − G'_β(c_β)                          = 0
        r₂ = G_β(c_β) − G_α(c_α) − G'_α(c_α)·(c_β − c_α)    = 0

    Returns (c_alpha, c_beta) as JAX scalars — differentiable w.r.t. T and
    anything the curves close over (Ω, engine atoms, ...).

    Seeds select WHICH tangent you get when several exist (e.g. α–L on the
    A-rich side vs β–L on the B-rich side): seed each variable on the branch
    you want.
    """
    dG_a = jax.grad(G_alpha, argnums=0)
    dG_b = jax.grad(G_beta, argnums=0)

    def residual(x):
        ca, cb = _sig(x[0]), _sig(x[1])
        mu_a = dG_a(ca, T)
        r1 = mu_a - dG_b(cb, T)
        r2 = G_beta(cb, T) - G_alpha(ca, T) - mu_a * (cb - ca)
        return jnp.stack([r1, r2])

    u0 = jnp.array([jnp.log(c_alpha_guess / (1 - c_alpha_guess)),
                    jnp.log(c_beta_guess / (1 - c_beta_guess))])
    u = _newton_solve(residual, u0, n_iter=n_iter, damping=damping)
    return _sig(u[0]), _sig(u[1])


def tangent_residual(G_alpha, G_beta, c_alpha, c_beta, T):
    """Max |residual| of the two tangency conditions — convergence diagnostic."""
    dG_a = jax.grad(G_alpha, argnums=0)
    dG_b = jax.grad(G_beta, argnums=0)
    mu_a = dG_a(c_alpha, T)
    r1 = mu_a - dG_b(c_beta, T)
    r2 = G_beta(c_beta, T) - G_alpha(c_alpha, T) - mu_a * (c_beta - c_alpha)
    return jnp.maximum(jnp.abs(r1), jnp.abs(r2))


# ── Eutectic point: one line tangent to three curves ─────────────────────────

def eutectic_point(
    G_solid: FreeEnergyFn,
    G_liquid: FreeEnergyFn,
    c_alpha_guess: float = 0.1,
    c_liq_guess: float = 0.4,
    c_beta_guess: float = 0.9,
    T_guess: float = 1000.0,
    n_iter: int = 80,
    damping: float = 0.7,
):
    """Solve for the eutectic: a single tangent line touching the solid curve
    at c_α and c_β (the two solvus limbs) and the liquid curve at c_L, all at
    one temperature T_e.

    Unknowns x = (u_α, u_L, u_β, T). Residuals:
        r₁ = G'_s(c_α) − G'_L(c_L)
        r₂ = G'_s(c_β) − G'_L(c_L)
        r₃ = G_L(c_L) − G_s(c_α) − G'_s(c_α)·(c_L − c_α)
        r₄ = G_s(c_β) − G_s(c_α) − G'_s(c_α)·(c_β − c_α)

    Returns (c_alpha, c_liq, c_beta, T_e) — every output differentiable
    w.r.t. whatever the curves close over (Ω_s from an engine, ΔH_fus, ...).
    T is scaled by 1/1000 internally so the Newton Jacobian is well-conditioned.
    """
    dG_s = jax.grad(G_solid, argnums=0)
    dG_l = jax.grad(G_liquid, argnums=0)

    def residual(x):
        ca, cl, cb = _sig(x[0]), _sig(x[1]), _sig(x[2])
        T = x[3] * 1000.0
        mu_a = dG_s(ca, T)
        r1 = mu_a - dG_l(cl, T)
        r2 = dG_s(cb, T) - dG_l(cl, T)
        r3 = G_liquid(cl, T) - G_solid(ca, T) - mu_a * (cl - ca)
        r4 = G_solid(cb, T) - G_solid(ca, T) - mu_a * (cb - ca)
        return jnp.stack([r1, r2, r3, r4])

    x0 = jnp.array([jnp.log(c_alpha_guess / (1 - c_alpha_guess)),
                    jnp.log(c_liq_guess / (1 - c_liq_guess)),
                    jnp.log(c_beta_guess / (1 - c_beta_guess)),
                    T_guess / 1000.0])
    x = _newton_solve(residual, x0, n_iter=n_iter, damping=damping)
    return _sig(x[0]), _sig(x[1]), _sig(x[2]), x[3] * 1000.0


# ── General three-phase invariant: eutectic / peritectic / monotectic ────────

def three_phase_equilibrium(
    G_1: FreeEnergyFn,
    G_2: FreeEnergyFn,
    G_3: FreeEnergyFn,
    c_guesses: tuple[float, float, float],
    T_guess: float,
    n_iter: int = 80,
    damping: float = 0.7,
):
    """One straight line tangent to three (generally distinct) curves — the
    invariant reaction underlying every three-phase topology:

        eutectic    L → α + β        (liquid curve in the middle)
        peritectic  α + L → β        (new solid in the middle)
        monotectic  L₁ → α + L₂      (pass the SAME liquid fn twice)
        eutectoid/peritectoid        (all three solid)

    Curves are passed in tangent order c₁ < c₂ < c₃ (the seeds select the
    branch). Unknowns (u₁, u₂, u₃, T); residuals: equal slopes at 1–2 and
    2–3, plus collinearity of points 2 and 3 with point 1.

    Returns (c_1, c_2, c_3, T_inv), all differentiable w.r.t. whatever the
    curves close over. `eutectic_point` remains the convenience wrapper for
    the single-solid-curve case.
    """
    dG_1 = jax.grad(G_1, argnums=0)
    dG_2 = jax.grad(G_2, argnums=0)
    dG_3 = jax.grad(G_3, argnums=0)

    def residual(x):
        c1, c2, c3 = _sig(x[0]), _sig(x[1]), _sig(x[2])
        T = x[3] * 1000.0
        mu = dG_1(c1, T)
        r1 = mu - dG_2(c2, T)
        r2 = mu - dG_3(c3, T)
        r3 = G_2(c2, T) - G_1(c1, T) - mu * (c2 - c1)
        r4 = G_3(c3, T) - G_1(c1, T) - mu * (c3 - c1)
        return jnp.stack([r1, r2, r3, r4])

    g1, g2, g3 = c_guesses
    x0 = jnp.array([jnp.log(g1 / (1 - g1)),
                    jnp.log(g2 / (1 - g2)),
                    jnp.log(g3 / (1 - g3)),
                    T_guess / 1000.0])
    x = _newton_solve(residual, x0, n_iter=n_iter, damping=damping)
    return _sig(x[0]), _sig(x[1]), _sig(x[2]), x[3] * 1000.0


def global_tangency_gap(
    curves,
    c_points,
    T,
    n_grid: int = 2001,
    anchor: int = 0,
):
    """Verify a candidate invariant line is a TRUE common tangent.

    The Newton residuals of `three_phase_equilibrium` encode stationarity
    only — spurious roots exist where the line is tangent locally but cuts
    through a curve elsewhere (measured on Al-Pb: a stationary
    configuration 100 K above the true monotectic, rejected by this check
    with gap −1.2e-2 eV). A root is a physical invariant iff the minimum
    signed gap min_c [G_φ(c,T) − line(c)] is ≥ −tol for EVERY phase curve.

    Args:
        curves:   iterable of FreeEnergyFn (each phase present anywhere
                  on the diagram, not only the three tangent phases).
        c_points: (c1, c2, c3) from the invariant solve; the line is
                  anchored at c1 on curves[anchor].
        T:        invariant temperature.
        anchor:   index into `curves` of the phase c_points[0] belongs
                  to (default 0). MISPAIRING THE ANCHOR BUILDS A
                  NONSENSE LINE and can spuriously reject a true
                  invariant (measured: −4e-2 on a verified Ni-Al
                  eutectic anchored on the wrong curve) — pass
                  curves in solve order or set `anchor` explicitly.

    Returns:
        float: the worst (most negative) gap over all curves — host-side
        diagnostic, not differentiable (use in verification, not in loss).
    """
    import numpy as np

    c1 = float(c_points[0])
    G0 = curves[anchor]
    mu = float(jax.grad(G0, argnums=0)(jnp.asarray(c1), jnp.asarray(T)))
    g0 = float(G0(jnp.asarray(c1), jnp.asarray(T)))
    cs = np.linspace(1e-4, 1.0 - 1e-4, n_grid)
    worst = np.inf
    for G in curves:
        vals = np.array([float(G(jnp.asarray(c), jnp.asarray(T)))
                         for c in cs])
        gap = vals - (g0 + mu * (cs - c1))
        worst = min(worst, float(gap.min()))
    return worst


def classify_invariant(kinds: tuple[str, str, str]) -> str:
    """Name the three-phase reaction from the phase kinds in tangent order
    (c₁ < c₂ < c₃). `kinds` entries are "solid" or "liquid".

    The topology is decided by which kind sits at the MIDDLE tangent point:
    the middle phase is the one that decomposes into (or forms from) the
    outer two as the invariant line is crossed.
    """
    k1, k2, k3 = (k.lower() for k in kinds)
    n_liq = sum(k == "liquid" for k in (k1, k2, k3))
    if n_liq == 0:
        return "eutectoid-type (all-solid)"
    if n_liq == 1:
        return "eutectic" if k2 == "liquid" else "peritectic"
    if n_liq == 2:
        return "monotectic" if k2 == "liquid" else "syntectic"
    return "three-liquid (unphysical)"


# ── Batched tangent solves (whole-boundary Jacobians in one pass) ─────────────

def tangent_sweep(
    G_alpha: FreeEnergyFn,
    G_beta: FreeEnergyFn,
    Ts: jnp.ndarray,
    seeds_alpha: jnp.ndarray,
    seeds_beta: jnp.ndarray,
    n_iter: int = 60,
    damping: float = 1.0,
):
    """vmapped common-tangent solve over a temperature array.

    Two-pass pattern for differentiable whole-boundary sweeps: run the
    host-side continuation sweep (`eutectic_diagram`) ONCE forward to get
    branch-tracking seeds, then call this with those seeds — one vmapped,
    fully traced solve per boundary. jax.jacobian of the output w.r.t.
    anything the curves close over gives ∂(entire liquidus)/∂(Ω, engine R, …)
    in a single backward pass.

    Returns (c_alpha(T), c_beta(T)) arrays, differentiable.
    """
    def solve_one(T, ca0, cb0):
        eps = 1e-4
        ca0 = jnp.clip(ca0, eps, 1 - eps)
        cb0 = jnp.clip(cb0, eps, 1 - eps)
        return common_tangent(G_alpha, G_beta, T,
                              c_alpha_guess=ca0, c_beta_guess=cb0,
                              n_iter=n_iter, damping=damping)

    return jax.vmap(solve_one)(jnp.asarray(Ts),
                               jnp.asarray(seeds_alpha),
                               jnp.asarray(seeds_beta))


# ── Full binary eutectic diagram (host-side sweep over the solvers) ──────────

def eutectic_diagram(
    G_solid: FreeEnergyFn,
    G_liquid: FreeEnergyFn,
    T_m_A: float,
    T_m_B: float,
    n_T: int = 60,
    T_floor: float | None = None,
):
    """Assemble liquidus/solidus (both limbs), solvus, and the eutectic point.

    Returns a dict of numpy-convertible arrays for plotting. This is a
    convenience sweep for figures — for gradients, differentiate
    `eutectic_point` / `common_tangent` directly.
    """
    import numpy as np

    ca_e, cl_e, cb_e, T_e = eutectic_point(G_solid, G_liquid)
    T_e_f = float(T_e)

    # α–L limb: from just above T_e to T_m_A;  β–L limb: to T_m_B.
    out = {"T_e": T_e_f, "c_alpha_e": float(ca_e),
           "c_liq_e": float(cl_e), "c_beta_e": float(cb_e)}

    def limb(T_hi, side):
        """Sweep from T_e up to T_m with CONTINUATION: each solve is seeded by
        the previous solution, so the tangent tracks its own branch even as
        both points collapse toward the pure end at T_m."""
        Ts = np.linspace(T_e_f + 1.0, T_hi - 2.0, n_T)
        sol, liq = [], []
        if side == "A":
            gs, gl = float(ca_e), float(cl_e)
        else:
            gl, gs = float(cl_e), float(cb_e)
        for T in Ts:
            eps = 1e-4
            if side == "A":
                cs, cl = common_tangent(
                    G_solid, G_liquid, float(T),
                    c_alpha_guess=float(np.clip(gs, eps, 1 - eps)),
                    c_beta_guess=float(np.clip(gl, eps, 1 - eps)))
            else:
                cl, cs = common_tangent(
                    G_liquid, G_solid, float(T),
                    c_alpha_guess=float(np.clip(gl, eps, 1 - eps)),
                    c_beta_guess=float(np.clip(gs, eps, 1 - eps)))
            gs, gl = float(cs), float(cl)
            sol.append(gs); liq.append(gl)
        return Ts, np.array(sol), np.array(liq)

    out["T_A"], out["solidus_A"], out["liquidus_A"] = limb(T_m_A, "A")
    out["T_B"], out["solidus_B"], out["liquidus_B"] = limb(T_m_B, "B")

    # Solvus: solid–solid gap below T_e, continuation downward from (c_α, c_β).
    T_lo = T_floor if T_floor is not None else 0.35 * T_e_f
    Ts = np.linspace(T_e_f - 1.0, T_lo, n_T)
    sa, sb = [], []
    g1, g2 = float(ca_e), float(cb_e)
    for T in Ts:
        eps = 1e-4
        c1, c2 = common_tangent(
            G_solid, G_solid, float(T),
            c_alpha_guess=float(np.clip(g1, eps, 1 - eps)),
            c_beta_guess=float(np.clip(g2, eps, 1 - eps)))
        g1, g2 = float(c1), float(c2)
        sa.append(g1); sb.append(g2)
    out["T_solvus"] = Ts
    out["solvus_A"] = np.array(sa)
    out["solvus_B"] = np.array(sb)
    return out
