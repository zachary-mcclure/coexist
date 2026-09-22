"""
The low-T Ag-Cu-Ni tie-triangle by BLOCK decomposition.

At the converged 108-atom energetics the three-phase field exists only
below T_c^CuNi = 284 K, where the corner solubilities reach
x_Ag ~ 1e-15 (Cu-rich vertex) and ~1e-21 (Ni-rich vertex). Two hard
lessons, one dimension up from the binary float64 finding:

  1. The coupled 6x6 Newton cannot reach this triangle: mid-simplex and
     corner seeds stall or nan (residual-checked), and one stalled
     output passes the tangency gap while another is rejected at
     -4.3e-2 eV -- residual AND gap must both be reported.
  2. Below x ~ 1e-14 the composition cannot even be REPRESENTED in the
     simplex chart: x_A = 1 - x_B - x_C is subtractive, so float64
     returns 0 and mu_A = log(0). The chart, not the arithmetic, is
     the limit.

The construction that works: solve the well-conditioned blocks
  (a) the Cu-Ni binodal pair (1D, exact),
  (b) the Ag admixtures of the pair and the Ag-vertex solutes from the
      dilute-limit mu closures (asymptotically exact at x ~ 1e-15),
then VERIFY mu equality in an unconstrained three-component chart where
the tiny x_A is an explicit input (no subtraction), plus the simplex
tangency gap of the vertex plane.

Pure JAX/NumPy on stored engine numbers - zero engine calls.
Output: examples/output/ternary_triangle_250K.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from coexist.core.ternary import (  # noqa: E402
    plane_tangency_gap, ternary_regular_solution)

OUT = Path(__file__).parent / "output"
K_B = 8.617333e-5

t = json.load(open(OUT / "ternary_agcuni.json"))
mr = json.load(open(OUT / "mace_rung.json"))
O_AB = mr["AgCu"]["omega_eV"]       # Ag-Cu
O_BC = mr["CuNi"]["omega_eV"]       # Cu-Ni
O_AC = t["omega_AgNi_eV"]           # Ag-Ni
L = t["L_ABC_eV"]
T = 250.0
kT = K_B * T
print(f"engine inputs: O_AgCu={O_AB*1e3:+.0f}, O_AgNi={O_AC*1e3:+.0f}, "
      f"O_CuNi={O_BC*1e3:+.0f} meV, L_ABC={L*1e3:+.0f} meV, T={T:.0f} K")


# ── unconstrained-chart chemical potentials: x = (xA, xB, xC) explicit ──────
def g_molar(x):
    xA, xB, xC = x
    ex = (O_AB * xA * xB + O_AC * xA * xC + O_BC * xB * xC
          + L * xA * xB * xC)
    ent = kT * (xA * jnp.log(xA) + xB * jnp.log(xB) + xC * jnp.log(xC))
    return ex + ent


def mus_unconstrained(x):
    """mu_i = g + dg/dx_i - sum_j x_j dg/dx_j (exact for a molar g)."""
    x = jnp.asarray(x, dtype=jnp.float64)
    g = g_molar(x)
    dg = jax.grad(g_molar)(x)
    corr = jnp.dot(x, dg)
    return np.array(g + dg - corr)


# ── block (a): Cu-Ni binodal pair (1D, exact) ───────────────────────────────
x_lo = brentq(lambda x: np.log(x / (1 - x)) - (O_BC / kT) * (2 * x - 1),
              1e-6, 0.499, xtol=1e-15)
print(f"Cu-Ni binodal pair: x = {x_lo:.6f} / {1-x_lo:.6f}")

# ── block (b): dilute-limit closures, then 1D Newton polish on each ─────────
# Ag admixture of a BC phase from mu_A equality with the Ag vertex
# (mu_A(vertex) ~ 0); solute contents of the Ag vertex from mu_B/mu_C.
def xA_of_pair(xB, xC):
    xA = np.exp(-(O_AB * xB + O_AC * xC - O_BC * xB * xC + L * xB * xC)
                / kT)
    for _ in range(4):                      # 1D Newton on log xA
        m = mus_unconstrained((xA, xB * (1 - xA), xC * (1 - xA)))
        xA *= np.exp(-m[0] / kT)
    return xA


xA2 = xA_of_pair(1 - x_lo, x_lo)            # Cu-rich vertex
xA3 = xA_of_pair(x_lo, 1 - x_lo)            # Ni-rich vertex
v2 = np.array([xA2, (1 - x_lo) * (1 - xA2), x_lo * (1 - xA2)])
v3 = np.array([xA3, x_lo * (1 - xA3), (1 - x_lo) * (1 - xA3)])
m2 = mus_unconstrained(v2)
m3 = mus_unconstrained(v3)

xB1 = np.exp((0.5 * (m2[1] + m3[1]) - O_AB) / kT)
xC1 = np.exp((0.5 * (m2[2] + m3[2]) - O_AC) / kT)
for _ in range(4):                          # polish the Ag vertex
    v1 = np.array([1 - xB1 - xC1, xB1, xC1])
    m1 = mus_unconstrained(v1)
    xB1 *= np.exp(-(m1[1] - 0.5 * (m2[1] + m3[1])) / kT)
    xC1 *= np.exp(-(m1[2] - 0.5 * (m2[2] + m3[2])) / kT)
v1 = np.array([1 - xB1 - xC1, xB1, xC1])
m1 = mus_unconstrained(v1)

res = max(float(np.max(np.abs(m1 - m2))), float(np.max(np.abs(m1 - m3))),
          float(np.max(np.abs(m2 - m3))))
print("vertices (x_Ag, x_Cu, x_Ni):")
print(f"  Ag      : ({v1[0]:.8f}, {v1[1]:.3e}, {v1[2]:.3e})")
print(f"  Cu-rich : ({v2[0]:.3e}, {v2[1]:.6f}, {v2[2]:.6f})")
print(f"  Ni-rich : ({v3[0]:.3e}, {v3[1]:.6f}, {v3[2]:.6f})")
print(f"max |d mu| across all pairs/components = {res:.2e} eV "
      f"(unconstrained chart)")

# ── simplex verifier: plane at the Ag vertex over the (xB, xC) chart ────────
G = ternary_regular_solution(O_AB, O_AC, O_BC, L_ABC=L)
gap = float(plane_tangency_gap((G,), jnp.array([v1[1], v1[2]]), G, T,
                               n_grid=150))
print(f"simplex tangency gap (plane at Ag vertex) = {gap:+.2e} eV → "
      f"{'VERIFIED' if gap > -1e-6 else 'REJECTED'}")

json.dump(dict(T=T, v1=v1.tolist(), v2=v2.tolist(), v3=v3.tolist(),
               mu_res_eV=res, gap_eV=gap, binodal_x=x_lo),
          open(OUT / "ternary_triangle_250K.json", "w"), indent=1)
print(f"Wrote {OUT/'ternary_triangle_250K.json'}")
