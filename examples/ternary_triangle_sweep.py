"""
T-continuation of the Ag-Cu-Ni tie-triangle by block decomposition (paper
review pass): the SAME block solve as `ternary_triangle_blocks.py`, at a
handful of temperatures approaching the Cu-Ni consolute point, to make the
collapse mechanism (Sec. 4.10: "the collapsed point marking the limiting
tie-line") visible as a strip rather than described only at two isolated
snapshots (250 K triangle, 800 K section).

Pure JAX/NumPy on stored engine numbers (ternary_agcuni.json,
mace_rung.json) -- zero engine calls, seconds to run.
Output: examples/output/ternary_triangle_sweep.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

OUT = Path(__file__).parent / "output"
K_B = 8.617333e-5

t = json.load(open(OUT / "ternary_agcuni.json"))
mr = json.load(open(OUT / "mace_rung.json"))
O_AB = mr["AgCu"]["omega_eV"]
O_BC = mr["CuNi"]["omega_eV"]
O_AC = t["omega_AgNi_eV"]
L = t["L_ABC_eV"]
T_C = O_BC / (2 * K_B)
print(f"engine inputs: O_AgCu={O_AB*1e3:+.0f}, O_AgNi={O_AC*1e3:+.0f}, "
      f"O_CuNi={O_BC*1e3:+.0f} meV, L_ABC={L*1e3:+.0f} meV, "
      f"T_c^CuNi={T_C:.1f} K")


def g_molar(x):
    xA, xB, xC = x
    ex = (O_AB * xA * xB + O_AC * xA * xC + O_BC * xB * xC
          + L * xA * xB * xC)
    ent = K_B * 1.0 * (xA * jnp.log(xA) + xB * jnp.log(xB)
                        + xC * jnp.log(xC))
    return ex + ent


def solve_triangle(T):
    kT = K_B * T

    def g_molar_T(x):
        xA, xB, xC = x
        ex = (O_AB * xA * xB + O_AC * xA * xC + O_BC * xB * xC
              + L * xA * xB * xC)
        ent = kT * (xA * jnp.log(xA) + xB * jnp.log(xB) + xC * jnp.log(xC))
        return ex + ent

    def mus_unconstrained(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        g = g_molar_T(x)
        dg = jax.grad(g_molar_T)(x)
        corr = jnp.dot(x, dg)
        return np.array(g + dg - corr)

    x_lo = brentq(lambda x: np.log(x / (1 - x)) - (O_BC / kT) * (2 * x - 1),
                  1e-6, 0.499999, xtol=1e-15)

    def xA_of_pair(xB, xC):
        xA = np.exp(-(O_AB * xB + O_AC * xC - O_BC * xB * xC + L * xB * xC)
                    / kT)
        for _ in range(6):
            m = mus_unconstrained((xA, xB * (1 - xA), xC * (1 - xA)))
            xA *= np.exp(-m[0] / kT)
        return xA

    xA2 = xA_of_pair(1 - x_lo, x_lo)
    xA3 = xA_of_pair(x_lo, 1 - x_lo)
    v2 = np.array([xA2, (1 - x_lo) * (1 - xA2), x_lo * (1 - xA2)])
    v3 = np.array([xA3, x_lo * (1 - xA3), (1 - x_lo) * (1 - xA3)])
    m2 = mus_unconstrained(v2)
    m3 = mus_unconstrained(v3)

    xB1 = np.exp((0.5 * (m2[1] + m3[1]) - O_AB) / kT)
    xC1 = np.exp((0.5 * (m2[2] + m3[2]) - O_AC) / kT)
    for _ in range(6):
        v1 = np.array([1 - xB1 - xC1, xB1, xC1])
        m1 = mus_unconstrained(v1)
        xB1 *= np.exp(-(m1[1] - 0.5 * (m2[1] + m3[1])) / kT)
        xC1 *= np.exp(-(m1[2] - 0.5 * (m2[2] + m3[2])) / kT)
    v1 = np.array([1 - xB1 - xC1, xB1, xC1])

    collapse = float(np.max(np.abs(v2 - v3)))
    return dict(T=T, v1=v1.tolist(), v2=v2.tolist(), v3=v3.tolist(),
                collapse_gap=collapse, binodal_x=float(x_lo))


Ts = [250.0, 264.0, 274.0, 280.0]
frames = [solve_triangle(T) for T in Ts]
for f in frames:
    print(f"  T={f['T']:.0f} K  binodal x={f['binodal_x']:.4f}  "
          f"|v2-v3|={f['collapse_gap']:.2e}")

json.dump(dict(T_c_CuNi=T_C, frames=frames),
          open(OUT / "ternary_triangle_sweep.json", "w"), indent=1)
print(f"Wrote {OUT/'ternary_triangle_sweep.json'}")
