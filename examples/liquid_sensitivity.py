"""
Is the peritectic-gap missing-physics attribution (Cu-Zn 37%, Cu-Sn 86%)
an artifact of assuming an *ideal* liquid?

The reference gaps against which Debye/quasichemical closures are quoted
(Sections 4.7, 4.8) are measured against an fcc-liquid common tangent
built with an ideal liquid (Omega_L = 0). A skeptical reading: a real,
attractive metallic melt (Omega_L < 0) might move that tangent enough to
close the gap on its own, making the attribution soft.

This script tests it directly -- the sensitivity d(gap)/d(Omega_L) is one
derivative through the same tangent construction the paper already
differentiates. Result: the gap is insensitive to Omega_L, and an
*attractive* liquid makes it slightly *larger*, not smaller (a more
competitive liquid is harder for a solid beta to intercept). The ideal-
liquid reference is therefore conservative, not a hidden gap-closing
mechanism. Zero engine calls; pure JAX on stored energetics.

Output: examples/output/liquid_sensitivity.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from coexist.core.phase_diagram import (  # noqa: E402
    K_B, common_tangent, liquid_solution)

OUT = Path(__file__).parent / "output"
CG = jnp.linspace(1e-4, 1.0 - 1e-4, 4001)
H = 0.005  # eV, central-difference step on Omega_L


def rk(L, c):
    return sum(L[k] * (1.0 - 2.0 * c) ** k for k in range(len(L)))


def gap_and_tangent(G_fcc, G_bcc, fus, T, OmL, seed):
    G_L = liquid_solution(OmL, dH_fus_A=fus[0], T_m_A=fus[1],
                          dH_fus_B=fus[2], T_m_B=fus[3])
    cf, cl = common_tangent(G_fcc, G_L, float(T),
                            c_alpha_guess=seed[0], c_beta_guess=seed[1])
    s = (G_L(cl, T) - G_fcc(cf, T)) / (cl - cf)
    line = G_fcc(cf, T) + s * (CG - cf)
    return float(jnp.min(G_bcc(CG, T) - line)), float(cl)


def sensitivity(G_fcc, G_bcc, fus, T, seed):
    g0, cl = gap_and_tangent(G_fcc, G_bcc, fus, T, 0.0, seed)
    gp, _ = gap_and_tangent(G_fcc, G_bcc, fus, T, +H, seed)
    gm, _ = gap_and_tangent(G_fcc, G_bcc, fus, T, -H, seed)
    dg = (gp - gm) / (2 * H)
    grid = {f"{o:+.2f}": gap_and_tangent(G_fcc, G_bcc, fus, T, o, seed)[0] * 1e3
            for o in (-0.15, -0.10, -0.05, 0.0)}
    return dict(gap0_meV=g0 * 1e3, c_L_tangent=cl,
                dgap_dOmL=dg, gap_vs_OmL_meV=grid)


out = {}

# ---- Cu-Sn (assessed peritectic 1071.15 K; Debye closes 86% of gap0) ----
d = json.load(open(OUT / "cusn_peritectic.json"))
dSf, dCb, dSb = (float(d["dE_Sn_fcc_minus_betaSn_eV"]),
                 float(d["dE_Cu_bcc_minus_fcc_eV"]),
                 float(d["dE_Sn_bcc_minus_betaSn_eV"]))
Lf, Lb = d["L_fcc_eV"], d["L_bcc_eV"]
G_fcc = lambda c, T: c * dSf + c * (1 - c) * rk(Lf, c) + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c))
G_bcc = lambda c, T: (1 - c) * dCb + c * dSb + c * (1 - c) * rk(Lb, c) + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c))
out["CuSn"] = sensitivity(G_fcc, G_bcc,
                          (0.13742, 1357.77, 0.07286, 505.08), 1071.15, (0.05, 0.5))
out["CuSn"]["dF_vib_bcc_minus_fcc_meV"] = -18.63

# ---- Cu-Zn (assessed peritectic 1176 K; Debye+SRO close 37% of gap0) ----
d = json.load(open(OUT / "cuzn_peritectic.json"))
dCb, dZb, dZf = (float(d["dE_Cu_bcc_minus_fcc_eV"]),
                 float(d["dE_Zn_bcc_minus_hcp_eV"]),
                 float(d["dE_Zn_fcc_minus_hcp_eV"]))
Lf, Lb = d["L_fcc_eV"], d["L_bcc_eV"]
G_fcc = lambda c, T: c * dZf + c * (1 - c) * rk(Lf, c) + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c))
G_bcc = lambda c, T: (1 - c) * dCb + c * dZb + c * (1 - c) * rk(Lb, c) + K_B * T * (c * jnp.log(c) + (1 - c) * jnp.log(1 - c))
out["CuZn"] = sensitivity(G_fcc, G_bcc,
                          (0.13742, 1357.77, 0.07000, 692.68), 1176.0, (0.7, 0.95))

for sys, r in out.items():
    print(f"{sys}: gap0={r['gap0_meV']:+.2f} meV  c_L={r['c_L_tangent']:.3f}  "
          f"dgap/dOmL={r['dgap_dOmL']:+.3f}  grid={r['gap_vs_OmL_meV']}")

with open(OUT / "liquid_sensitivity.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"\nWrote {OUT / 'liquid_sensitivity.json'}")
