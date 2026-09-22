"""
Ag-Cu Eutectic Phase Diagram from an External Engine — Differentiable Throughout

The classic eutectic construction — solid and liquid Gibbs curves, common
tangents, the triple-tangent eutectic point — computed with the solid
interaction parameter Ω_s taken LIVE from an external engine (EMT via
SimulatorAdapter; identical code path for LAMMPS/VASP), and differentiated:

    ∂T_eutectic/∂Ω_s      (implicit, through the 4×4 Newton solve)
    ∂T_eutectic/∂R_atoms  (all the way into the engine, one backward pass)

Thermodynamic model (per-atom eV, solid reference states):
    G_s(c,T) = Ω_s·c(1−c) + kT·[c ln c + (1−c) ln(1−c)]
    G_L(c,T) = (1−c)·ΔG_m,Ag + c·ΔG_m,Cu + Ω_L·c(1−c) + kT·[ideal]
    ΔG_m,i(T) = ΔH_m,i·(1 − T/T_m,i)

Inputs:
    Ω_s   — EMT, 32-atom periodic fcc supercells (this script computes it)
    Ω_L   — CALPHAD liquid assessment (Witusiewicz et al., J. Alloys Compd.
            2004: L0_liq ≈ +15 kJ/mol → 0.155 eV/atom), literature value
    ΔH_m, T_m — experimental fusion data (Ag: 11.28 kJ/mol, 1234.9 K;
            Cu: 13.26 kJ/mol, 1357.8 K)

Experimental benchmark (Massalski): T_e = 1052 K, x_Cu(eutectic) = 0.399,
solid solubilities at T_e: c_α ≈ 0.141 (Cu in Ag), c_β ≈ 0.951 (Ag in Cu).

Run:
    cd /Users/zmcclure/Coexist && source .venv/bin/activate
    python examples/eutectic_phase_diagram.py
"""
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

from ase.calculators.emt import EMT
from ase.build import bulk

from coexist.adapters import ASEAdapter
from coexist.core.phase_diagram import (
    K_B, regular_solution, liquid_solution,
    common_tangent, eutectic_point, eutectic_diagram,
)

# Fusion data (experimental)
T_M_AG, DH_AG = 1234.93, 0.11691   # K, eV/atom (11.28 kJ/mol)
T_M_CU, DH_CU = 1357.77, 0.13742   # K, eV/atom (13.26 kJ/mol)
OMEGA_L = 0.1555                    # eV/atom — CALPHAD liquid L0 (+15 kJ/mol)

# Experimental eutectic (Massalski binary alloy phase diagrams)
T_E_EXP, C_E_EXP = 1052.0, 0.399
C_ALPHA_EXP, C_BETA_EXP = 0.141, 0.951

print("=" * 68)
print("  Ag-Cu EUTECTIC phase diagram — Ω_s from an external engine,")
print("  every boundary differentiable (c = x_Cu)")
print("=" * 68)

# ── Step 1: Ω_s from the external engine ─────────────────────────────────────
print(f"\n{'─'*68}")
print("  STEP 1 — Ω_solid from EMT (32-atom periodic fcc supercells)")
print(f"{'─'*68}")

A_AG, A_CU = 4.09, 3.61
A_MIX = 0.5 * (A_AG + A_CU)

ag  = bulk("Ag", "fcc", a=A_AG,  cubic=True).repeat((2, 2, 2))
cu  = bulk("Cu", "fcc", a=A_CU,  cubic=True).repeat((2, 2, 2))
mix = bulk("Ag", "fcc", a=A_MIX, cubic=True).repeat((2, 2, 2))
N_AT = len(mix)
rng = np.random.default_rng(0)
cu_sites = rng.choice(N_AT, N_AT // 2, replace=False)
sym = np.array(mix.get_chemical_symbols())
sym[cu_sites] = "Cu"
mix.set_chemical_symbols(sym.tolist())
POS = np.asarray(mix.get_positions(), dtype=np.float64)
X_CU = 0.5

eng_ag  = ASEAdapter(ag.get_atomic_numbers().tolist(),  lambda: EMT(),
                     cell=ag.get_cell()[:],  pbc=True, name="EMT")
eng_cu  = ASEAdapter(cu.get_atomic_numbers().tolist(),  lambda: EMT(),
                     cell=cu.get_cell()[:],  pbc=True, name="EMT")
eng_mix = ASEAdapter(mix.get_atomic_numbers().tolist(), lambda: EMT(),
                     cell=mix.get_cell()[:], pbc=True, name="EMT")

E_ag = eng_ag(jnp.array(ag.get_positions()))
E_cu = eng_cu(jnp.array(cu.get_positions()))


def omega_s_from_R(pos):
    dH = eng_mix(pos) / N_AT - X_CU * E_cu / N_AT - (1 - X_CU) * E_ag / N_AT
    return dH / (X_CU * (1 - X_CU))


omega_s = omega_s_from_R(jnp.array(POS))
om_f = float(omega_s)
print(f"  Ω_s(EMT) = {om_f*1e3:+.1f} meV/atom  "
      f"(CALPHAD fcc Ag-Cu L0 ≈ +33-36 kJ/mol = 0.34-0.37 eV — EMT is soft)")
print(f"  Ω_L      = {OMEGA_L*1e3:+.1f} meV/atom  (CALPHAD literature)")

# ── Step 2: eutectic point (differentiable 4×4 Newton) ───────────────────────
print(f"\n{'─'*68}")
print("  STEP 2 — Eutectic point: one line tangent to three curves")
print(f"{'─'*68}")

G_L = liquid_solution(OMEGA_L, dH_fus_A=DH_AG, T_m_A=T_M_AG,
                      dH_fus_B=DH_CU, T_m_B=T_M_CU)


def eutectic_from_omega(om):
    G_s = regular_solution(om)
    return eutectic_point(G_s, G_L)


t0 = time.perf_counter()
ca, cl, cb, Te = eutectic_from_omega(omega_s)
t_e = time.perf_counter() - t0
ca_f, cl_f, cb_f, Te_f = map(float, (ca, cl, cb, Te))

print(f"  {'':16s}{'model':>10s}{'experiment':>12s}")
print(f"  {'T_e [K]':16s}{Te_f:>10.0f}{T_E_EXP:>12.0f}")
print(f"  {'x_Cu eutectic':16s}{cl_f:>10.3f}{C_E_EXP:>12.3f}")
print(f"  {'c_α (Cu in Ag)':16s}{ca_f:>10.3f}{C_ALPHA_EXP:>12.3f}")
print(f"  {'c_β (Ag in Cu)':16s}{cb_f:>10.3f}{C_BETA_EXP:>12.3f}")
print(f"  solve time: {t_e*1e3:.0f} ms")

# ── Step 3: gradients of the eutectic point ──────────────────────────────────
print(f"\n{'─'*68}")
print("  STEP 3 — Differentiate the eutectic point")
print(f"{'─'*68}")

dTe_dOm = jax.grad(lambda om: eutectic_from_omega(om)[3])(omega_s)
h = 1e-4
fd = (float(eutectic_from_omega(omega_s + h)[3])
      - float(eutectic_from_omega(omega_s - h)[3])) / (2 * h)
print(f"  ∂T_e/∂Ω_s = {float(dTe_dOm):+.1f} K/eV   (FD: {fd:+.1f}, "
      f"rel.err {abs(float(dTe_dOm)-fd)/abs(fd):.2e})")
print(f"    → +10 meV of solid repulsion deepens the eutectic by "
      f"{-float(dTe_dOm)*0.01:.1f} K")

dce_dOm = jax.grad(lambda om: eutectic_from_omega(om)[1])(omega_s)
print(f"  ∂c_e/∂Ω_s = {float(dce_dOm):+.3f} /eV")

t0 = time.perf_counter()
eng_mix.n_calls = 0
dTe_dR = jax.grad(lambda p: eutectic_from_omega(omega_s_from_R(p))[3])(jnp.array(POS))
t_g = time.perf_counter() - t0
mag = np.linalg.norm(np.array(dTe_dR), axis=1)
print(f"  ∂T_e/∂R_atoms (through the ENGINE): max |∂T_e/∂R| = {mag.max():.1f} K/Å "
      f"at atom {int(mag.argmax())} ({sym[int(mag.argmax())]})")
print(f"    [{t_g:.1f} s, {eng_mix.n_calls} engine call(s); FD would need "
      f"{2*N_AT*3}]")

# ── Step 4: full diagram sweep + figure ──────────────────────────────────────
print(f"\n{'─'*68}")
print("  STEP 4 — Full diagram sweep (liquidus / solidus / solvus)")
print(f"{'─'*68}")

G_s = regular_solution(omega_s)
t0 = time.perf_counter()
d = eutectic_diagram(G_s, G_L, T_m_A=T_M_AG, T_m_B=T_M_CU, n_T=50,
                     T_floor=500.0)
print(f"  swept {3*50} tangent solves in {time.perf_counter()-t0:.1f} s")

BLUE, ORANGE, GREY = "#2E5FA3", "#D9782D", "#666666"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.4))

# Panel (a): G curves + triple tangent at T_e
cg = np.linspace(1e-4, 1 - 1e-4, 500)
gs = np.array([float(G_s(c, Te_f)) for c in cg])
gl = np.array([float(G_L(c, Te_f)) for c in cg])
mu_e = float(jax.grad(G_s, argnums=0)(ca, Te))
tang = float(G_s(ca, Te)) + mu_e * (cg - ca_f)
ax1.plot(cg, gs * 1e3, color=BLUE, lw=2, label="$G_{solid}$")
ax1.plot(cg, gl * 1e3, color=ORANGE, lw=2, label="$G_{liquid}$")
ax1.plot(cg, tang * 1e3, ":", color=GREY, lw=1.5, label="triple tangent")
for c_pt, G_pt, col in [(ca_f, G_s, BLUE), (cb_f, G_s, BLUE), (cl_f, G_L, ORANGE)]:
    ax1.plot([c_pt], [float(G_pt(c_pt, Te_f)) * 1e3], "o", color=col, ms=7,
             mec="white", mew=1)
ax1.set_xlabel("$x_{Cu}$")
ax1.set_ylabel("G  [meV/atom]")
ax1.set_title(f"(a) One line tangent to three curves at $T_e$ = {Te_f:.0f} K\n"
              f"$\\Omega_s$ = {om_f*1e3:+.0f} meV (engine), "
              f"$\\Omega_L$ = {OMEGA_L*1e3:+.0f} meV (CALPHAD)", fontsize=10)
ax1.legend(fontsize=8, loc="upper center")
ax1.grid(alpha=0.25, lw=0.5)

# Panel (b): full T-x diagram
ax2.plot(d["liquidus_A"], d["T_A"], color=ORANGE, lw=2.2, label="liquidus")
ax2.plot(d["liquidus_B"], d["T_B"], color=ORANGE, lw=2.2)
ax2.plot(d["solidus_A"], d["T_A"], color=BLUE, lw=2.2, label="solidus")
ax2.plot(d["solidus_B"], d["T_B"], color=BLUE, lw=2.2)
ax2.plot(d["solvus_A"], d["T_solvus"], color=BLUE, lw=1.6, ls="--",
         label="solvus (α+β gap)")
ax2.plot(d["solvus_B"], d["T_solvus"], color=BLUE, lw=1.6, ls="--")
# eutectic isotherm
ax2.plot([ca_f, cb_f], [Te_f, Te_f], color=GREY, lw=1.2)
ax2.plot([cl_f], [Te_f], "o", color="#333333", ms=8, mec="white", mew=1,
         label=f"eutectic (model): {Te_f:.0f} K, $x_{{Cu}}$={cl_f:.2f}")
ax2.plot([C_E_EXP], [T_E_EXP], "*", color="#B03A48", ms=16, mec="white", mew=0.8,
         label=f"eutectic (exp.): {T_E_EXP:.0f} K, $x_{{Cu}}$={C_E_EXP:.2f}")
ax2.plot([0], [T_M_AG], "^", color=GREY, ms=7, mec="white")
ax2.plot([1], [T_M_CU], "^", color=GREY, ms=7, mec="white")
ax2.annotate(f"$T_m$(Ag)", (0.01, T_M_AG), fontsize=8, color=GREY)
ax2.annotate(f"$T_m$(Cu)", (0.88, T_M_CU), fontsize=8, color=GREY)
ax2.set_xlabel("$x_{Cu}$")
ax2.set_ylabel("T  [K]")
ax2.set_xlim(0, 1)
ax2.set_title("(b) Ag-Cu eutectic diagram — every boundary differentiable\n"
              f"$\\partial T_e/\\partial\\Omega_s$ = {float(dTe_dOm):+.0f} K/eV; "
              f"$\\partial T_e/\\partial R$ through the engine", fontsize=10)
ax2.legend(fontsize=8, loc="lower left", framealpha=0.92)
ax2.grid(alpha=0.25, lw=0.5)

fig.suptitle("Eutectic construction as a differentiable program: "
             "engine energetics → common tangents → $T_e$, with gradients",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "eutectic_01_agcu.png", dpi=150, bbox_inches="tight")
print(f"  Saved {OUT/'eutectic_01_agcu.png'}")

print(f"\n{'='*68}")
print("  SUMMARY")
print(f"{'='*68}")
print(f"  Ω_s(engine) = {om_f*1e3:+.1f} meV | T_e = {Te_f:.0f} K (exp {T_E_EXP:.0f}) "
      f"| x_e = {cl_f:.3f} (exp {C_E_EXP})")
print(f"  ∂T_e/∂Ω_s = {float(dTe_dOm):+.1f} K/eV (FD-verified) | "
      f"∂T_e/∂R: 1 engine call")
