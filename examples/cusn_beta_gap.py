"""
Cu-Sn missing-physics attribution, part 1: Debye-Grueneisen on the two
lattices -- the same measurement run for Cu-Zn (examples/
cuzn_peritectic_scan.py), applied here so Cu-Sn's "the same missing-physics
story likely applies here too, not attempted in this pass" (paper Sec. 4.9) becomes a computed number instead of an assertion.

cusn_peritectic_scan.py established: bcc sits +20.6 to +24.2 meV above the
fcc-liquid tangent line, 750-1300 K, at 0 K RK-fit enthalpy alone. This
script asks how much of that gap Debye vibrations close, using MACE-MP-0
E(V) scans on the SAME compositions (x_Sn=8/108 fcc, 17/128 bcc) already
used for the stored mixing energetics in cusn_peritectic.json, so the
lattice-constant/composition pairing is internally consistent with the
rest of the Cu-Sn construction rather than a fresh, disconnected guess.

Runtime: MACE-MP-0 E(V) scans on two ~108-128 atom cells (9 volume points
each, internal-coordinate relaxation at every point) -- comparable cost to
the analogous Cu-Zn --debye path.
Output: examples/output/cusn_beta_gap.json
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from ase.build import bulk  # noqa: E402

from coexist.adapters.lammps_factory import eos_scan  # noqa: E402
from coexist.core.thermo_vib import (  # noqa: E402
    debye_temperature, eos_fit, f_vib_debye)

OUT = Path(__file__).parent / "output"
d = json.load(open(OUT / "cusn_peritectic.json"))
scan = json.load(open(OUT / "cusn_peritectic_scan.json"))
T_ASSESSED = d["assessed_T_K"]
M_CU, M_SN = 63.546, 118.710

print("Loading MACE-MP-0 (small, float64, CPU)...")
from mace.calculators import mace_mp  # noqa: E402

_calc = mace_mp(model="small", device="cpu", default_dtype="float64")
fac = lambda: _calc  # noqa: E731

# Same cells/compositions as cusn_peritectic.py's mixing-energetics scan
# (fcc x_Sn=8/108=0.0741, close to the assessed alpha 0.077; bcc
# x_Sn=17/128=0.1328, close to the assessed beta 0.13) -- reusing the
# already-relaxed lattice-constant Vegard guesses from that script.
CELLS = {
    "fcc": dict(a=(1 - 8 / 108) * 3.61 + (8 / 108) * 4.30,
               lat="fcc", reps=(3, 3, 3), n_sub=8),
    "bcc": dict(a=(1 - 17 / 128) * 2.88 + (17 / 128) * 3.30,
               lat="bcc", reps=(4, 4, 4), n_sub=17),
}

THETA, MASS = {}, {}
for lat, spec in CELLS.items():
    print(f"-- {lat}: EOS scan ({spec['n_sub']} Sn substitutions)")
    at = bulk("Cu", spec["lat"], a=spec["a"], cubic=True).repeat(spec["reps"])
    n = len(at)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=spec["n_sub"], replace=False)
    sy = np.array(at.get_chemical_symbols(), dtype=object)
    sy[idx] = "Sn"
    at.set_chemical_symbols(list(sy))
    x_sn = spec["n_sub"] / n
    M = (1 - x_sn) * M_CU + x_sn * M_SN
    V, E = eos_scan(at, fac)
    V0, _, B0, _ = eos_fit(jnp.array(V), jnp.array(E))
    THETA[lat] = float(debye_temperature(V0, B0, M))
    MASS[lat] = M
    print(f"   x_Sn={x_sn:.4f}, M={M:.2f} amu, "
         f"V0={float(V0):.2f} A^3/atom, B0={float(B0):.3f} eV/A^3, "
         f"theta_D={THETA[lat]:.1f} K")

dF = float(f_vib_debye(T_ASSESSED, THETA["bcc"])
          - f_vib_debye(T_ASSESSED, THETA["fcc"]))
# The reference gap is the tangent-gap value AT the assessed invariant
# temperature specifically (interpolated from the scan), not the range
# minimum over 750-1300K -- these differ by ~1 meV here (21.7 vs 20.6),
# and the assessed-T value is the one the paper actually quotes as "the"
# gap for Cu-Zn's analogous attribution (28.0 meV at 1176K, not the range
# floor of 27.4 meV), so this mirrors that choice for consistency.
scan_Ts = [s["T"] for s in scan["scan"]]
scan_gaps_meV = [s["gap_bcc"] * 1e3 for s in scan["scan"]]
gap_at_T_meV = float(np.interp(T_ASSESSED, scan_Ts, scan_gaps_meV))
min_gap_meV = scan["min_gap_bcc_meV"]
print(f"\ntheta_D: fcc {THETA['fcc']:.0f} K, bcc {THETA['bcc']:.0f} K -> "
     f"F_vib(bcc)-F_vib(fcc) = {dF*1e3:+.1f} meV/atom at {T_ASSESSED:.0f} K")
print(f"gap to close at {T_ASSESSED:.0f} K specifically (interpolated from "
     f"cusn_peritectic_scan.json): +{gap_at_T_meV:.1f} meV "
     f"(range minimum 750-1300K: +{min_gap_meV:.1f} meV)")
if dF < 0:
    print(f"   correct sign (bcc-stabilizing): supplies "
         f"{abs(dF*1e3)/gap_at_T_meV*100:.0f}% of the demand at {T_ASSESSED:.0f} K")
else:
    print("   WRONG SIGN for bcc stabilization -- Debye alone does not "
         "help close this gap (reported as-is, not hidden)")

results = dict(
    theta_D_K=THETA, mass_amu=MASS,
    dF_vib_bcc_minus_fcc_meV=dF * 1e3,
    T_K=T_ASSESSED,
    gap_bcc_at_assessed_T_meV=gap_at_T_meV,
    min_gap_bcc_meV=min_gap_meV,
    fraction_of_gap_closed_by_debye=(-dF * 1e3 / gap_at_T_meV
                                     if dF < 0 else 0.0),
)
with open(OUT / "cusn_beta_gap.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'cusn_beta_gap.json'}")
