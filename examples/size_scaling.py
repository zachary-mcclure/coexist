"""
Supercell size scaling: how big can the laptop run, and is Ω
converged at 32 atoms?

Two questions a referee will ask of the 32-atom protocol:
  1. Finite-size: does the equimolar interaction Ω(x=½) = 4·ΔH_mix(½)
     move when the random supercell grows (32 → 108 → 256 → 500)?
     (EAM Williams Ag-Cu, the paper's flagship solid energetics; MACE
     additionally checks 32 → 108, where the ~10 Å receptive field wraps
     around a 7.3 Å box.)
  2. Cost: single energy+forces call and full relaxation timings per
     size, EAM and MACE-MP-0 (small, CPU, float64) — the practical
     ceiling of a laptop campaign.

Output: examples/output/size_scaling.json
"""
import json
import time
from pathlib import Path

import numpy as np

from coexist.adapters.lammps_factory import (
    eam_alloy_factory, fcc_solution, relax_full)

OUT = Path(__file__).parent / "output"
results = {"EAM": {}, "MACE": {}}

fac = eam_alloy_factory("CuAg.eam.alloy", ("Cu", "Ag"))

REPS = {32: (2, 2, 2), 108: (3, 3, 3), 256: (4, 4, 4), 500: (5, 5, 5)}
SEEDS_FOR = {32: (0, 1, 2), 108: (0, 1, 2), 256: (0, 1), 500: (0,)}


def timed_singlepoint(atoms, factory, n_rep=3):
    work = atoms.copy()
    work.calc = factory()
    work.get_potential_energy()          # build neighbor lists etc.
    t0 = time.perf_counter()
    for _ in range(n_rep):
        work.rattle(1e-6)                # force recompute
        work.get_potential_energy()
        work.get_forces()
    return (time.perf_counter() - t0) / n_rep


# ═══ 1. EAM: Ω(x=½) and timings vs size ══════════════════════════════════════
print("── EAM (Williams Ag-Cu): Ω(x=½) and cost vs supercell size")
E_ref = {}
for el, other in [("Ag", "Cu"), ("Cu", "Ag")]:
    rel, _ = relax_full(fcc_solution(el, other, 0.0), fac)
    E_ref[el] = float(rel.get_potential_energy()) / len(rel)

for n, reps in REPS.items():
    t_sp = timed_singlepoint(fcc_solution("Ag", "Cu", 0.5, reps=reps), fac)
    omegas, t_rel = [], None
    for s in SEEDS_FOR[n]:
        t0 = time.perf_counter()
        rel, _ = relax_full(fcc_solution("Ag", "Cu", 0.5, reps=reps, seed=s),
                            fac)
        t_rel = time.perf_counter() - t0
        dH = (float(rel.get_potential_energy()) / len(rel)
              - 0.5 * E_ref["Ag"] - 0.5 * E_ref["Cu"])
        omegas.append(4.0 * dH)
    om, om_sd = np.mean(omegas), (np.std(omegas, ddof=1)
                                  if len(omegas) > 1 else 0.0)
    print(f"   N={n:4d}: Ω = {om*1e3:+.1f} ± {om_sd*1e3:.1f} meV "
          f"({len(omegas)} seeds) | single point {t_sp*1e3:.1f} ms, "
          f"relax_full {t_rel:.1f} s")
    results["EAM"][n] = dict(omega_meV=om * 1e3, omega_sd_meV=om_sd * 1e3,
                             seeds=len(omegas), t_singlepoint_s=t_sp,
                             t_relax_s=t_rel)

# how big can a single EAM call go?
print("\n── EAM single-point ceiling")
for n, reps in [(2048, (8, 8, 8)), (6912, (12, 12, 12)),
                (16384, (16, 16, 16))]:
    t_sp = timed_singlepoint(fcc_solution("Ag", "Cu", 0.5, reps=reps), fac,
                             n_rep=1)
    print(f"   N={n:5d}: energy+forces in {t_sp*1e3:.0f} ms")
    results["EAM"][f"singlepoint_{n}"] = t_sp

# ═══ 2. MACE-MP-0: timings + 32→108 Ω check ══════════════════════════════════
print("\n── MACE-MP-0 (small, CPU, float64)")
from mace.calculators import mace_mp  # noqa: E402

_calc = mace_mp(model="small", device="cpu", default_dtype="float64")
mace_fac = lambda: _calc  # noqa: E731 — single shared calculator

for n, reps in [(32, (2, 2, 2)), (108, (3, 3, 3)), (256, (4, 4, 4))]:
    t_sp = timed_singlepoint(fcc_solution("Ag", "Cu", 0.5, reps=reps),
                             mace_fac, n_rep=1)
    print(f"   N={n:4d}: energy+forces in {t_sp:.2f} s")
    results["MACE"][f"singlepoint_{n}"] = t_sp

print("   Ω(x=½) 32 → 108 (1 seed, relax_full):")
E_ref_m = {}
for el, other in [("Ag", "Cu"), ("Cu", "Ag")]:
    rel, _ = relax_full(fcc_solution(el, other, 0.0), mace_fac)
    E_ref_m[el] = float(rel.get_potential_energy()) / len(rel)
for n, reps in [(32, (2, 2, 2)), (108, (3, 3, 3))]:
    t0 = time.perf_counter()
    rel, _ = relax_full(fcc_solution("Ag", "Cu", 0.5, reps=reps, seed=0),
                        mace_fac)
    dH = (float(rel.get_potential_energy()) / len(rel)
          - 0.5 * E_ref_m["Ag"] - 0.5 * E_ref_m["Cu"])
    dt = time.perf_counter() - t0
    print(f"   N={n:4d}: Ω = {4*dH*1e3:+.1f} meV (relax {dt:.0f} s)")
    results["MACE"][n] = dict(omega_meV=4 * dH * 1e3, t_relax_s=dt)

with open(OUT / "size_scaling.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'size_scaling.json'}")
