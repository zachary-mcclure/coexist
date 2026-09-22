"""Did the undercooled liquids stay liquid?

The Ag-Cu liquid campaign (liquid_omega.py) samples liquid Cu at 1150 K,
208 K below its EAM melting point, and liquid Ag 85 K below its own.
A crystallizing sample would corrupt the window-mean estimator silently:
the energy would drift down and the mixing enthalpy with it. This script
reruns the identical protocol (same factory, expand 1.06, Langevin melt
at 2400 K, Berendsen NPT equilibration, MTK NPT production with 2 ps
burn-in, same seeds) recording positions and energies through the 30 ps
production window, and reports per seed:

  - the mean-square displacement at the end of the window and the
    diffusion coefficient from a linear fit over 5-30 ps (a diffusive
    liquid sustains D ~ 1e-9 m^2/s; a crystallized sample plateaus at
    the sub-A^2 Lindemann scale);
  - the potential-energy drift across the window (crystallization
    releases tens of meV/atom).

Output: examples/output/liquid_msd_check.json
"""
import json
from pathlib import Path

import numpy as np

from coexist.adapters.lammps_factory import eam_alloy_factory, fcc_solution

OUT = Path(__file__).parent / "output"
T_K = 1150.0
SEEDS = (0, 1, 2)
DT_FS = 2.0
SAMPLE_EVERY = 20
MELT_T, MELT_PS, EQUIL_PS, BURN_PS, PROD_PS = 2400.0, 2.0, 4.0, 2.0, 30.0
FIT_WINDOW_PS = (5.0, 30.0)

fac = eam_alloy_factory("CuAg.eam.alloy", ("Cu", "Ag"))


def msd_run(atoms, seed):
    """sample_liquid_energy's protocol, recording positions and energy."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.nose_hoover_chain import IsotropicMTKNPT
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    rng = np.random.default_rng(seed)
    work = atoms.copy()
    work.set_cell(work.get_cell() * 1.06, scale_atoms=True)
    work.calc = fac()
    MaxwellBoltzmannDistribution(work, temperature_K=MELT_T,
                                 rng=np.random.RandomState(seed))
    dt = DT_FS * units.fs
    steps = lambda ps: int(ps * 1000 / DT_FS)                  # noqa: E731

    Langevin(work, timestep=dt, temperature_K=MELT_T, friction=0.02,
             rng=rng).run(steps(MELT_PS))
    NPTBerendsen(work, timestep=dt, temperature_K=T_K, pressure_au=0.0,
                 taut=100 * units.fs, taup=500 * units.fs,
                 compressibility_au=1.2).run(steps(EQUIL_PS))
    prod = IsotropicMTKNPT(work, timestep=dt, temperature_K=T_K,
                           pressure_au=0.0, tdamp=100 * units.fs,
                           pdamp=1000 * units.fs)
    prod.run(steps(BURN_PS))

    n = len(work)
    r0 = work.get_positions().copy()
    t_ps, msd, e_pot = [0.0], [0.0], [work.get_potential_energy() / n]
    for k in range(steps(PROD_PS) // SAMPLE_EVERY):
        prod.run(SAMPLE_EVERY)
        dr = work.get_positions() - r0
        t_ps.append((k + 1) * SAMPLE_EVERY * DT_FS / 1000.0)
        msd.append(float((dr ** 2).sum(axis=1).mean()))
        e_pot.append(work.get_potential_energy() / n)
    t_ps, msd, e_pot = map(np.asarray, (t_ps, msd, e_pot))

    m = (t_ps >= FIT_WINDOW_PS[0]) & (t_ps <= FIT_WINDOW_PS[1])
    slope = np.polyfit(t_ps[m], msd[m], 1)[0]          # A^2/ps
    half = len(e_pot) // 2
    return {
        "msd_final_A2": float(msd[-1]),
        "D_m2_per_s": slope / 6.0 * 1e-8,              # A^2/ps -> m^2/s
        "E_drift_meV": (float(e_pot[half:].mean())
                        - float(e_pot[:half].mean())) * 1e3,
    }


results = {}
print(f"── stayed-liquid check at {T_K:.0f} K "
      f"(protocol of liquid_omega.py, {PROD_PS:.0f} ps production)")
for tag, elA, elB, x in [("Cu", "Cu", "Ag", 0.0), ("Ag", "Ag", "Cu", 0.0),
                         ("mix", "Ag", "Cu", 0.5)]:
    per_seed = [msd_run(fcc_solution(elA, elB, x, reps=(3, 3, 3), seed=s), s)
                for s in SEEDS]
    results[tag] = per_seed
    D = [p["D_m2_per_s"] for p in per_seed]
    msd_f = [p["msd_final_A2"] for p in per_seed]
    drift = [p["E_drift_meV"] for p in per_seed]
    print(f"   {tag:3s}: D = {min(D)*1e9:.1f}-{max(D)*1e9:.1f}e-9 m^2/s, "
          f"MSD(30 ps) = {min(msd_f):.0f}-{max(msd_f):.0f} A^2, "
          f"E drift = {min(drift):+.1f} to {max(drift):+.1f} meV/atom")

allv = [p for v in results.values() for p in v]
results["summary"] = {
    "D_min_max_1e9_m2_s": [min(p["D_m2_per_s"] for p in allv) * 1e9,
                           max(p["D_m2_per_s"] for p in allv) * 1e9],
    "msd_final_min_A2": min(p["msd_final_A2"] for p in allv),
    "E_drift_max_abs_meV": max(abs(p["E_drift_meV"]) for p in allv),
}
with open(OUT / "liquid_msd_check.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nWrote {OUT/'liquid_msd_check.json'}")
