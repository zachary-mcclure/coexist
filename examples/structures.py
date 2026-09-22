"""
Figure: representative atomic structures behind the energetics.

Every number in this paper comes from one of three kinds of supercell —
a random substitutional solid solution, an ordered line compound, and a
melted liquid — and none of them had ever been drawn. This script renders
one real example of each, built with the exact same functions the paper's
own campaigns use (`fcc_solution` from lammps_factory.py; the same
L1_2 Ni3Al Atoms object as topology_ladder.py / gamma_prime_entropy.py),
plus a genuine short LAMMPS melt (not illustrative-only: the liquid panel
is an actual post-melt MD frame from the real CuAg.eam.alloy potential).

Output: docs/figures/structures.png
Runtime: ~15 s (one short Langevin melt, EAM).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase import Atoms, units  # noqa: E402
from ase.md.langevin import Langevin  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402
from ase.visualize.plot import plot_atoms  # noqa: E402

from coexist.adapters.lammps_factory import (  # noqa: E402
    eam_alloy_factory, fcc_solution)

FIG = Path(__file__).parent.parent / "docs" / "figures"
FIG.mkdir(exist_ok=True)

INK, MUT = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.titleweight": "bold", "text.color": INK,
    "savefig.dpi": 220, "figure.facecolor": "white",
})

# ── (a) random substitutional solid solution — same builder, same size ──────
# used for every Omega_s extraction in the paper (Sec. 3.2)
solid = fcc_solution("Cu", "Ag", 0.5, reps=(3, 3, 3), seed=0)  # 108 atoms

# ── (b) ordered line compound — identical Atoms object to topology_ladder.py /
# gamma_prime_entropy.py (Sec. 4.4/4.6): L1_2 Ni3Al, 108 atoms
a0 = 3.57
compound = Atoms(
    "AlNi3",
    positions=[[0, 0, 0], [0, a0 / 2, a0 / 2],
               [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]],
    cell=np.eye(3) * a0, pbc=True,
).repeat((3, 3, 3))

# ── (c) liquid — a genuine post-melt MD frame, real CuAg.eam.alloy potential
# (short Langevin melt; not the full sample_liquid_energy protocol, which
# also equilibrates/samples for statistics — this is a structure snapshot).
fac = eam_alloy_factory("CuAg.eam.alloy", ("Cu", "Ag"))
liquid = fcc_solution("Cu", "Ag", 0.5, reps=(3, 3, 3), seed=0)
liquid.set_cell(liquid.get_cell() * 1.06, scale_atoms=True)
liquid.calc = fac()
rng = np.random.default_rng(0)
MaxwellBoltzmannDistribution(liquid, temperature_K=2400.0,
                             rng=np.random.RandomState(0))
Langevin(liquid, timestep=2.0 * units.fs, temperature_K=2400.0,
         friction=0.02, rng=rng).run(500)  # 1 ps melt
liquid.wrap()

fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.1), layout="constrained")
panels = [
    (solid, "Random solid solution", "Cu-Ag, $x=0.5$, 108 atoms\n(every $\\Omega_s$ extraction)"),
    (compound, "Ordered line compound", "L1$_2$ Ni$_3$Al, 108 atoms\n($\\gamma'$, Sec. 4.4/4.6)"),
    (liquid, "Liquid (post-melt MD frame)", "Cu-Ag, 108 atoms, 2400 K\n(real CuAg.eam.alloy potential)"),
]
for ax, (atoms, title, sub) in zip(axes, panels):
    plot_atoms(atoms, ax, radii=0.55, rotation="15x,-15y,5z", show_unit_cell=1)
    ax.set_title(title, fontsize=9)
    # plot_atoms() calls ax.set_axis_off(), which also suppresses xlabel —
    # an explicit axes-fraction text survives that.
    ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top",
            fontsize=7, color=MUT)

fig.savefig(FIG / "structures.png")
plt.close(fig)
print("wrote", FIG / "structures.png")
