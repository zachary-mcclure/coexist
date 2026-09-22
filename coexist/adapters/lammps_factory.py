"""
LAMMPS calculator factories + alloy supercell builders for the phase-diagram
ladder.

Everything here is HOST-SIDE construction: builders return ASE calculators /
Atoms objects that get wrapped by ASEAdapter, which is where differentiability
lives. The division of labor:

    lammps_factory   — which engine, which potential file, which supercell
    ASEAdapter       — one-engine-call (E, F) with forces as the VJP residual
    core/mixing      — JAX-traced ΔH_mix / Ω / Redlich-Kister on top

Potential files live in `<repo>/potentials/` with provenance in its README.
All EAM files are NIST Interatomic Potentials Repository setfl (eam/alloy).

Honest scope notes:
  * Volume relaxation (`relax_volume`) is a forward-only host loop — the
    traced gradient is exact at the relaxed geometry (snapshot bridge), it
    does not include ∂(relaxed volume)/∂(inputs) response terms.
  * `fcc_solution`/`bcc_solution`/`hcp_solution` build random substitutional
    solutions on an ideal cubic (fcc/bcc) or hexagonal (hcp) lattice — the
    right construction for mixing energetics of disordered solutions
    (13-atom clusters give wrong-sign Ω). β-Sn's body-centered
    tetragonal cell (Cu-Sn's pure-Sn reference, anisotropic, needs full
    cell-shape relaxation) and diamond-cubic (Al-Si) are each built ad hoc
    at their point of use rather than through a shared factory here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "potentials_dir", "eam_alloy_factory", "meam_factory",
    "fcc_solution", "bcc_solution", "hcp_solution",
    "relax_volume", "adapter_for_atoms",
    "FCC_LATTICE_CONSTANTS",
]

# 0 K lattice constants (Å) — initial guesses only; relax_volume refines.
FCC_LATTICE_CONSTANTS = {
    "Cu": 3.615, "Ni": 3.524, "Ag": 4.085, "Au": 4.078,
    "Al": 4.050, "Pb": 4.950,
}


def potentials_dir() -> Path:
    """<repo root>/potentials — resolved relative to this file."""
    return Path(__file__).resolve().parents[2] / "potentials"


# ── Calculator factories (each returns a zero-arg factory for ASEAdapter) ────

def eam_alloy_factory(
    pot_file: str | Path,
    elements: Sequence[str],
) -> Callable:
    """Factory for a LAMMPSlib calculator with a setfl eam/alloy potential.

    Args:
        pot_file:  filename in potentials/ (or absolute path).
        elements:  element symbols IN THE FILE'S ORDER (see setfl header
                   line 4, e.g. ``2 Cu Ag`` → ``("Cu", "Ag")``). LAMMPS type
                   i is mapped to elements[i-1].
    """
    path = Path(pot_file)
    if not path.is_absolute():
        path = potentials_dir() / path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} — download it per potentials/README.md")
    elements = list(elements)
    cmds = [
        "pair_style eam/alloy",
        f"pair_coeff * * {path} {' '.join(elements)}",
    ]
    atom_types = {el: i + 1 for i, el in enumerate(elements)}

    def factory():
        from ase.calculators.lammpslib import LAMMPSlib
        return LAMMPSlib(lmpcmds=cmds, atom_types=atom_types,
                         keep_alive=True, log_file=None)

    return factory


def meam_factory(
    library_file: str | Path,
    param_file: str | Path,
    elements: Sequence[str],
) -> Callable:
    """Factory for a LAMMPSlib MEAM calculator (e.g. Pb-Sn Etesami 2018).

    Requires the LAMMPS build to include the MEAM package (the 2025 pip
    wheel does; `pair_style meam` is the merged meam/c implementation).
    """
    lib = Path(library_file)
    par = Path(param_file)
    if not lib.is_absolute():
        lib = potentials_dir() / lib
    if not par.is_absolute():
        par = potentials_dir() / par
    elements = list(elements)
    els = " ".join(elements)
    cmds = [
        "pair_style meam",
        f"pair_coeff * * {lib} {els} {par} {els}",
    ]
    atom_types = {el: i + 1 for i, el in enumerate(elements)}

    def factory():
        from ase.calculators.lammpslib import LAMMPSlib
        return LAMMPSlib(lmpcmds=cmds, atom_types=atom_types,
                         keep_alive=True, log_file=None)

    return factory


# ── Supercell builders ────────────────────────────────────────────────────────

def fcc_solution(
    el_A: str,
    el_B: str,
    x_B: float,
    reps: tuple[int, int, int] = (2, 2, 2),
    a: float | None = None,
    seed: int = 0,
):
    """Random substitutional fcc solid solution A_{1-x}B_x (periodic, cubic).

    Vegard's-rule lattice constant unless `a` is given; 2×2×2 cubic reps
    → 32 atoms, so x is representable in units of 1/32. Returns ASE Atoms
    (no calculator attached).
    """
    from ase.build import bulk

    if a is None:
        a_A = FCC_LATTICE_CONSTANTS[el_A]
        a_B = FCC_LATTICE_CONSTANTS[el_B]
        a = (1.0 - x_B) * a_A + x_B * a_B
    atoms = bulk(el_A, "fcc", a=a, cubic=True).repeat(reps)
    n = len(atoms)
    n_B = int(round(x_B * n))
    if abs(n_B - x_B * n) > 1e-6:
        # keep silent rounding out of the physics: caller sees actual x
        pass
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_B, replace=False)
    symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
    symbols[idx] = el_B
    atoms.set_chemical_symbols(list(symbols))
    return atoms


def bcc_solution(
    el_A: str,
    el_B: str,
    x_B: float,
    reps: tuple[int, int, int] = (3, 3, 3),
    a: float | None = None,
    seed: int = 0,
):
    """Random substitutional bcc solid solution A_{1-x}B_x (periodic, cubic).

    Same construction as `fcc_solution`, on a bcc host instead -- caller
    supplies `a` (no built-in bcc lattice-constant table, since bcc systems
    in this package so far each set it explicitly per rung). 3x3x3 cubic
    reps -> 54 atoms.
    """
    from ase.build import bulk

    if a is None:
        raise ValueError("bcc_solution requires an explicit lattice "
                         "constant `a` (no built-in bcc table)")
    atoms = bulk(el_A, "bcc", a=a, cubic=True).repeat(reps)
    n = len(atoms)
    n_B = int(round(x_B * n))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_B, replace=False)
    symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
    symbols[idx] = el_B
    atoms.set_chemical_symbols(list(symbols))
    return atoms


def hcp_solution(
    el_A: str,
    el_B: str,
    x_B: float,
    reps: tuple[int, int, int] = (3, 3, 2),
    a: float | None = None,
    c: float | None = None,
    seed: int = 0,
):
    """Random substitutional hcp solid solution A_{1-x}B_x (periodic).

    Same construction as `fcc_solution`/`bcc_solution`, on an hcp host --
    caller supplies `a` and (optionally) `c` (defaults to the ideal
    c/a = sqrt(8/3) if not given). hcp is NOT cubic, so `ase.build.bulk`
    is called without `cubic=True`; 3x3x2 reps -> 36 atoms (hcp's
    conventional cell already has 2 atoms/cell, unlike fcc/bcc's 1).
    """
    from ase.build import bulk

    if a is None:
        raise ValueError("hcp_solution requires an explicit lattice "
                         "constant `a` (no built-in hcp table)")
    kwargs = dict(a=a)
    if c is not None:
        kwargs["c"] = c
    atoms = bulk(el_A, "hcp", **kwargs).repeat(reps)
    n = len(atoms)
    n_B = int(round(x_B * n))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_B, replace=False)
    symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
    symbols[idx] = el_B
    atoms.set_chemical_symbols(list(symbols))
    return atoms


def relax_volume(
    atoms,
    calculator_factory: Callable,
    scale_range: float = 0.04,
    n_points: int = 9,
):
    """Isotropic volume relaxation by quadratic fit on E(scale).

    Scans `n_points` isotropic strains in [1−scale_range, 1+scale_range],
    fits a parabola around the minimum, rebuilds the cell at the optimum.
    Returns (relaxed_atoms, scale_opt, energies) — forward-only host loop
    (~n_points engine calls); the differentiable single point comes after,
    through ASEAdapter at the relaxed geometry.
    """
    calc = calculator_factory()
    base_cell = atoms.get_cell()[:]
    base_pos = atoms.get_positions().copy()
    scales = np.linspace(1.0 - scale_range, 1.0 + scale_range, n_points)
    energies = []
    work = atoms.copy()
    work.calc = calc
    for s in scales:
        work.set_cell(base_cell * s, scale_atoms=False)
        work.set_positions(base_pos * s)
        energies.append(float(work.get_potential_energy()))
    energies = np.asarray(energies)

    # parabola through the 3 points bracketing the minimum
    i = int(np.argmin(energies))
    i = min(max(i, 1), n_points - 2)
    p = np.polyfit(scales[i - 1:i + 2], energies[i - 1:i + 2], 2)
    s_opt = float(np.clip(-p[1] / (2 * p[0]), scales[0], scales[-1]))

    relaxed = atoms.copy()
    relaxed.set_cell(base_cell * s_opt, scale_atoms=False)
    relaxed.set_positions(base_pos * s_opt)
    return relaxed, s_opt, energies


def relax_positions(
    atoms,
    calculator_factory: Callable,
    fmax: float = 0.02,
    steps: int = 200,
):
    """Relax internal coordinates at fixed cell (host-side BFGS, quiet).

    Essential for size-mismatched alloys (Ag-Cu: 13% radius mismatch) —
    unrelaxed random supercells overestimate ΔH_mix by tens of meV/atom.
    Returns the relaxed Atoms (calculator attached).
    """
    from ase.optimize import BFGS

    work = atoms.copy()
    work.calc = calculator_factory()
    opt = BFGS(work, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return work


def relax_full(
    atoms,
    calculator_factory: Callable,
    fmax: float = 0.02,
    n_cycles: int = 2,
):
    """Alternating volume + internal-coordinate relaxation.

    volume → positions, repeated `n_cycles` times — adequate for cubic
    random solutions (full cell-shape relaxation is not needed on a cubic
    lattice). Returns (relaxed_atoms, final_scale).
    """
    work = atoms.copy()
    s_total = 1.0
    for _ in range(n_cycles):
        work, s, _ = relax_volume(work, calculator_factory)
        s_total *= s
        work = relax_positions(work, calculator_factory, fmax=fmax)
    return work, s_total


def scaled_cross_potential(
    pot_file: str | Path,
    scale: float,
    out_dir: str | Path | None = None,
) -> Path:
    """Write a copy of a 2-element setfl (eam/alloy) file with the CROSS
    pair function φ_AB(r) multiplied by `scale`.

    This is the engine-parameter knob for the inverse capstone: in eam/alloy, E = Σ F_i(ρ_i) + ½Σ φ_ij(r), and densities ρ are
    per-element — so scaling only the A-B pair block leaves embedding and
    pure-element energetics EXACTLY unchanged, and the total energy of any
    fixed configuration is LINEAR in `scale`:

        E(λ) = E(0) + λ·[E(1) − E(0)]

    Two engine evaluations therefore determine E(λ) for all λ — the entire
    inverse-fitting loop downstream runs engine-free.

    Pair blocks follow setfl order (i=1..N, j=1..i): AA, AB, BB — the
    middle block is scaled. Returns the path of the generated file.
    """
    src = Path(pot_file)
    if not src.is_absolute():
        src = potentials_dir() / src
    lines = src.read_text().splitlines()
    ntypes = int(lines[3].split()[0])
    if ntypes != 2:
        raise ValueError(f"cross-pair scaling needs 2 elements, got {ntypes}")
    nrho, _, nr, _, _ = lines[4].split()
    nrho, nr = int(nrho), int(nr)

    # token stream of the numeric body, tracking element header lines
    head = lines[:5]
    body = lines[5:]
    tokens: list[str] = []
    elem_headers: list[tuple[int, str]] = []   # (token index, header line)
    for ln in body:
        parts = ln.split()
        # element header lines start with an integer Z and have 4 fields
        if (len(parts) == 4 and parts[0].isdigit()
                and len(elem_headers) < ntypes
                and len(tokens) in (0, nrho + nr)):
            elem_headers.append((len(tokens), ln))
        else:
            tokens.extend(parts)

    n_expected = ntypes * (nrho + nr) + 3 * nr
    if len(tokens) != n_expected:
        raise ValueError(
            f"setfl parse mismatch: {len(tokens)} values, expected {n_expected}")

    vals = np.array([float(t) for t in tokens])
    ab_start = ntypes * (nrho + nr) + nr      # after AA block
    vals[ab_start:ab_start + nr] *= scale

    out_dir = Path(out_dir) if out_dir else src.parent
    dst = out_dir / f"{src.stem}_lam{scale:.6f}{src.suffix}"
    with open(dst, "w") as f:
        f.write("\n".join(head) + "\n")
        i = 0
        for k in range(ntypes):
            f.write(elem_headers[k][1] + "\n")
            for v in vals[i:i + nrho + nr]:
                f.write(f" {v:24.16E}\n")
            i += nrho + nr
        for v in vals[i:]:
            f.write(f" {v:24.16E}\n")
    return dst


def eos_scan(
    atoms,
    calculator_factory: Callable,
    scale_range: float = 0.04,
    n_points: int = 9,
    relax_internal: bool = True,
    fmax: float = 0.01,
):
    """Per-atom E(V) scan for EOS fitting (feeds `core.thermo_vib.eos_fit`).

    Relaxes the input structure fully, then scans isotropic volume strains,
    re-relaxing internal coordinates at each volume when `relax_internal`
    (essential for random alloys — frozen coordinates inflate the apparent
    bulk modulus; pure elements are insensitive). Returns
    (V_per_atom, E_per_atom) numpy arrays.
    """
    rel, _ = relax_full(atoms, calculator_factory)
    base_cell = rel.get_cell()[:]
    base_pos = rel.get_positions().copy()
    n = len(rel)
    V_cell = rel.get_volume()
    scales = np.linspace(1.0 - scale_range, 1.0 + scale_range, n_points)
    energies = []
    for s in scales:
        work = rel.copy()
        work.set_cell(base_cell * s, scale_atoms=False)
        work.set_positions(base_pos * s)
        if relax_internal:
            work = relax_positions(work, calculator_factory, fmax=fmax)
        else:
            work.calc = calculator_factory()
        energies.append(float(work.get_potential_energy()))
    return V_cell * scales**3 / n, np.asarray(energies) / n


def sample_liquid_energy(
    atoms,
    calculator_factory: Callable,
    T_K: float,
    melt_T: float = 2400.0,
    melt_ps: float = 2.0,
    equil_ps: float = 4.0,
    prod_ps: float = 10.0,
    dt_fs: float = 2.0,
    expand: float = 1.06,
    seed: int = 0,
    sample_every: int = 20,
    compressibility_au: float = 1.2,
    production: str = "mtk",
    ttime_fs: float = 100.0,
    ptime_fs: float = 1000.0,
    prod_burnin_ps: float = 2.0,
):
    """Liquid potential energy per atom at (T_K, P≈0) — forward-only MD.

    Protocol: linear expansion by `expand` (≈ liquid density) → Langevin
    NVT melt at `melt_T` → Berendsen NPT at (T_K, P=0) equilibration
    (Berendsen is a relaxation scheme, fine for equilibration only) →
    production in the correct-ensemble isotropic MTK NPT (Nosé-Hoover
    chains on both thermostat and barostat), sampling E_pot every
    `sample_every` steps. `production="berendsen"` restores the older Berendsen
    protocol.

    Returns dict(E_mean, E_sem, V_mean, T_mean, samples). E_sem is the
    standard error from 5-block averaging of the production window (the
    window-mean estimator). This is the sampling bridge for LIQUID
    free-energy constructors — not differentiable through the trajectory
    (adapter scope: snapshot bridges only); error bars propagate to
    boundaries via the delta method downstream.
    """
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.nose_hoover_chain import IsotropicMTKNPT
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    rng = np.random.default_rng(seed)
    work = atoms.copy()
    work.set_cell(work.get_cell() * expand, scale_atoms=True)
    work.calc = calculator_factory()
    MaxwellBoltzmannDistribution(work, temperature_K=melt_T,
                                 rng=np.random.RandomState(seed))

    dt = dt_fs * units.fs
    n_melt = int(melt_ps * 1000 / dt_fs)
    n_equil = int(equil_ps * 1000 / dt_fs)
    n_prod = int(prod_ps * 1000 / dt_fs)

    Langevin(work, timestep=dt, temperature_K=melt_T, friction=0.02,
             rng=rng).run(n_melt)

    npt = NPTBerendsen(work, timestep=dt, temperature_K=T_K,
                       pressure_au=0.0, taut=100 * units.fs,
                       taup=500 * units.fs,
                       compressibility_au=compressibility_au)
    npt.run(n_equil)

    if production == "mtk":
        prod_dyn = IsotropicMTKNPT(work, timestep=dt, temperature_K=T_K,
                                   pressure_au=0.0,
                                   tdamp=ttime_fs * units.fs,
                                   pdamp=ptime_fs * units.fs)
    elif production == "berendsen":
        prod_dyn = npt
    else:
        raise ValueError(f"unknown production dynamics: {production!r}")

    n_burn = int(prod_burnin_ps * 1000 / dt_fs)
    if n_burn > 0:
        prod_dyn.run(n_burn)  # burn in the production thermostat/barostat

    n = len(work)
    E_s, V_s, T_s = [], [], []
    for _ in range(n_prod // sample_every):
        prod_dyn.run(sample_every)
        E_s.append(work.get_potential_energy() / n)
        V_s.append(work.get_volume() / n)
        T_s.append(work.get_temperature())
    E_s = np.asarray(E_s)

    blocks = np.array_split(E_s, 5)
    block_means = np.array([b.mean() for b in blocks])
    return {
        "E_mean": float(E_s.mean()),
        "E_sem": float(block_means.std(ddof=1) / np.sqrt(len(block_means))),
        "V_mean": float(np.mean(V_s)),
        "T_mean": float(np.mean(T_s)),
        "samples": E_s,
    }


def adapter_for_atoms(atoms, calculator_factory: Callable, dtype=None,
                      name: str | None = None):
    """ASEAdapter for an already-built periodic Atoms object."""
    import jax.numpy as jnp

    from .simulator_adapter import ASEAdapter

    return ASEAdapter(
        atoms.get_atomic_numbers().tolist(), calculator_factory,
        cell=atoms.get_cell()[:], pbc=True,
        dtype=dtype if dtype is not None else jnp.float32,
        name=name or "LAMMPS",
    )
