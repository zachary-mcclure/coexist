"""Tests for coexist.adapters.lammps_factory — builders + EAM plumbing.

Potential-file tests skip if the NIST downloads (potentials/README.md) or
the lammps wheel are absent.
"""
import numpy as np
import pytest

from coexist.adapters.lammps_factory import (
    FCC_LATTICE_CONSTANTS, eam_alloy_factory, fcc_solution, potentials_dir,
    relax_volume)


def test_fcc_solution_composition_and_count():
    at = fcc_solution("Ag", "Cu", 0.25, reps=(2, 2, 2), seed=3)
    syms = at.get_chemical_symbols()
    assert len(at) == 32
    assert syms.count("Cu") == 8
    assert at.get_pbc().all()


def test_fcc_solution_vegard_cell():
    at = fcc_solution("Ag", "Cu", 0.5, reps=(1, 1, 1))
    a_expected = 0.5 * (FCC_LATTICE_CONSTANTS["Ag"]
                        + FCC_LATTICE_CONSTANTS["Cu"])
    np.testing.assert_allclose(at.cell[0, 0], a_expected, rtol=1e-12)


def test_fcc_solution_seeds_differ():
    a = fcc_solution("Ag", "Cu", 0.5, seed=0).get_chemical_symbols()
    b = fcc_solution("Ag", "Cu", 0.5, seed=1).get_chemical_symbols()
    assert a != b


def test_missing_potential_raises():
    with pytest.raises(FileNotFoundError):
        eam_alloy_factory("NoSuchPotential.eam.alloy", ("Cu", "Ag"))


# ── engine-dependent (skipped without lammps wheel + downloaded files) ────────

lammps = pytest.importorskip("lammps")
needs_cuag = pytest.mark.skipif(
    not (potentials_dir() / "CuAg.eam.alloy").exists(),
    reason="CuAg.eam.alloy not downloaded (see potentials/README.md)")


@needs_cuag
def test_eam_agcu_mixing_enthalpy_positive():
    """Ag-Cu is a demixer: ΔH_mix(0.5) > 0 from the Williams 2006 EAM.
    Volume-relaxed only (fast); the full campaign adds internal relaxation."""
    fac = eam_alloy_factory("CuAg.eam.alloy", ("Cu", "Ag"))
    E = {}
    for tag, elA, elB, x in [("Ag", "Ag", "Cu", 0.0),
                             ("Cu", "Cu", "Ag", 0.0),
                             ("mix", "Ag", "Cu", 0.5)]:
        rel, _, _ = relax_volume(fcc_solution(elA, elB, x, seed=0), fac)
        rel.calc = fac()
        E[tag] = float(rel.get_potential_energy()) / len(rel)
    dH = E["mix"] - 0.5 * E["Ag"] - 0.5 * E["Cu"]
    assert 0.05 < dH < 0.20        # meV-scale window around the known +110


@needs_cuag
def test_eam_adapter_grad_matches_forces():
    """jax.grad through the eam/alloy adapter equals LAMMPS forces."""
    import jax
    import jax.numpy as jnp

    from coexist.adapters.lammps_factory import adapter_for_atoms

    fac = eam_alloy_factory("CuAg.eam.alloy", ("Cu", "Ag"))
    at = fcc_solution("Ag", "Cu", 0.5, seed=0)
    rng = np.random.default_rng(0)
    at.set_positions(at.get_positions() + rng.normal(0, 0.03, (len(at), 3)))

    ad = adapter_for_atoms(at, fac)
    g = np.asarray(jax.grad(ad)(jnp.array(at.get_positions())))

    ref = at.copy()
    ref.calc = fac()
    np.testing.assert_allclose(g, -ref.get_forces(), atol=1e-4)


def test_scaled_cross_potential_scales_only_middle_block(tmp_path):
    """Round-trip parse: the AB pair block scales, everything else is
    bit-identical (as floats)."""
    src = potentials_dir() / "CuAg.eam.alloy"
    if not src.exists():
        pytest.skip("CuAg.eam.alloy not downloaded")
    from coexist.adapters.lammps_factory import scaled_cross_potential

    dst = scaled_cross_potential(src, 0.5, out_dir=tmp_path)

    def tokens(path):
        lines = path.read_text().splitlines()
        nrho, _, nr, _, _ = lines[4].split()
        nrho, nr = int(nrho), int(nr)
        vals, headers_seen = [], 0
        for ln in lines[5:]:
            parts = ln.split()
            if (len(parts) == 4 and parts[0].isdigit() and headers_seen < 2
                    and len(vals) in (0, nrho + nr)):
                headers_seen += 1
            else:
                vals.extend(float(t) for t in parts)
        return np.asarray(vals), nrho, nr

    v_src, nrho, nr = tokens(src)
    v_dst, _, _ = tokens(dst)
    ab = 2 * (nrho + nr) + nr
    np.testing.assert_allclose(v_dst[:ab], v_src[:ab], rtol=1e-12)
    np.testing.assert_allclose(v_dst[ab:ab + nr], 0.5 * v_src[ab:ab + nr],
                               rtol=1e-12)
    np.testing.assert_allclose(v_dst[ab + nr:], v_src[ab + nr:], rtol=1e-12)


@needs_cuag
def test_scaled_cross_energy_linear_in_lambda(tmp_path):
    """E(λ) is exactly linear at fixed geometry, and pure elements are
    λ-invariant — the foundation of the engine-free inverse loop."""
    from coexist.adapters.lammps_factory import scaled_cross_potential

    p0 = scaled_cross_potential("CuAg.eam.alloy", 0.0, out_dir=tmp_path)
    p5 = scaled_cross_potential("CuAg.eam.alloy", 0.5, out_dir=tmp_path)

    def energy(atoms, pot):
        w = atoms.copy()
        w.calc = eam_alloy_factory(pot, ("Cu", "Ag"))()
        return float(w.get_potential_energy())

    mix = fcc_solution("Ag", "Cu", 0.5, seed=0)
    e1 = energy(mix, "CuAg.eam.alloy")
    e0 = energy(mix, p0)
    e5 = energy(mix, p5)
    assert abs(e5 - 0.5 * (e0 + e1)) < 1e-9

    pure = fcc_solution("Cu", "Ag", 0.0, seed=0)
    assert abs(energy(pure, "CuAg.eam.alloy") - energy(pure, p0)) < 1e-10


@needs_cuag
def test_sample_liquid_energy_smoke():
    """Tiny trajectory: sane thermostat, finite window-mean stats."""
    from coexist.adapters.lammps_factory import sample_liquid_energy

    at = fcc_solution("Ag", "Cu", 0.5, reps=(2, 2, 2), seed=0)
    fac = eam_alloy_factory("CuAg.eam.alloy", ("Cu", "Ag"))
    out = sample_liquid_energy(at, fac, T_K=1400.0, melt_T=2200.0,
                               melt_ps=0.2, equil_ps=0.2, prod_ps=0.5,
                               seed=0, sample_every=10)
    assert np.isfinite(out["E_mean"]) and out["E_sem"] >= 0.0
    assert 900.0 < out["T_mean"] < 2200.0     # short run: loose thermostat band
    assert out["V_mean"] > 10.0               # expanded past solid density
    assert len(out["samples"]) == 25
