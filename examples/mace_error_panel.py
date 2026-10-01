"""
A comprehensive error panel for pretrained MACE-MP-0: where does the off-the-shelf
foundation model deviate from experiment/DFT, across THERMODYNAMIC and
KINETIC/MECHANICAL properties?  This quantifies "everything MACE gets wrong that
phase-diagram fine-tuning could fix."

Two groups:

  THERMODYNAMIC  (reuses examples/mace_benchmark.py logic/values)
    * lattice constants      Cu/Ni/Al/Ag fcc, Si diamond
    * polymorph stabilities  Zn fcc-hcp, Si fcc-diamond, Cu bcc-fcc
    * equimolar mixing Omega Ag-Cu, Cu-Ni
    * formation energies     Ni3Al L12, NiAl B2
    * phase-diagram features Tc(Cu-Ni), Te(Ag-Cu)

  KINETIC / MECHANICAL  (NEW here)
    * vacancy formation E_f  Cu, Ni, Al   via  E_f = E(vac) - (N-1)/N E(perfect)
    * bulk modulus B         Cu, Ni, Al   via  eos_fit on a 9-point E(V) scan
    * migration barrier E_m  Cu           (from the cached NEB tune result)

The thermodynamic panel is loaded from the cached pretrained benchmark JSON when
present (MACE on CPU is slow); pass --recompute-thermo to regenerate it. The NEW
kinetic/mechanical quantities are always computed fresh.

Run:  cd /path/to/coexist && PYTHONPATH=. python examples/mace_error_panel.py [--reps 2]
"""
import argparse, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter

from coexist.core.thermo_vib import eos_fit

OUT = Path(__file__).parent / "output"; OUT.mkdir(exist_ok=True)
EV_A3_TO_GPA = 160.21766208         # 1 eV/A^3 in GPa
K_B = 8.617333e-5

# fcc lattice constants used to seed the NEW calcs (A); match mace_benchmark refs
A_FCC = {"Cu": 3.615, "Ni": 3.524, "Al": 4.050}

# ---- references: property -> (ref, group, note). group: "thermo" | "kinmech".
# % error flag threshold is 10% (sign-sensitive polymorphs also flagged if wrong sign).
REF = {
    # --- THERMODYNAMIC (ref, group, lower-is-better-note) ---
    "a_Cu_fcc": (3.615, "thermo", "A"), "a_Ni_fcc": (3.524, "thermo", "A"),
    "a_Ag_fcc": (4.085, "thermo", "A"), "a_Al_fcc": (4.050, "thermo", "A"),
    "a_Si_dia": (5.431, "thermo", "A"),
    "dE_Zn_fcc_hcp": (30.0,  "thermo", "meV/atom"),   # hcp ground state -> +; MACE inverts sign
    "dE_Si_fcc_dia": (500.0, "thermo", "meV/atom"),
    "dE_Cu_bcc_fcc": (40.0,  "thermo", "meV/atom"),
    "Omega_AgCu": (340.0, "thermo", "meV/atom"),
    "Omega_CuNi": (108.0, "thermo", "meV/atom"),      # 2kB*625K
    "Hf_Ni3Al_L12": (-0.43, "thermo", "eV/atom"),
    "Hf_NiAl_B2":   (-0.67, "thermo", "eV/atom"),
    "Tc_CuNi_K":    (625.0, "thermo", "K"),
    "Te_AgCu_K":    (1052.0, "thermo", "K"),
    # --- KINETIC / MECHANICAL (NEW) ---
    "Ef_Cu": (1.28, "kinmech", "eV"), "Ef_Ni": (1.79, "kinmech", "eV"),
    "Ef_Al": (0.68, "kinmech", "eV"),
    "B_Cu": (140.0, "kinmech", "GPa"), "B_Ni": (180.0, "kinmech", "GPa"),
    "B_Al": (76.0,  "kinmech", "GPa"),
    "Em_Cu": (0.71, "kinmech", "eV"),
}

# human labels for the printed table
LABEL = {
    "a_Cu_fcc": "a Cu fcc", "a_Ni_fcc": "a Ni fcc", "a_Ag_fcc": "a Ag fcc",
    "a_Al_fcc": "a Al fcc", "a_Si_dia": "a Si diamond",
    "dE_Zn_fcc_hcp": "dE Zn fcc-hcp", "dE_Si_fcc_dia": "dE Si fcc-dia",
    "dE_Cu_bcc_fcc": "dE Cu bcc-fcc",
    "Omega_AgCu": "Omega Ag-Cu", "Omega_CuNi": "Omega Cu-Ni",
    "Hf_Ni3Al_L12": "Hf Ni3Al L12", "Hf_NiAl_B2": "Hf NiAl B2",
    "Tc_CuNi_K": "Tc Cu-Ni", "Te_AgCu_K": "Te Ag-Cu",
    "Ef_Cu": "E_f vacancy Cu", "Ef_Ni": "E_f vacancy Ni", "Ef_Al": "E_f vacancy Al",
    "B_Cu": "B bulk mod Cu", "B_Ni": "B bulk mod Ni", "B_Al": "B bulk mod Al",
    "Em_Cu": "E_m migration Cu",
}


def build(device="cpu"):
    from mace.calculators import mace_mp
    return mace_mp(model="small", device=device, default_dtype="float64")


def relax_pos(calc, atoms, fmax=0.03):
    """Relax atomic positions at fixed cell."""
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=200)
    return atoms


def relax_cell(calc, atoms, fmax=0.03):
    """Relax positions + cell."""
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=200)
    return atoms


def e_total(calc, atoms):
    a = atoms.copy(); a.calc = calc
    return a.get_potential_energy()


# ─────────────────────────── NEW: vacancy formation ───────────────────────────
def vacancy_formation(calc, el, a0, reps):
    """E_f = E(vacancy, N-1) - (N-1)/N * E(perfect, N).  Relax the perfect cell
    (cell+positions), then remove one atom and relax positions at fixed cell."""
    perfect = relax_cell(calc, bulk(el, "fcc", a=a0, cubic=True).repeat((reps,) * 3))
    N = len(perfect)
    e_perf = e_total(calc, perfect)
    vac = perfect.copy(); del vac[0]
    vac = relax_pos(calc, vac)
    e_vac = e_total(calc, vac)
    return float(e_vac - (N - 1) / N * e_perf)


# ─────────────────────────── NEW: bulk modulus ────────────────────────────────
def bulk_modulus(calc, el, a0, npts=9, span=0.04):
    """B from an E(V) scan of the 4-atom conventional fcc cell via eos_fit.
    Static energies (pure element, symmetric fcc -> nothing to relax per volume).
    Returns (B_GPa, V0_A3_per_cell, Bprime)."""
    scales = np.linspace(1 - span, 1 + span, npts)   # linear-dimension scale
    vols, enes = [], []
    for s in scales:
        at = bulk(el, "fcc", a=a0 * s, cubic=True)
        vols.append(at.get_volume())
        enes.append(e_total(calc, at))
    V0, E0, B0_evA3, Bprime = eos_fit(np.array(vols), np.array(enes))
    return float(B0_evA3) * EV_A3_TO_GPA, float(V0), float(Bprime)


# ─────────────────────────── thermo panel (reuse) ─────────────────────────────
def load_thermo(calc, recompute, reps):
    cached = OUT / "mace_benchmark_pretrained.json"
    if cached.exists() and not recompute:
        print(f"Loading cached thermodynamic panel: {cached.name}")
        d = json.load(open(cached))
    else:
        print("Recomputing thermodynamic panel via mace_benchmark.evaluate() ...")
        import mace_benchmark
        d = mace_benchmark.evaluate(calc, reps=reps)
    # remap benchmark Omega_CuNi ref alignment is handled by our REF; just pass through.
    return d


def migration_barrier():
    """Cu vacancy migration barrier from the cached NEB result."""
    f = OUT / "mace_finetune_neb_Cu_all_reg0.1.json"
    if f.exists():
        d = json.load(open(f))
        return float(d["Em_neb_pretrained"]), "cached NEB (mace_finetune_neb_Cu_all_reg0.1.json)"
    return 0.624, "literature-noted MACE value"


# ─────────────────────────── table + scoring ──────────────────────────────────
def pct_err(v, ref):
    return 100.0 * (v - ref) / ref if ref else float("nan")


def flagged(k, v, ref):
    """Flag properties MACE gets materially wrong."""
    p = pct_err(v, ref)
    # sign-sensitive polymorph stabilities: wrong sign is a hard fail
    if k in ("dE_Zn_fcc_hcp", "dE_Si_fcc_dia", "dE_Cu_bcc_fcc"):
        if np.sign(v) != np.sign(ref):
            return True
    return abs(p) > 10.0


def print_group(title, keys, vals):
    print(f"\n{title}")
    print(f"{'property':<20}{'MACE':>12}{'reference':>12}{'% error':>12}   flag")
    print("-" * 72)
    for k in keys:
        if k not in vals:
            continue
        v = vals[k]; ref, _, unit = REF[k]
        p = pct_err(v, ref)
        fl = "  <-- WRONG" if flagged(k, v, ref) else ""
        print(f"{LABEL[k]:<20}{v:>12.3f}{ref:>12.3f}{p:>+11.1f}%{fl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2, help="supercell reps (2 -> 32-atom fcc)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--recompute-thermo", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default="mace_error_panel.json")
    args = ap.parse_args()

    calc = build(args.device)
    vals = {}

    # --- thermodynamic (reuse) ---
    thermo = load_thermo(calc, args.recompute_thermo, args.reps)
    for k in REF:
        if REF[k][1] == "thermo" and k in thermo:
            vals[k] = thermo[k]

    # --- NEW kinetic/mechanical ---
    print(f"\nComputing vacancy formation energies ({args.reps}x{args.reps}x{args.reps} fcc)...")
    for el in ("Cu", "Ni", "Al"):
        vals[f"Ef_{el}"] = vacancy_formation(calc, el, A_FCC[el], args.reps)
        print(f"  E_f {el} = {vals[f'Ef_{el}']:.3f} eV")

    print("\nComputing bulk moduli via E(V) scan + eos_fit...")
    for el in ("Cu", "Ni", "Al"):
        B, V0, Bp = bulk_modulus(calc, el, A_FCC[el])
        vals[f"B_{el}"] = B
        print(f"  B {el} = {B:.1f} GPa  (V0={V0:.2f} A^3/cell, B'={Bp:.2f})")

    em, em_src = migration_barrier()
    vals["Em_Cu"] = em
    print(f"\nMigration barrier E_m Cu = {em:.3f} eV  [{em_src}]")

    # --- table ---
    thermo_keys = [k for k in REF if REF[k][1] == "thermo"]
    kin_keys = [k for k in REF if REF[k][1] == "kinmech"]
    print("\n" + "=" * 72)
    print("MACE-MP-0 (pretrained, small, float64)  ERROR PANEL")
    print("=" * 72)
    print_group("THERMODYNAMIC", thermo_keys, vals)
    print_group("KINETIC / MECHANICAL  (NEW)", kin_keys, vals)

    # summary of what's wrong
    wrong = [k for k in REF if k in vals and flagged(k, vals[k], REF[k][0])]
    print("\n" + "-" * 72)
    print(f"Flagged (materially wrong) properties: {len(wrong)} / {len(vals)}")
    for k in wrong:
        print(f"  {LABEL[k]:<20} MACE {vals[k]:+.3f}  ref {REF[k][0]:+.3f}  "
              f"({pct_err(vals[k], REF[k][0]):+.0f}%)  [{REF[k][1]}]")

    # --- save JSON ---
    payload = {
        "model": "MACE-MP-0 small float64 pretrained",
        "reps": args.reps,
        "values": vals,
        "references": {k: REF[k][0] for k in REF},
        "groups": {k: REF[k][1] for k in REF},
        "units": {k: REF[k][2] for k in REF},
        "pct_error": {k: pct_err(vals[k], REF[k][0]) for k in vals},
        "flagged_wrong": wrong,
        "migration_barrier_source": em_src,
    }
    json.dump(payload, open(OUT / args.out, "w"), indent=1)
    print(f"\nwrote {OUT / args.out}")

    # --- optional bar chart of signed % errors ---
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            keys = thermo_keys + kin_keys
            keys = [k for k in keys if k in vals]
            errs = [pct_err(vals[k], REF[k][0]) for k in keys]
            colors = ["#c44" if flagged(k, vals[k], REF[k][0]) else "#48a" for k in keys]
            fig, ax = plt.subplots(figsize=(9, 7))
            y = np.arange(len(keys))
            ax.barh(y, errs, color=colors)
            ax.set_yticks(y); ax.set_yticklabels([LABEL[k] for k in keys], fontsize=8)
            ax.axvline(0, color="k", lw=0.8)
            ax.axvline(10, color="grey", ls=":", lw=0.8); ax.axvline(-10, color="grey", ls=":", lw=0.8)
            n_thermo = len([k for k in thermo_keys if k in vals])
            ax.axhline(n_thermo - 0.5, color="k", ls="--", lw=0.8)
            ax.set_xlabel("signed % error  (MACE vs reference)")
            ax.set_title("MACE-MP-0 error panel: thermodynamic (top) vs kinetic/mechanical (bottom)\n"
                         "red = materially wrong (>10% or wrong sign)", fontsize=9)
            ax.invert_yaxis()
            fig.tight_layout()
            png = OUT / "mace_error_panel.png"
            fig.savefig(png, dpi=130)
            print(f"wrote {png}")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
