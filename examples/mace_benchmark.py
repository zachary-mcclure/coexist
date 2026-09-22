"""
A property benchmark for tracking whack-a-mole during phase-diagram fine-tuning.

Fine-tuning an MLIP against one phase diagram can silently break unrelated
properties. This evaluates a diverse, cheap (relaxation/statics-only) panel on
a given MACE model state and scores it against experiment/DFT references, so a
tune's collateral damage is visible. Run it on the pretrained model for a
baseline, then on a tuned checkpoint, and diff.

Panel (all fast — small cells, one relaxation each):
  * lattice constants of the elements we touch          (structural)
  * polymorph lattice stabilities (the orderings)       (energetics, sign-sensitive)
  * equimolar mixing interactions Omega                 (alloy energetics)
  * compound formation energies                         (intermetallics)
  * constructed phase-diagram features                  (the end targets)

Targets (should IMPROVE toward ref when tuned) and CONTROLS (must NOT degrade)
are labelled. A tune is only credible if its controls hold.

Run:      python examples/mace_benchmark.py [--checkpoint tuned.pt] [--out name.json]
Diff:     python examples/mace_benchmark.py --diff baseline.json tuned.json
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

OUT = Path(__file__).parent / "output"; OUT.mkdir(exist_ok=True)
K_B = 8.617333e-5

# ---- references (experiment unless noted DFT/CALPHAD); role: target|control|watch
REF = {
    # lattice constants, A
    "a_Cu_fcc":  (3.615, "control"), "a_Ni_fcc": (3.524, "control"),
    "a_Ag_fcc":  (4.085, "control"), "a_Al_fcc": (4.050, "control"),
    "a_Si_dia":  (5.431, "control"), "a_Zn_hcp": (2.665, "watch"),
    # polymorph lattice stabilities, meV/atom (DFT/CALPHAD sign is what matters)
    "dE_Zn_fcc_hcp": (30.0,  "target"),   # hcp is ground state -> POSITIVE (MACE: -9, inverted)
    "dE_Si_fcc_dia": (500.0, "watch"),    # DFT ~ +0.5 eV      (MACE: +424)
    "dE_Cu_bcc_fcc": (40.0,  "watch"),    # DFT ~ +40 meV      (MACE: +26)
    # equimolar mixing Omega, meV/atom
    "Omega_AgCu": (340.0, "control"),     # CALPHAD ~ +340     (MACE: +374, good)
    "Omega_CuNi": (105.0, "target"),      # 2kB*625K ~ +108    (MACE: +49, too low)
    # compound formation energies, eV/atom
    "Hf_Ni3Al_L12": (-0.43, "watch"),     # exp -0.42..-0.44
    "Hf_NiAl_B2":   (-0.67, "watch"),     # exp -0.66..-0.68
    # constructed phase-diagram features
    "Tc_CuNi_K":    (625.0, "target"),    # assessed 600-650   (MACE: 282)
    "Te_AgCu_K":    (1052.0, "control"),  # exp 1052           (MACE: 1054, good)
}


def build():
    from mace.calculators import mace_mp
    calc = mace_mp(model="small", device="cpu", default_dtype="float64")
    return calc, calc.models[0]


def relax(calc, atoms, cell_filter=True, fmax=0.03):
    atoms = atoms.copy(); atoms.calc = calc
    opt = FrechetCellFilter(atoms) if cell_filter else atoms
    FIRE(opt, logfile=None).run(fmax=fmax, steps=200)
    return atoms, atoms.get_potential_energy() / len(atoms)


def epa(calc, atoms):
    atoms = atoms.copy(); atoms.calc = calc
    return atoms.get_potential_energy() / len(atoms)


def evaluate(calc, ckpt=None):
    """Compute the full panel on the current model (optionally a tuned ckpt)."""
    if ckpt:                                   # overlay tuned weights
        model = calc.models[0]
        state = torch.load(ckpt, map_location="cpu")
        sd = dict(model.named_parameters())
        for name, w in state.items():
            if name in sd:
                sd[name].data.copy_(w)
    r = {}
    # --- lattice constants (relax cubic conventional cells) ---
    for el, lat, a0 in [("Cu","fcc",3.6),("Ni","fcc",3.5),("Ag","fcc",4.1),
                        ("Al","fcc",4.05),("Si","diamond",5.43)]:
        at, e = relax(calc, bulk(el, lat, a=a0, cubic=True))
        r[f"a_{el}_{'dia' if lat=='diamond' else lat}"] = float(at.cell.lengths()[0])
        r[f"_E_{el}_{lat}"] = e
    at, e = relax(calc, bulk("Zn","hcp", a=2.66, c=4.95))
    r["a_Zn_hcp"] = float(at.cell.lengths()[0]); r["_E_Zn_hcp"] = e
    # --- polymorph lattice stabilities (meV/atom) ---
    _, e_zn_fcc = relax(calc, bulk("Zn","fcc", a=3.9, cubic=True))
    r["dE_Zn_fcc_hcp"] = (e_zn_fcc - r["_E_Zn_hcp"]) * 1e3
    _, e_si_fcc = relax(calc, bulk("Si","fcc", a=3.9, cubic=True))
    r["dE_Si_fcc_dia"] = (e_si_fcc - r["_E_Si_diamond"]) * 1e3
    _, e_cu_bcc = relax(calc, bulk("Cu","bcc", a=2.9, cubic=True))
    r["dE_Cu_bcc_fcc"] = (e_cu_bcc - r["_E_Cu_fcc"]) * 1e3

    # --- equimolar mixing Omega (meV/atom), 32-atom random fcc ---
    def omega(elA, elB, aA, aB):
        rng = np.random.default_rng(0)
        at = bulk(elA, "fcc", a=(aA+aB)/2, cubic=True).repeat((2,2,2))
        nums = np.array([bulk(elA,"fcc").numbers[0]]*32)
        nums[rng.permutation(32)[:16]] = bulk(elB,"fcc").numbers[0]
        at.set_atomic_numbers(nums)
        _, e_mix = relax(calc, at, cell_filter=True)
        eA = r.get(f"_E_{elA}_fcc") or epa(calc, bulk(elA,"fcc",a=aA,cubic=True).repeat((2,2,2)))
        eB = r.get(f"_E_{elB}_fcc") or epa(calc, bulk(elB,"fcc",a=aB,cubic=True).repeat((2,2,2)))
        return 4.0 * (e_mix - 0.5*eA - 0.5*eB) * 1e3
    r["Omega_AgCu"] = omega("Ag","Cu",4.085,3.615)
    r["Omega_CuNi"] = omega("Cu","Ni",3.615,3.524)

    # --- compound formation energies (eV/atom) ---
    ni3al = bulk("Ni","fcc", a=3.57, cubic=True); ni3al.set_chemical_symbols(["Al","Ni","Ni","Ni"])
    _, e_ni3al = relax(calc, ni3al)
    e_al_fcc = epa(calc, bulk("Al","fcc", a=4.05, cubic=True))
    r["Hf_Ni3Al_L12"] = e_ni3al - 0.75*r["_E_Ni_fcc"] - 0.25*e_al_fcc
    nial = bulk("Ni","bcc", a=2.88, cubic=True); nial.set_chemical_symbols(["Ni","Al"])
    _, e_nial = relax(calc, nial)
    r["Hf_NiAl_B2"] = e_nial - 0.5*r["_E_Ni_fcc"] - 0.5*e_al_fcc

    # --- constructed phase-diagram features ---
    r["Tc_CuNi_K"] = r["Omega_CuNi"]/1e3 / (2*K_B)          # regular-solution consolute
    from coexist.core.phase_diagram import eutectic_point, regular_solution, liquid_solution
    G_L = liquid_solution(0.1555, dH_fus_A=0.11691, T_m_A=1234.93,
                          dH_fus_B=0.13742, T_m_B=1357.77)   # Ag-Cu lit. liquid
    _,_,_,Te = eutectic_point(regular_solution(r["Omega_AgCu"]/1e3), G_L)
    r["Te_AgCu_K"] = float(Te)

    return {k: v for k, v in r.items() if not k.startswith("_E_")}


def score(vals):
    print(f"\n{'property':<16}{'role':<9}{'value':>10}{'ref':>10}{'error':>10}")
    print("-"*55)
    for k, (ref, role) in REF.items():
        v = vals.get(k)
        if v is None: continue
        err = v - ref
        pct = 100*err/ref if ref else float("nan")
        print(f"{k:<16}{role:<9}{v:>10.3f}{ref:>10.3f}{err:>+9.2f} ({pct:+.0f}%)")


def diff(base, tuned):
    b, t = json.load(open(base)), json.load(open(tuned))
    print(f"\n{'property':<16}{'role':<9}{'baseline':>10}{'tuned':>9}{'ref':>9}   verdict")
    print("-"*66)
    for k, (ref, role) in REF.items():
        if k not in b or k not in t: continue
        eb, et = abs(b[k]-ref), abs(t[k]-ref)
        tag = "IMPROVED" if et < eb-1e-9 else ("REGRESSED" if et > eb+1e-9 else "flat")
        flag = "  <-- control regressed!" if (role=="control" and tag=="REGRESSED") else ""
        print(f"{k:<16}{role:<9}{b[k]:>10.3f}{t[k]:>9.3f}{ref:>9.3f}   {tag}{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--out", default="mace_benchmark_pretrained.json")
    ap.add_argument("--diff", nargs=2, metavar=("BASE","TUNED"))
    args = ap.parse_args()
    if args.diff:
        diff(*args.diff)
    else:
        calc, _ = build()
        vals = evaluate(calc, ckpt=args.checkpoint)
        score(vals)
        json.dump(vals, open(OUT/args.out, "w"), indent=1)
        print(f"\nwrote {OUT/args.out}")
