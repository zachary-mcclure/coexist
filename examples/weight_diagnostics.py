"""
How much did the potential have to move? Weight-change diagnostics per tune.

For every tuned checkpoint, compare its weights to pretrained MACE-MP-0 and
report how many parameters were trained and by how much: the L2 change (absolute
and relative), the largest single-weight move, the per-module distribution, and
a histogram of the weight deltas. Lining these up against each tune's benchmark
verdict (controls held / regressed, read from the sibling JSON) shows the
potential's "budget" for each property -- how big a weight perturbation a target
demands, and when that perturbation starts breaking controls.

Run:  PYTHONPATH=. python examples/weight_diagnostics.py
Out:  examples/output/weight_diagnostics.png  (+ .json)
"""
import glob, json
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mace.calculators import mace_mp

OUT = Path(__file__).parent / "output"
CKPTS = sorted(glob.glob(str(OUT / "mace_finetune_*.pt")))

print("Loading pretrained MACE-MP-0 reference weights…")
calc = mace_mp(model="small", device="cpu", default_dtype="float64")
theta0 = {n: p.detach().cpu().double().clone()
          for n, p in calc.models[0].named_parameters()}


def verdict_for(ckpt_path):
    j = Path(str(ckpt_path).replace(".pt", ".json"))
    if j.exists():
        d = json.load(open(j))
        ok = d.get("controls_ok")
        return {True: "held", False: "REGRESSED"}.get(ok, "n/a")
    return "n/a"


rows, hist_data = [], {}
for ck in CKPTS:
    state = torch.load(ck, map_location="cpu")
    deltas, per_mod = [], {}
    ref_sq = 0.0
    for name, w in state.items():
        if name not in theta0:
            continue
        d = (w.double() - theta0[name]).flatten().numpy()
        deltas.append(d)
        ref_sq += float((theta0[name] ** 2).sum())
        mod = name.split(".")[0]
        per_mod[mod] = per_mod.get(mod, 0.0) + float((d ** 2).sum())
    if not deltas:
        continue
    d = np.concatenate(deltas)
    l2 = float(np.sqrt((d ** 2).sum()))
    rel = l2 / np.sqrt(ref_sq) if ref_sq > 0 else float("nan")
    name = Path(ck).name.replace("mace_finetune_", "").replace(".pt", "")
    subset = "all" if d.size > 1e5 else "readout"
    rows.append(dict(name=name, n=d.size, subset=subset, l2=l2, rel=rel,
                     maxabs=float(np.abs(d).max()), rms=float(np.sqrt((d ** 2).mean())),
                     frac_moved=float((np.abs(d) > 1e-4).mean()),
                     verdict=verdict_for(ck),
                     top_mod=max(per_mod, key=per_mod.get) if per_mod else "-"))
    hist_data[name] = d

rows.sort(key=lambda r: r["rel"])
print(f"\n{'tune':<34}{'subset':<9}{'n_train':>9}{'rel.L2':>9}{'max|dw|':>10}"
      f"{'%moved':>8}{'top module':>14}{'controls':>11}")
print("-" * 104)
for r in rows:
    print(f"{r['name']:<34}{r['subset']:<9}{r['n']:>9}{r['rel']*100:>8.3f}%"
          f"{r['maxabs']:>10.2e}{r['frac_moved']*100:>7.1f}%{r['top_mod']:>14}"
          f"{r['verdict']:>11}")

# ── figure: weight-delta histograms (all-weights tunes) + relative-L2 bars ────
fig, (axh, axb) = plt.subplots(1, 2, figsize=(11, 4.4))
big = [r for r in rows if r["subset"] == "all"]
for r in big:
    d = hist_data[r["name"]]
    axh.hist(d, bins=200, range=(-0.02, 0.02), histtype="step", lw=1.6,
             density=True, label=f"{r['name'][:26]}  ({r['verdict']})")
axh.set_yscale("log"); axh.set_xlabel("weight change  $\\Delta w$")
axh.set_ylabel("density (log)"); axh.set_title("Distribution of weight changes (all-weights tunes)")
axh.legend(fontsize=7)

colors = {"held": "C2", "REGRESSED": "C3", "n/a": "0.6"}
y = np.arange(len(rows))
axb.barh(y, [r["rel"] * 100 for r in rows],
         color=[colors[r["verdict"]] for r in rows])
axb.set_yticks(y); axb.set_yticklabels([r["name"][:30] for r in rows], fontsize=7)
axb.set_xlabel("relative L2 weight change  $\\|\\Delta\\theta\\| / \\|\\theta_0\\|$  (%)")
axb.set_title("How far each tune moved the weights (color = controls verdict)")
fig.tight_layout()
out = OUT / "weight_diagnostics.png"
fig.savefig(out, dpi=150)
json.dump(rows, open(OUT / "weight_diagnostics.json", "w"), indent=1)
print(f"\nwrote {out}")
