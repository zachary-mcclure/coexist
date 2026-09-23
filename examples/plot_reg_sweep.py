"""
Pareto of the regularizer sweep — the paper's whack-a-mole figure.

Tuning ~3.85M MACE weights against four phase-diagram targets is wildly
under-determined: the null space lets you satisfy the targets while quietly
wrecking a control. The regularizer (stay-near-pretrained) shrinks that null
space. Too loose -> a control breaks; too tight -> a target stops landing.
Somewhere between is the credible window where all four targets land AND every
control holds.

Reads the per-reg JSONs written by `mace_finetune_joint.py --benchmark` (each
carries its pre/post benchmark panels) and plots, against reg:
  * mean target error   (|value-ref| in units of the property tolerance; want <1)
  * worst control drift (increase in |error| vs pretrained, in tolerances; want <1)
shading the reg range where the tune stayed credible.

Run:  python examples/plot_reg_sweep.py
"""
import json, glob
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mace_benchmark import REF          # {property: (ref, role, tol)}

OUT = Path(__file__).parent / "output"

rows = []
for f in sorted(glob.glob(str(OUT / "mace_finetune_joint_*_reg*_anc*.json"))):
    d = json.load(open(f))
    if "bench_pre" not in d:            # skip runs without --benchmark
        continue
    b, t = d["bench_pre"], d["bench_post"]
    n_tgt = n_ctrl = hit = broke = 0
    mean_terr = worst_ctrl = 0.0
    for k, (ref, role, tol) in REF.items():
        if k not in t:
            continue
        if role == "target":
            n_tgt += 1
            mean_terr += abs(t[k] - ref) / tol
            hit += abs(t[k] - ref) <= tol
        elif role == "control":
            n_ctrl += 1
            drift = (abs(t[k] - ref) - abs(b[k] - ref)) / tol   # +>0 worse than pretrained
            worst_ctrl = max(worst_ctrl, drift)
            broke += drift > 1.0
    rows.append(dict(reg=d["reg"], hit=hit, n_tgt=n_tgt, broke=broke, n_ctrl=n_ctrl,
                     mean_terr=mean_terr / max(n_tgt, 1), worst_ctrl=worst_ctrl,
                     credible=d.get("controls_ok")))

if not rows:
    raise SystemExit("no benchmarked joint JSONs found — run scripts/reg_sweep.sh first")
rows.sort(key=lambda r: r["reg"])

print(f"\n{'reg':>8}{'targets_hit':>13}{'ctrl_broke':>12}{'mean_tgt_err':>14}{'worst_ctrl':>12}{'credible':>10}")
print("-" * 79)
for r in rows:
    print(f"{r['reg']:>8}{r['hit']:>7}/{r['n_tgt']:<5}{r['broke']:>6}/{r['n_ctrl']:<5}"
          f"{r['mean_terr']:>14.2f}{r['worst_ctrl']:>12.2f}{str(bool(r['credible'])):>10}")

regs = [r["reg"] for r in rows]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(regs, [r["mean_terr"] for r in rows], "o-", color="C3",
        label="mean target error  (|err| / tol)")
ax.plot(regs, [r["worst_ctrl"] for r in rows], "s-", color="C0",
        label="worst control drift  (Δ|err| / tol)")
ax.axhline(1.0, ls=":", color="gray", lw=1)          # tolerance line: below = within noise
ax.set_xscale("log")
ax.set_xlabel("regularizer  (weight on staying near pretrained)")
ax.set_ylabel("error / drift  (in property tolerances; <1 = holds)")
cred = [r["reg"] for r in rows if r["credible"]]
if cred:
    ax.axvspan(min(cred) * 0.85, max(cred) * 1.18, color="C2", alpha=0.12,
               label="credible window (all targets land, all controls hold)")
ax.legend(fontsize=8, loc="upper center")
ax.set_title("Regularizer sweep: the credible window for a joint 4-target tune")
fig.tight_layout()
fig.savefig(OUT / "reg_sweep_pareto.png", dpi=150)
print(f"\nwrote {OUT / 'reg_sweep_pareto.png'}")
