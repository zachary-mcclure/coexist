#!/usr/bin/env bash
# Map the regularizer Pareto for the joint all-weights tune. Too-loose reg
# breaks a control (e.g. pure-Ag lattice); too-tight caps a target; somewhere
# between is the credible window where all four phase-diagram targets land AND
# every control holds. Writes one benchmarked JSON per reg, then the figure.
#
#   bash scripts/reg_sweep.sh
#   # knobs: DEVICE=cpu REPS=2 STEPS=100 REGS="0.01 0.1 1.0" ANCHOR=5 bash scripts/reg_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

DEV=${DEVICE:-cuda}
REPS=${REPS:-3}                          # 3 -> 108-atom converged cells
STEPS=${STEPS:-150}
ANCHOR=${ANCHOR:-5}
REGS=${REGS:-"0.003 0.01 0.03 0.1 0.3 1.0"}

for reg in $REGS; do
  echo "=== joint all-weights tune, reg=$reg (anchor $ANCHOR) ==="
  python examples/mace_finetune_joint.py --reps "$REPS" --steps "$STEPS" \
      --subset all --reg "$reg" --anchor "$ANCHOR" --device "$DEV" --benchmark
done

echo "=== Pareto: targets-land vs controls-hold across reg ==="
python examples/plot_reg_sweep.py
echo "done — table above; figure at examples/output/reg_sweep_pareto.png"
