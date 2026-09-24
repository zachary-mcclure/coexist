#!/usr/bin/env bash
# Generate the FAILING tune checkpoints so weight_diagnostics.py can line up
# work-vs-fail: how far a broken tune moved the weights, and which module, vs a
# credible one. Runs the known control-breakers -- each now saves its checkpoint,
# benchmark verdict, and weight-movement stats (distinct tags, no overwrite).
#
#   bash scripts/failcases_pod.sh
#   # knobs: DEVICE=cpu REPS=2 STEPS=100 bash scripts/failcases_pod.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
DEV=${DEVICE:-cuda}; REPS=${REPS:-3}; STEPS=${STEPS:-150}

echo "=== FAIL 1: barrier tune WITHOUT the Ag-Cu anchor (broke Ag-Cu Omega 377->413) ==="
python examples/mace_finetune_neb.py --element Cu --reps "$REPS" --images 5 \
    --steps "$STEPS" --subset all --reg 0.1 --anchor 0 --target 0.71 --device "$DEV" --benchmark

echo "=== FAIL 2: diagram tune too-loose reg 0.01 (broke Ag lattice 4.165->4.187) ==="
python examples/mace_finetune_joint.py --reps "$REPS" --steps "$STEPS" \
    --subset all --reg 0.01 --anchor 5 --device "$DEV" --benchmark

echo "=== FAIL 3: diagram tune reg 0.003 (broke 2 controls) ==="
python examples/mace_finetune_joint.py --reps "$REPS" --steps "$STEPS" \
    --subset all --reg 0.003 --anchor 5 --device "$DEV" --benchmark

echo "=== regenerate the diagnostic (credible + failing tunes now lined up) ==="
python examples/weight_diagnostics.py
echo "done -- examples/output/weight_diagnostics.png (color = controls verdict)"
