#!/usr/bin/env bash
# Larger fine-tune runs for a rented GPU pod, each guarded by the property
# benchmark (pre/post whack-a-mole diff). Converged 108-atom cells (--reps 3).
#
# One-time setup on the pod:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   uv venv --python 3.11 && source .venv/bin/activate
#   uv pip install -e ".[mace]" && uv pip install pytest
#
# Then:  bash scripts/run_pod.sh
# Outputs (checkpoints *.pt, JSON logs, benchmark panels) land in examples/output/.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

DEV=${DEVICE:-cuda}          # override: DEVICE=cpu bash scripts/run_pod.sh
REPS=${REPS:-3}              # 3 -> 108-atom converged cells
STEPS=${STEPS:-150}

echo "=== 0. baseline benchmark (pretrained MACE-MP-0) ==="
python examples/mace_benchmark.py --device "$DEV" --reps "$REPS" --out mace_benchmark_pretrained.json

echo "=== 1. Cu-Ni miscibility gap (consolute) — flagship ==="
python examples/mace_finetune_stage2.py --reps "$REPS" --steps "$STEPS" \
    --lr 2e-3 --reg 0.5 --system cuni --device "$DEV" --benchmark

echo "=== 2. Ni-Al gamma-prime solvus ==="
python examples/mace_finetune_stage2.py --reps "$REPS" --steps "$STEPS" \
    --lr 2e-3 --reg 0.5 --system nial --device "$DEV" --benchmark

echo "=== 3. Si fcc-diamond lattice stability (Al-Si Si-side reference) ==="
python examples/mace_finetune_lattice.py --pair Si_fcc_dia --reps "$REPS" \
    --steps "$STEPS" --device "$DEV" --benchmark

echo "=== 4. Zn fcc-hcp lattice ordering (Cu-Zn Zn reference) ==="
python examples/mace_finetune_lattice.py --pair Zn_fcc_hcp --reps "$REPS" \
    --steps "$STEPS" --device "$DEV" --benchmark

echo "=== 5. JOINT all-weights tune (the headline) — all four targets at once ==="
# tuning all ~3.85M weights vs 4 phase-diagram targets: the reverse-mode regime
# forward-mode can't touch. reg small since it sums over ~200x more params.
python examples/mace_finetune_joint.py --reps "$REPS" --steps "$STEPS" \
    --subset all --reg 0.01 --device "$DEV" --benchmark

echo "=== done. Each run printed a pre->post benchmark diff; a tune is only"
echo "    credible if its controls held. Checkpoints + panels in examples/output/."
