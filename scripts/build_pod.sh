#!/usr/bin/env bash
# One-shot pod setup: (clone if needed) + uv venv + torch matched to the pod's
# CUDA driver + GPU verify. Idempotent-ish; safe to re-run.
#
# From inside the repo:   bash scripts/build_pod.sh
# Bare pod (bootstrap):
#   curl -LsSf https://raw.githubusercontent.com/zachary-mcclure/coexist/main/scripts/build_pod.sh | bash
#
# Then run the battery:   bash scripts/run_pod.sh
set -uo pipefail

# 0. make sure we're in the coexist repo (clone if bootstrapped on a bare pod)
if [ ! -f pyproject.toml ] || ! grep -q 'name = "coexist"' pyproject.toml 2>/dev/null; then
  echo "=== cloning coexist ==="
  git clone https://github.com/zachary-mcclure/coexist.git
  cd coexist
fi

# 1. uv
if ! command -v uv >/dev/null 2>&1; then
  echo "=== installing uv ==="
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source "$HOME/.local/bin/env" 2>/dev/null || true

# 2. venv + install, with torch matched to the driver
echo "=== venv + install (torch-backend=auto) ==="
uv venv --python 3.11
source .venv/bin/activate
if ! uv pip install -e ".[mace]" --torch-backend=auto; then
  echo "!! this uv lacks --torch-backend; installing then reinstalling torch"
  uv pip install -e ".[mace]"
  uv pip install --reinstall torch --torch-backend=auto || true
fi
uv pip install pytest

# 3. verify the GPU is actually usable
echo "=== GPU check ==="
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "  cuda: True — ready"
else
  echo "  cuda: FALSE — reinstalling torch matched to the driver…"
  uv pip install --reinstall torch --torch-backend=auto || true
  if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "  cuda: True — ready"
  else
    echo "  !! still no CUDA. Check 'nvidia-smi' and match its CUDA version manually, e.g.:"
    echo "     uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124"
    echo "     (or run on CPU: DEVICE=cpu bash scripts/run_pod.sh)"
  fi
fi

echo ""
echo "=== build done. Now run the battery (run_pod.sh self-activates the venv): ==="
echo "    bash scripts/run_pod.sh"
echo "    # knobs: DEVICE=cpu  REPS=2  STEPS=150  bash scripts/run_pod.sh"
