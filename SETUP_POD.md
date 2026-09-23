# Running the fine-tune battery on a GPU pod

Fresh pod (RunPod PyTorch template or bare CUDA), clone through run:

```bash
# 1. clone
git clone https://github.com/zachary-mcclure/coexist.git
cd coexist

# 2. uv + venv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.11
source .venv/bin/activate

# 3. install — torch matched to the pod's actual CUDA driver
uv pip install -e ".[mace]" --torch-backend=auto
uv pip install pytest

# 4. VERIFY the GPU before running
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # must be: cuda: True

# 5. run (~30-60 min on a 4090)
bash scripts/run_pod.sh
```

If step 4 prints `cuda: False`:
```bash
uv pip install --reinstall torch --torch-backend=auto
# or match nvidia-smi's CUDA version manually, e.g.:
#   uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124
```

Notes:
- Ignore `jax ... falling back to cpu` — the construction runs on CPU by design; MACE uses the GPU.
- Outputs (checkpoints `*.pt`, benchmark panels, JSON logs) land in `examples/output/`.
  Pull them back, then STOP the pod so billing ends.
- `run_pod.sh` env knobs: `DEVICE=cpu`, `REPS=2` (32-atom, faster/rougher), `STEPS=150`.
- MACE-MP-0 small is light; a single 4090/A10 (or even CPU) is plenty — don't over-buy.
