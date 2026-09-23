# Reproducible environment for the coexist fine-tune battery on a GPU pod.
# Base image ships torch already matched to a CUDA runtime, so there's no
# runtime torch/CUDA guessing — the #1 source of pod friction.
#
# Build + push (once):
#   docker build -t <you>/coexist:latest .
#   docker push  <you>/coexist:latest
# Then on RunPod: deploy a pod with this image and run `bash scripts/run_pod.sh`.
#
# NB: not built/tested locally (no GPU here) — verify the base tag matches a
# torch that mace-torch accepts, and that the base torch survives the install
# (the constraint below pins it so pip can't swap in a CPU/other-CUDA build).
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /coexist
COPY . /coexist

RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install coexist + its stack, but keep the base image's CUDA-matched torch
# (constraint prevents mace-torch's deps from swapping torch out).
RUN python -c "import torch; open('/tmp/torch.txt','w').write('torch=='+torch.__version__)" \
 && pip install --no-cache-dir -c /tmp/torch.txt -e ".[mace]" \
 && pip install --no-cache-dir pytest

# Fail the build early if anything's off.
RUN python -c "import torch, mace, coexist, ase, jax; \
print('env OK — torch', torch.__version__, '| cuda build', torch.version.cuda)"

CMD ["bash", "scripts/run_pod.sh"]
