#!/usr/bin/env bash
# MUFASA SFT on a rented 8-GPU box. Run from /workspace after cloning.
#
#   export HF_TOKEN=hf_xxx
#   export HF_USER=your_user
#   bash run.sh gemma3-1b-cpt-v4
#
# Everything comes from the Hub and goes back to the Hub. The instance disk
# holds only transient checkpoints.
set -euo pipefail

RUN="${1:?usage: run.sh <cpt-run-name>}"
: "${HF_TOKEN:?export HF_TOKEN first}"
: "${HF_USER:?export HF_USER first}"

GPUS="$(nvidia-smi --list-gpus | wc -l)"
echo "== ${GPUS} GPU(s) detected"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

pip install -q -U "transformers==5.5.0" "trl==0.24.0" "datasets==4.3.0" \
                  "accelerate>=1.0" "huggingface_hub>=0.34.0" hf_transfer
pip install -q flash-attn --no-build-isolation || \
  echo "!! flash-attn unavailable - pass --attn sdpa"

export HF_HUB_ENABLE_HF_TRANSFER=1     # fast parallel download of the weights
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
# huggingface_hub reads HF_TOKEN directly; no CLI login needed, and the
# CLI has been renamed between versions.
export HF_TOKEN
python - <<'EOF'
import os
from huggingface_hub import HfApi
who = HfApi(token=os.environ["HF_TOKEN"]).whoami()
print("authenticated as", who["name"])
EOF

# Settings follow the card. A 1B model plus optimizer is ~12 GB; what varies is
# how much room is left for activations, which decides the batch and whether
# activations must be recomputed.
VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
CAP="$(python -c 'import torch;print("%d%d" % torch.cuda.get_device_capability(0))')"

if   [ "${VRAM}" -ge 140000 ]; then BATCH=32; ACCUM=1; GC=""
elif [ "${VRAM}" -ge 70000  ]; then BATCH=16; ACCUM=2; GC=""
elif [ "${VRAM}" -ge 40000  ]; then BATCH=16; ACCUM=2; GC="--grad-checkpointing"
else                                BATCH=8;  ACCUM=2; GC="--grad-checkpointing"
fi

# Consumer Blackwell (sm_120) and Ada (sm_89) cap shared memory at 99 KB per
# block, which the flex-attention backward kernel exceeds. sdpa never compiles
# that kernel, so it sidesteps the crash entirely.
case "${CAP}" in
  120|89) ATTN="sdpa" ;;
  *)      ATTN="flash_attention_2" ;;
esac

echo "== ${VRAM} MB/card, sm_${CAP} -> batch ${BATCH} x accum ${ACCUM} ${GC:-(no checkpointing)}, attn ${ATTN}"

torchrun --standalone --nproc_per_node "${GPUS}" train_sft.py \
  --base   "${HF_USER}/mufasa-${RUN}" \
  --data   "${HF_USER}/mufasa-sft-mixed" \
  --output "${HF_USER}/mufasa-${RUN/-cpt/}-sft" \
  --epochs 1 \
  --batch-size "${BATCH}" \
  --accum "${ACCUM}" \
  ${GC} \
  --attn "${ATTN}" \
  --lr 2e-5 \
  --max-seq-length 4096 \
  --eval-steps 100 \
  --save-steps 200 \
  --local-dir /workspace/sft_run

echo "== done. weights are on the Hub; this box can be destroyed."
