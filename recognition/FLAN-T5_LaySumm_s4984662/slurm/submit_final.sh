#!/bin/bash
# submit_final.sh - submit all headline runs in one go (safe to re-run: skips finished jobs).
# Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662) | COMP3710 Project 2.6
#
# Usage (from the project folder on Rangpur, after git pull):
#   bash slurm/submit_final.sh
#
# Then disconnect. Check in the morning with:
#   squeue --me
#   ls -la runs/*/metrics.json
#   grep -H 'validation:' runs/*_*.out | tail
#
# Learning rates: from the Colab sweep (dev ROUGE-Lsum). Change here only if you re-sweep.
# Batch: micro-batch 8 x grad_accum 2 (= effective 16). Rangpur smoke at batch 16 hit ~29 GiB;
# the split keeps peak VRAM safer overnight without changing the effective batch size.
# Time: Colab timed ~3.5 h train+val per LoRA/full run and ~70 min for zero-shot; 5 h / 2 h
# leave headroom so Slurm does not kill the job during the final validation pass.

set -euo pipefail

LORA_LR=3e-3
FULL_LR=3e-4
BATCH="--batch_size 8 --grad_accum 2"

if [[ ! -f train.py || ! -f slurm/train.sh ]]; then
  echo "Run this from recognition/FLAN-T5_LaySumm_s4984662/ (need train.py and slurm/train.sh)" >&2
  exit 1
fi

mkdir -p runs

submit_one() {
  local name="$1" wall="$2"
  shift 2
  if [[ -f "runs/${name}/metrics.json" ]]; then
    echo "skip (already finished): ${name}"
    return 0
  fi
  # Fresh output dir so a half-finished local run cannot confuse the next attempt.
  rm -rf "runs/${name}"
  local jid
  jid=$(sbatch --parsable \
    --job-name="rrs-${name}" \
    --time="${wall}" \
    --output="runs/${name}_%j.out" \
    --error="runs/${name}_%j.err" \
    slurm/train.sh \
    --output_dir "runs/${name}" \
    ${BATCH} \
    "$@")
  echo "submitted ${name}  job ${jid}  wall ${wall}"
}

# Four independent jobs (up to 4 of 10 class GPUs). Order does not matter.
submit_one "zeroshot"            "02:00:00" --mode zeroshot
submit_one "lora_r8_lr${LORA_LR}"  "05:00:00" --mode lora --lora_r 8  --lr "${LORA_LR}"
submit_one "lora_r16_lr${LORA_LR}" "05:00:00" --mode lora --lora_r 16 --lr "${LORA_LR}"
submit_one "full_lr${FULL_LR}"     "05:00:00" --mode full --lr "${FULL_LR}"

echo
echo "Queue:"
squeue --me
echo
echo "When every job shows runs/<name>/metrics.json, the overnight batch is done."
echo "Quota check (home fills with full-FT checkpoints ~1 GB each):  du -sh ~/hf_cache runs 2>/dev/null; quota -s 2>/dev/null || true"
