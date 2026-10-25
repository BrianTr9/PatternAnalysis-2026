#!/bin/bash
# train.sh - Slurm job: train and evaluate one configuration (zero-shot, LoRA or full FT).
# Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662) | COMP3710 Project 2.6
#
# Submit from the project folder on the Rangpur login node; every argument after the script
# name is passed to train.py (see `python train.py --help`):
#   sbatch slurm/train.sh --mode zeroshot
#   sbatch slurm/train.sh --mode lora --lora_r 8 --lr 3e-4
#   sbatch slurm/train.sh --mode full --lr 5e-5
#   sbatch --time=00:20:00 slurm/train.sh --mode lora --epochs 1 --max_train_samples 200 \
#          --max_eval_samples 100 --output_dir runs/smoke        # short check before a real run
# Then watch it with `squeue --me` (R = running; PD (PartitionConfig) = wrong account) and
# `tail -f run_<jobid>.out`. Results are written to runs/<name>/ (git-ignored).
#
# Rangpur rules found by testing (see HPC_Notes.md): --account=comp3710 is required, and --mem
# must NOT be set (the job is rejected at submit time). One A100 per node, so one process.
#SBATCH --job-name=rrs-train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=run_%j.out
#SBATCH --error=run_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# A batch job starts from a clean environment, so activate conda and set the cache here.
source "$HOME/miniconda3/bin/activate"
conda activate torch
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
echo "code version: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "arguments: $@"

# torchrun with one process per GPU (train.py also runs plain, and with several processes).
# The rendezvous stays on this node: loopback address and a port derived from the job id.
torchrun --nnodes=1 --nproc_per_node=1 --local_addr=127.0.0.1 \
         --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:$((20000 + SLURM_JOB_ID % 20000)) \
         train.py "$@"
