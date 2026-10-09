#!/bin/bash
# env_check.sh - Slurm job: sanity check of the Rangpur A100 environment.
# Author : Truong Trung Bao (s4984662) | COMP3710 Project 2.6
#
# Submit from the project folder:  sbatch slurm/env_check.sh
#
# Rangpur behaviour observed on 2026-10-09 (the course guide's example job.sh, which has
# --mem=16G and no --account, did not run as written):
#   * --account=comp3710 is REQUIRED. Without it the job stayed PENDING with
#     Reason=PartitionConfig, even with --mem removed (the partition only allows account
#     comp3710). Note: `sbatch --test-only` still reported success in that case, so it does
#     not catch this problem; check `squeue --me` after a real submit.
#   * Do NOT pass --mem. Rangpur rejects it at submit time, even together with
#     --account=comp3710: "Memory specification can not be satisfied / Requested node
#     configuration is not available" (Slurm lists only ~1 MB of schedulable memory per
#     node). The node itself has ~62 GB of RAM.
#SBATCH --job-name=env-check
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=env_check_%j.out
#SBATCH --error=env_check_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
free -h

# A batch job starts from a clean environment, so activate conda inside the script.
source "$HOME/miniconda3/bin/activate"
conda activate torch
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1   # weights/data were pre-downloaded

python slurm/env_check.py
