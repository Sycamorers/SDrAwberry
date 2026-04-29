#!/usr/bin/env bash
#SBATCH --job-name=sdrawberry-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

SCENE_DIR="${1:?Usage: sbatch examples/slurm_batching.sbatch /path/to/scene}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sdrawberry

python scripts/run_batching.py \
  --scene_dir "${SCENE_DIR}" \
  --batch_size 10 \
  --overlap_frames 5 \
  --symlink \
  --da3_export_ply
