#!/usr/bin/env bash
#SBATCH --job-name=sdrawberry-fusion
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

SCENE_DIR="${1:?Usage: sbatch examples/slurm_fusion.sbatch /path/to/scene}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sdrawberry

python scripts/run_fusion.py \
  --base_dir "${SCENE_DIR}" \
  --sparse_subfolder sparse \
  --ply points.ply \
  --out "${SCENE_DIR}/merged/merged_global.ply" \
  --window_kdt 4 \
  --window_match 2 \
  --pre_voxel 0.003 \
  --voxel 0.002
