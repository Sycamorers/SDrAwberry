#!/usr/bin/env bash
#SBATCH --job-name=sdrawberry-cluster
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

SCENE_DIR="${1:?Usage: sbatch examples/slurm_clustering.sbatch /path/to/scene}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sdrawberry

python scripts/run_clustering.py \
  --input_pcd "${SCENE_DIR}/merged/merged_global.ply" \
  --out_dir "${SCENE_DIR}/clusters" \
  --dbscan_eps 0.015 \
  --dbscan_min_pts 120 \
  --plane_dist_threshold 0.005 \
  --z_axis_inverted
