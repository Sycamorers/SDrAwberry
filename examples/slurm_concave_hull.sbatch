#!/usr/bin/env bash
#SBATCH --job-name=sdrawberry-hull
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

IN_PCD="${1:?Usage: sbatch examples/slurm_concave_hull.sbatch /path/to/cluster.ply /path/to/out_dir}"
OUT_DIR="${2:?Usage: sbatch examples/slurm_concave_hull.sbatch /path/to/cluster.ply /path/to/out_dir}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sdrawberry

python scripts/run_concave_hull.py \
  --in_pcd "${IN_PCD}" \
  --out_dir "${OUT_DIR}" \
  --alpha_voxel 0.006 \
  --alpha_m_list "2,3,4,5,6,7,8,9,10,15,20" \
  --min_triangles 2000 \
  --fill_holes \
  --save_only_selected
