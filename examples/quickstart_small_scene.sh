#!/usr/bin/env bash
set -euo pipefail

SCENE_DIR="${1:-/path/to/scene}"

python scripts/run_batching.py \
  --scene_dir "${SCENE_DIR}" \
  --batch_size 10 \
  --overlap_frames 5 \
  --symlink \
  --da3_export_ply

python scripts/run_fusion.py \
  --base_dir "${SCENE_DIR}" \
  --sparse_subfolder sparse \
  --ply points.ply \
  --out "${SCENE_DIR}/merged/merged_global.ply" \
  --window_kdt 4 \
  --window_match 2 \
  --pre_voxel 0.003 \
  --voxel 0.002

python scripts/run_clustering.py \
  --input_pcd "${SCENE_DIR}/merged/merged_global.ply" \
  --out_dir "${SCENE_DIR}/clusters" \
  --dbscan_eps 0.015 \
  --dbscan_min_pts 120 \
  --plane_dist_threshold 0.005 \
  --z_axis_inverted

python scripts/run_concave_hull.py \
  --in_pcd "${SCENE_DIR}/clusters/cluster_0001.ply" \
  --out_dir "${SCENE_DIR}/meshes/cluster_0001" \
  --alpha_voxel 0.006 \
  --alpha_m_list "2,3,4,5,6,7,8,9,10,15,20" \
  --min_triangles 2000 \
  --fill_holes \
  --save_only_selected
