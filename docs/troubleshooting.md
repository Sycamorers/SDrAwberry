# Troubleshooting

## DA3 CLI Not Found

Error:

```text
DA3 CLI executable 'da3' was not found on PATH
```

Install Depth-Anything-3 separately and confirm:

```bash
which da3
da3 --help
```

To prepare batches without DA3, run batching with `--prepare_only`.

## Missing Images Folder

The batching stage expects:

```text
scene_dir/images/<camera_id>/rgb/rgb_<frame_id>.png
```

Check the path passed to `--scene_dir` and the file naming pattern.

## Missing COLMAP Model During Fusion

Fusion needs `cameras`, `images`, and `points3D` files as `.bin` or `.txt` under each selected `sparse/<batch_id>/` directory. It also needs the per-batch PLY file named by `--ply`, usually `points.ply`.

Run batching with `--da3_export_ply` or export PLY files from existing COLMAP models before fusion.

## pycolmap Import Error

Install `pycolmap` in the active environment:

```bash
pip install pycolmap
```

This repository does not vendor COLMAP model readers.

## Empty Point Clouds

An empty `points.ply` usually means DA3 exported no confident 3D points or a downsampling value is too large. Try reducing DA3 confidence thresholds or reducing `--pre_voxel`.

## Missing Colors During Clustering

HSV filtering requires RGB colors in the fused PLY. Confirm the per-batch PLY files have colors and that fusion preserved them.

## HSV Finds No Plants

Outdoor illumination and camera white balance can shift HSV values. Tune:

- `--leaf_h_min`, `--leaf_h_max`
- `--leaf_s_min`, `--leaf_v_min`
- stem/yellow/red thresholds

## DBSCAN Finds No Clusters

Increase `--dbscan_eps`, lower `--dbscan_min_pts`, or enable `--use_auto_scale` for scenes with very different reconstruction scale.

## All Clusters Are Filtered Out

Lower `--cluster_min_size`, disable ground-like filtering with `--no_filter_groundlike_clusters`, or relax:

- `--ground_median_margin_factor`
- `--ground_height_range_factor`
- `--ground_planarity_ratio_max`

## No Watertight Concave Hull

Try:

```bash
python scripts/run_concave_hull.py \
  --in_pcd /path/to/cluster.ply \
  --out_dir /path/to/meshes/cluster \
  --alpha_voxel 0.006 \
  --alpha_m_list "2,3,4,5,6,7,8,9,10,15,20,30" \
  --fill_holes \
  --hole_size_factor 5 \
  --normalize
```

Watertightness depends on point density, holes, outliers, and the chosen alpha scale.
