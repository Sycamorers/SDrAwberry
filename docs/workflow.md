# Workflow

SDrAwberry is organized as four explicit stages. Each stage can be run independently as long as its input files already exist.

## 1. Batching

`scripts/run_batching.py` scans `scene_dir/images/<camera_id>/rgb/rgb_<frame_id>.png`, groups files by frame id, and creates overlapping temporal batches under `scene_dir/batches/`.

The stage can also call the external DA3 CLI for each prepared batch and export COLMAP-compatible sparse models under `scene_dir/sparse/`. Use `--prepare_only` to create batches without running DA3.

Important options:

- `--batch_size`: number of temporal frames per batch.
- `--overlap_frames`: overlap between adjacent batches.
- `--symlink`: symlink images instead of copying them.
- `--da3_export_ply`: export `points.ply` from each COLMAP model.

## 2. Fusion

`scripts/run_fusion.py` streams per-batch `points.ply` files and COLMAP sparse models. It extracts shared COLMAP track observations across neighboring batches to seed 3D-3D correspondences.

Relative poses are estimated with roll-constrained alignment: one rotation around the reconstruction y-axis plus translation. This is intended for near-linear bed-aligned traversal where the main platform motion is along the bed and roll drift is the dominant constrained orientation component.

The fusion stage applies:

- match-seed local duplicate suppression,
- sliding-window geometric duplicate suppression,
- voxel-hash accumulation into the final global cloud.

## 3. Plant Clustering

`scripts/run_clustering.py` loads the fused point cloud, estimates the dominant support plane, computes signed plane-relative distances, and applies HSV filtering for leaves and stems/petioles.

Plant candidate points are clustered with DBSCAN. Clusters can be filtered by point count and rejected as ground-like when they are near the support plane, thin in plane-relative height, and planar by PCA.

## 4. Concave Hull Volume

`scripts/run_concave_hull.py` reconstructs alpha-shape concave hull meshes for plant clusters or reference-box point clouds. It sweeps alpha multipliers `m`, optionally fills holes, keeps the largest connected component, checks watertightness, and reports volume.

Use `--use_scale --scale <factor>` only when the scale factor converts reconstruction units to meters. When scale is used, volume is reported in `cm^3`; otherwise it is reported in reconstruction `unit^3`.

## Full Pipeline

Edit `configs/example_scene.yaml`, then run:

```bash
python scripts/run_full_pipeline.py --config configs/example_scene.yaml
```

The full runner is a convenience wrapper. For debugging and parameter tuning, run each stage separately and inspect its intermediate outputs.
