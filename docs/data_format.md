# Data Format

This page documents the expected local scene layout and the outputs produced by each SDrAwberry stage.

## Input Scene

Each scene is a synchronized multi-camera RGB sequence:

```text
scene_dir/
  images/
    <camera_id_1>/
      rgb/
        rgb_<frame_id>.png
    <camera_id_2>/
      rgb/
        rgb_<frame_id>.png
    <camera_id_3>/
      rgb/
        rgb_<frame_id>.png
```

Requirements:

- Camera folders must live directly under `images/`.
- RGB files must live under `images/<camera_id>/rgb/`.
- File names must follow `rgb_<frame_id>.png`.
- `<frame_id>` must be a decimal integer. Leading zeros are preserved in batch output names.
- All cameras should share the same frame ids when possible. Missing images are reported in `batch_summary.json`.

## Batched Scene

`run_batching.py` writes flattened mini-scenes:

```text
scene_dir/
  batches/
    01/
      images/
        <frame_id>_<camera_id>.png
    02/
      images/
        <frame_id>_<camera_id>.png
```

Batch names are deterministic 1-based indices. Values below 100 use two digits, such as `01`, `02`, and `99`; larger values expand naturally, such as `100`.

## DA3 / COLMAP Outputs

When DA3 is run, each batch exports a COLMAP-compatible sparse model:

```text
scene_dir/
  sparse/
    01/
      cameras.bin
      images.bin
      points3D.bin
      points.ply
```

`points.ply` is optional but required by the fusion stage. It can be created during batching with `--da3_export_ply` or produced separately from an existing COLMAP model.

## Fusion Outputs

```text
scene_dir/
  merged/
    merged_global.ply
```

The fused point cloud is voxelized and duplicate-suppressed. Its scale is still reconstruction scale unless a reference-object scale factor has been applied later.

## Clustering Outputs

```text
scene_dir/
  clusters/
    cluster_0001.ply
    cluster_0002.ply
    summary_clustering.json
```

Cluster file numbering is sequential and stable for a given clustering run. The JSON summary records source DBSCAN cluster ids, point counts, HSV counts, plane parameters, and filtering decisions.

## Concave Hull Outputs

```text
scene_dir/
  meshes/
    cluster_0001/
      cluster_0001_m*.ply
      summary_concave.json
```

Each mesh name includes the selected alpha multiplier `m`. `summary_concave.json` records all attempted multipliers, watertightness, triangle counts, repair settings, and volume.

## Dataset Placeholder

Dataset: https://huggingface.co/datasets/Sycamorers/SDrAwberry

Status: coming soon / upload in progress.

After download, place scenes under a local data directory and pass each scene path with `--scene_dir` or in `configs/example_scene.yaml`.
