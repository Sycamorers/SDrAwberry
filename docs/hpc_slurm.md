# HPC and Slurm Notes

The batching stage can call DA3 and usually benefits from a GPU. Fusion, clustering, and concave hull reconstruction are often CPU-heavy and memory-sensitive, although Open3D may still use optimized native libraries.

## Recommended Workflow

1. Stage input images on shared or node-local storage.
2. Run batching plus DA3 on a GPU node.
3. Run fusion after all batch `sparse/<batch_id>/points.ply` files exist.
4. Run clustering and concave hull reconstruction as separate jobs so thresholds can be tuned without rerunning DA3.

## Example Commands

```bash
sbatch examples/slurm_batching.sbatch /path/to/scene
sbatch examples/slurm_fusion.sbatch /path/to/scene
sbatch examples/slurm_clustering.sbatch /path/to/scene
sbatch examples/slurm_concave_hull.sbatch /path/to/scene/clusters/cluster_0001.ply /path/to/scene/meshes/cluster_0001
```

## Environment

The example scripts assume:

- a conda environment named `sdrawberry`,
- this repository has been installed with `pip install -e .`,
- the external `da3` CLI is available on `PATH` for the batching job.

If your cluster uses modules, add the relevant `module load ...` lines before `conda activate`.

## Practical Tips

- Use `--symlink` when the filesystem supports it to avoid duplicating input images.
- Reduce `--da3_process_res` if DA3 runs out of GPU memory.
- Reduce `--pre_voxel` and `--voxel` only when you need denser output and have enough memory.
- Keep raw datasets and generated point clouds outside git-tracked directories.
