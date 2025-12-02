#!/usr/bin/env python3
# Streamed merge with COLMAP matches + yaw-only (around Y-axis) ICP refinement + windowed KD dedup + voxel hash, float32

import os
import sys
import math
import argparse
from collections import defaultdict, deque

import numpy as np
import open3d as o3d

# =================== DEFAULT CONFIG ===================
BASE_DIR = "/home/ruoyao/Documents/Research_ABE/vggt/data_1201/data_20251201_M5"
SPARSE_SUBFOLDER = "sparse"
PLY_FILENAME = "points.ply"

# Culling radii
MATCH_TO_PLY_RADIUS = 0.0008
GEOM_RADIUS = 0.0015

# Optional speed/cleanup
PRE_DOWNSAMPLE_INCOMING = 0.003

# Streamed merge knobs
WINDOW_KDT = 4      # sliding window size for geometry-based dedup (KD-tree over last N global chunks)
WINDOW_MATCH = 2    # match-seed-based dedup only uses the last N parent batches (0 disables)
VOX = 0.002         # voxel size for voxel hashing (m)
DTYPE = np.float32

# Save
OUT_PLY = "./1110_M4/merged_2930.ply"

# =================== COLMAP I/O (txt/bin) ===================
try:
    from read_write_model import read_model
except Exception:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    from read_write_model import read_model


def load_views_and_points_from_colmap(model_dir, ext=""):
    """
    Load COLMAP model and build:
      - points: 3D point id -> xyz
      - view_index: (image_name, rounded_xy_2d) -> list of 3D point ids
    """
    cameras, images, points3D = read_model(model_dir, ext=ext)

    image_id_to_name = {}
    image_points = {}

    for img in images.values():
        image_id_to_name[img.id] = img.name
        xys = [tuple(round(v, 6) for v in xy) for xy in img.xys.tolist()]
        image_points[img.name] = xys

    points = {}
    view_index = defaultdict(list)

    for p3d in points3D.values():
        points[p3d.id] = p3d.xyz
        img_ids = p3d.image_ids.tolist()
        pt2d_ids = p3d.point2D_idxs.tolist()
        for img_id, pt_idx in zip(img_ids, pt2d_ids):
            img_name = image_id_to_name.get(img_id)
            if img_name is None:
                continue
            if 0 <= pt_idx < len(image_points[img_name]):
                xy = image_points[img_name][pt_idx]
                view_index[(img_name, xy)].append(p3d.id)

    return points, view_index


# =================== UTILITIES ===================
def sorted_batches(base_dir, sparse_subfolder):
    """
    List and sort COLMAP sparse sub-folders in numeric order if possible,
    otherwise lexicographically by folder name.
    """
    sparse_root = os.path.join(base_dir, sparse_subfolder)
    if not os.path.isdir(sparse_root):
        raise FileNotFoundError(f"sparse folder not found: {sparse_root}")

    folders = [
        os.path.join(sparse_root, f)
        for f in os.listdir(sparse_root)
        if os.path.isdir(os.path.join(sparse_root, f))
    ]

    def to_int_or_none(p):
        try:
            return int(os.path.basename(p))
        except Exception:
            return None

    infos = [(to_int_or_none(p), p) for p in folders]

    if any(k is not None for k, _ in infos):
        infos_num = sorted([x for x in infos if x[0] is not None], key=lambda t: t[0])
        infos_str = sorted([x for x in infos if x[0] is None], key=lambda t: os.path.basename(t[1]))
        infos = infos_num + infos_str
    else:
        infos = sorted(infos, key=lambda t: os.path.basename(t[1]))

    return infos


def parse_range(arg):
    """
    Parse batch range string "START:END" into (start, end).
    """
    try:
        start_str, end_str = arg.split(':')
        return int(start_str), int(end_str)
    except Exception:
        raise argparse.ArgumentTypeError("batch_range must be START:END, e.g., 15:75")


def select_batches(all_infos, batch_range=None, max_batches=None):
    """
    Select batch folders based on numeric index range and/or max count.
    """
    selected = []
    if batch_range is None:
        selected = list(all_infos)
    else:
        start, end = batch_range
        if start > end:
            start, end = end, start
        for k, p in all_infos:
            if k is None:
                continue
            if start <= k <= end:
                selected.append((k, p))
    if max_batches is not None and max_batches > 0:
        selected = selected[:max_batches]
    return selected


def find_matches_from_views(view_index_1, view_index_2, points1, points2, K=50):
    """
    From two COLMAP view indices, build 3D-3D correspondences based on shared 2D features:
      - For each shared (image_name, xy_2d), take up to K 3D points from each side.
      - Produce:
          X1: (N,3) from points1
          X2: (N,3) from points2
          MB: unique 3D points from the second set (for seed-based dedup)
    """
    pairs = []
    matched_B = []

    for view in view_index_2:
        if view in view_index_1:
            ids1 = view_index_1[view][:K]
            ids2 = view_index_2[view][:K]
            for id1 in ids1:
                for id2 in ids2:
                    pairs.append((points1[id1], points2[id2]))
            matched_B.extend([points2[idx] for idx in view_index_2[view]])

    if not pairs:
        X1 = np.zeros((0, 3))
        X2 = np.zeros((0, 3))
    else:
        X1, X2 = map(np.array, zip(*pairs))

    MB = np.unique(np.asarray(matched_B), axis=0) if matched_B else np.zeros((0, 3))
    return np.array(X1), np.array(X2), MB


def estimate_similarity_transform(A, B, force_unit_scale=True):
    """
    Umeyama similarity transform solver (A -> B).
    When force_unit_scale=True, the scale is fixed to 1, i.e. SE(3) only.
    Returns (scale, R, t) such that: B ~ s * R * A + t
    """
    assert A.shape == B.shape and A.shape[1] == 3

    muA = A.mean(0)
    muB = B.mean(0)
    X = A - muA
    Y = B - muB

    Sigma = (Y.T @ X) / max(1, len(A))
    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    varA = (X ** 2).sum() / max(1, len(A))

    if force_unit_scale:
        s = 1.0
    else:
        s = np.trace(np.diag(D) @ S) / (varA + 1e-12)

    t = muB - s * (R @ muA)
    return float(s), R, t


def _intvector_to_list(idx):
    """
    Convert Open3D IntVector (CPU/CUDA) to a Python list.
    """
    try:
        return list(idx)
    except TypeError:
        return np.asarray(idx, dtype=np.int64).tolist()


def to_f32_pts(arr):
    """
    Convert array to configured DTYPE (float32).
    """
    arr = np.asarray(arr)
    return arr.astype(DTYPE, copy=False)


# =================== YAW-ONLY SOLVER (R around Y-axis) ===================
def yaw_pairs_refined(X_src, X_tgt, T_init=None, max_iter=3):
    """
    Fixed correspondences, optimizing only yaw (rotation around Y-axis) + full translation:

        argmin_{θ, t} Σ || R_y(θ) X_src[k] + t - X_tgt[k] ||^2

    Strategy:
      1) If T_init is given, project its rotation to yaw-only and use its translation.
         Otherwise, estimate a full SE(3) transform then project to yaw-only.
      2) Refine yaw+translation in closed form for a few iterations, keeping correspondences fixed.

    Returns a 4x4 SE(3) matrix (R_yaw, t).
    """
    X_src = np.asarray(X_src, np.float64)
    X_tgt = np.asarray(X_tgt, np.float64)
    if len(X_src) < 3 or len(X_src) != len(X_tgt):
        return np.eye(4)

    # Initial guess: from T_init if available, else from SE(3) alignment
    if T_init is None:
        _, R0, t0 = estimate_similarity_transform(X_src, X_tgt, force_unit_scale=True)
    else:
        R0 = T_init[:3, :3]
        t0 = T_init[:3, 3]

    yaw = math.atan2(R0[0, 2], R0[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]], np.float64)
    t = t0.copy()

    muX = X_src.mean(0)
    muY = X_tgt.mean(0)

    for _ in range(max_iter):
        # Closed-form yaw update (fixed correspondences, centered)
        Xc = X_src - muX
        Yc = X_tgt - muY
        H = Yc.T @ Xc

        denom = H[0, 0] + H[2, 2]
        if abs(denom) < 1e-12:
            denom = 1e-12

        yaw = math.atan2(H[0, 2] - H[2, 0], denom)
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]], np.float64)

        # Closed-form translation update
        t = muY - R @ muX

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# =================== STREAMED MERGE HELPERS ===================
class VoxelHash:
    """
    Simple voxel hash for streaming accumulation:
      - key: voxel index (ix, iy, iz)
      - value: accumulated sums (sx, sy, sz, cr, cg, cb, n)
    Exported as centroids with average colors.
    """
    def __init__(self, voxel_size):
        self.vs = float(voxel_size)
        # key -> (sx, sy, sz, cr, cg, cb, n)
        self.map = {}

    def add_points(self, pts, cols=None):
        """
        Accumulate points and optional colors into the voxel map.

        pts:  (N,3) float32/64
        cols: (N,3) in [0,1] or None
        """
        vs = self.vs
        mp = self.map
        has_color = cols is not None and len(cols) == len(pts)

        if has_color:
            cols = np.asarray(cols, dtype=np.float32, order="C")

        for i, p in enumerate(pts):
            if not np.all(np.isfinite(p)):
                continue

            k = (
                int(math.floor(p[0] / vs)),
                int(math.floor(p[1] / vs)),
                int(math.floor(p[2] / vs)),
            )

            if k in mp:
                sx, sy, sz, cr, cg, cb, n = mp[k]
                sx += float(p[0])
                sy += float(p[1])
                sz += float(p[2])
                n += 1

                if has_color:
                    r, g, b = cols[i]
                    cr += float(r)
                    cg += float(g)
                    cb += float(b)

                mp[k] = (sx, sy, sz, cr, cg, cb, n)
            else:
                if has_color:
                    r, g, b = cols[i]
                    mp[k] = (
                        float(p[0]), float(p[1]), float(p[2]),
                        float(r), float(g), float(b), 1
                    )
                else:
                    mp[k] = (
                        float(p[0]), float(p[1]), float(p[2]),
                        0.0, 0.0, 0.0, 1
                    )

    def to_pointcloud(self):
        """
        Convert accumulated voxels into an Open3D point cloud with centroids and average colors.
        """
        nvox = len(self.map)
        pts = np.empty((nvox, 3), dtype=np.float32)
        cols = np.empty((nvox, 3), dtype=np.float32)

        i = 0
        any_color = False

        for (sx, sy, sz, cr, cg, cb, n) in self.map.values():
            inv = 1.0 / n
            pts[i] = (sx * inv, sy * inv, sz * inv)
            if cr != 0.0 or cg != 0.0 or cb != 0.0:
                cols[i] = (cr * inv, cg * inv, cb * inv)
                any_color = True
            else:
                cols[i] = (0.0, 0.0, 0.0)
            i += 1

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        if any_color:
            cols = np.clip(cols, 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

        return pcd


def build_kdt(pcd):
    """
    Build a KD-tree for a point cloud, or return None if empty.
    """
    if len(pcd.points) == 0:
        return None
    return o3d.geometry.KDTreeFlann(pcd)


def remove_by_match_seed_local(pcd_local, views_cur, pts3d_cur, recent_parents, k_for_svd, radius):
    """
    Match-seed-based dedup in the local point cloud (before global transform):

      For the current batch's local cloud:
        - For each recent parent batch:
            * Build COLMAP correspondences (parent -> current)
            * Take the matched 3D points MB (on current batch)
            * Remove local points that lie within 'radius' of any MB.

      Only the most recent 'recent_parents' batches are checked for memory control.
    """
    if len(pcd_local.points) == 0 or views_cur is None:
        return pcd_local, 0

    rm_idx_local = set()
    kdt_local = o3d.geometry.KDTreeFlann(pcd_local)

    for parent_path in recent_parents:
        try:
            pts3d_par, views_par = load_views_and_points_from_colmap(parent_path, ext="")
        except Exception:
            continue

        _, _, MB = find_matches_from_views(views_par, views_cur, pts3d_par, pts3d_cur, K=k_for_svd)

        for q in MB:
            c, idx, _ = kdt_local.search_radius_vector_3d(np.asarray(q, dtype=np.float64), radius)
            if c > 0:
                rm_idx_local.update(_intvector_to_list(idx))

        del pts3d_par, views_par, MB

    if rm_idx_local:
        keep = np.setdiff1d(
            np.arange(len(pcd_local.points)),
            np.fromiter(rm_idx_local, dtype=int)
        )
        pcd_local = pcd_local.select_by_index(keep)

    return pcd_local, len(rm_idx_local)


# =================== MAIN (STREAMED, YAW-ONLY ICP) ===================
def main():
    global BASE_DIR, OUT_PLY

    ap = argparse.ArgumentParser(
        description="Streamed merge with yaw-only ICP (yaw_pairs_refined) and voxel-hash dedup."
    )
    ap.add_argument("--base_dir", type=str, default=BASE_DIR)
    ap.add_argument("--sparse_subfolder", type=str, default=SPARSE_SUBFOLDER)
    ap.add_argument("--ply", type=str, default=PLY_FILENAME)
    ap.add_argument("--out", type=str, default=OUT_PLY)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--batch_range", type=parse_range, default=None)
    ap.add_argument("--model_ext", type=str, default="", choices=["", ".txt", ".bin"])

    # Yaw-only ICP refinement iterations
    ap.add_argument(
        "--icp_iters", type=int, default=3,
        help="Number of refinement iterations for yaw_pairs_refined."
    )

    # Windowing / dedup
    ap.add_argument("--window_kdt", type=int, default=WINDOW_KDT, help="Sliding window size for geometry dedup.")
    ap.add_argument("--window_match", type=int, default=WINDOW_MATCH, help="Number of parent batches used for match-seed dedup.")
    ap.add_argument("--voxel", type=float, default=VOX, help="Voxel size for voxel hash (m).")
    ap.add_argument("--pre_voxel", type=float, default=PRE_DOWNSAMPLE_INCOMING, help="Voxel size for pre-downsampling input clouds (m).")
    ap.add_argument("--geom_radius", type=float, default=GEOM_RADIUS, help="Geometry dedup radius in global space (m).")
    ap.add_argument("--match_radius", type=float, default=MATCH_TO_PLY_RADIUS, help="Match-seed dedup radius in local space (m).")
    ap.add_argument("--k_for_svd", type=int, default=5000, help="Max number of COLMAP correspondences per view for SVD / matching.")
    args = ap.parse_args()

    # 1) Discover and select batches
    all_infos = sorted_batches(args.base_dir, args.sparse_subfolder)
    if len(all_infos) == 0:
        print("No batch folders found.")
        sys.exit(1)

    selected_infos = select_batches(
        all_infos,
        batch_range=args.batch_range,
        max_batches=args.max_batches
    )
    if len(selected_infos) == 0:
        print("\n[select] No batches selected. Check --batch_range and --max_batches.")
        sys.exit(1)

    print("\n[select] Batches to reconstruct (in order):")
    for i, (k, p) in enumerate(selected_infos):
        name = k if k is not None else os.path.basename(p)
        print(f"  ({i:03d}) name={name} path={p}")
    print(f"[select] Total: {len(selected_infos)} batch(es)")

    # 2) Global voxel hash accumulator
    vox = VoxelHash(args.voxel)

    # 3) Sliding window: keep only the most recent window_kdt global chunks for geometry dedup
    window_global_pcds = deque()   # list of recent global chunks (Open3D point clouds)
    window_kdtrees = deque()       # corresponding KD-trees

    # 4) Chain global poses
    G_prev = np.eye(4, dtype=np.float64)

    # 5) Process batches in order
    for idx, (_k, path) in enumerate(selected_infos):
        print(f"\n[batch] {idx} -> {path}")

        # 5.1) Load current local point cloud and pre-downsample
        p_local = o3d.io.read_point_cloud(os.path.join(path, args.ply))
        if args.pre_voxel:
            p_local = p_local.voxel_down_sample(args.pre_voxel)

        # 5.2) Load current COLMAP (for matching) if not the very first batch
        views_cur = pts3d_cur = None
        if idx > 0:
            pts3d_cur, views_cur = load_views_and_points_from_colmap(path, ext=args.model_ext)

        # 5.3) Match-seed-based dedup in local coordinates
        removed_by_match = 0
        if idx > 0 and args.window_match > 0:
            parent_paths = [
                selected_infos[j][1]
                for j in range(max(0, idx - args.window_match), idx)
            ]
            p_local, removed_by_match = remove_by_match_seed_local(
                p_local,
                views_cur,
                pts3d_cur,
                parent_paths,
                args.k_for_svd,
                args.match_radius
            )
        print(f"  [seed] removed by matches (local): {removed_by_match} (r={args.match_radius})")

        # 5.4) Estimate relative pose to (idx-1) or (idx-2) via yaw-only ICP (fixed pairs)
        T_rel = np.eye(4)
        parent_found = False

        if idx > 0:
            for back in (1, 2):
                parent_idx = idx - back
                if parent_idx < 0:
                    continue

                parent_path = selected_infos[parent_idx][1]

                # Build fixed 3D-3D pairs from COLMAP between parent and current
                try:
                    pts3d_par, views_par = load_views_and_points_from_colmap(parent_path, ext=args.model_ext)
                except Exception:
                    continue

                # X_i: parent, X_j: current
                X_i, X_j, _MB = find_matches_from_views(
                    views_par, views_cur, pts3d_par, pts3d_cur, K=args.k_for_svd
                )
                del pts3d_par, views_par

                if len(X_i) < 3:
                    continue

                # SE(3) initial guess from SVD (current -> parent)
                _, R0, t0 = estimate_similarity_transform(X_j, X_i, force_unit_scale=True)
                T_init = np.eye(4)
                T_init[:3, :3] = R0
                T_init[:3, 3] = t0

                # Yaw-only refinement
                T_rel = yaw_pairs_refined(
                    X_src=X_j,
                    X_tgt=X_i,
                    T_init=T_init,
                    max_iter=max(1, args.icp_iters)
                )

                X_j_to_i = (T_rel[:3, :3] @ X_j.T).T + T_rel[:3, 3]
                rmse = float(np.sqrt(np.mean(np.sum((X_j_to_i - X_i) ** 2, axis=1))))
                yaw_deg = math.degrees(math.atan2(T_rel[0, 2], T_rel[0, 0]))

                print(
                    f"  [yaw+T refine] parent={parent_idx} "
                    f"pairs={len(X_i)} yaw={yaw_deg:.3f}° "
                    f"iters={max(1, args.icp_iters)} rmse={rmse:.6f}"
                )

                parent_found = True
                break

        # 5.5) Chain global pose
        if idx == 0 or not parent_found:
            G_cur = np.eye(4)
        else:
            G_cur = G_prev @ T_rel
        G_prev = G_cur

        # 5.6) Transform local points to global (colors do not depend on pose)
        P_local = to_f32_pts(np.asarray(p_local.points))
        C_local = None
        if hasattr(p_local, "colors") and len(p_local.colors) == len(p_local.points):
            C_local = np.asarray(p_local.colors, dtype=np.float32)

        if P_local.size == 0:
            print("  [skip] empty after seed/pre-voxel")
            del p_local, views_cur, pts3d_cur
            continue

        P_glb = (G_cur[:3, :3] @ P_local.T).T + G_cur[:3, 3]
        C_glb = C_local

        # 5.7) Geometry-based dedup against recent global chunks (sliding window)
        if len(window_kdtrees) > 0:
            keep = np.ones(len(P_glb), dtype=bool)
            for kdt in reversed(window_kdtrees):
                if kdt is None:
                    continue
                for i_pt, q in enumerate(P_glb):
                    if not keep[i_pt]:
                        continue
                    c, _, _ = kdt.search_radius_vector_3d(q.astype(np.float64), args.geom_radius)
                    if c > 0:
                        keep[i_pt] = False
            P_glb = P_glb[keep]
            if C_glb is not None:
                C_glb = C_glb[keep]

        print(f"  [geom] kept after window dedup: {len(P_glb)} (r={args.geom_radius})")

        # 5.8) Accumulate into voxel hash
        vox.add_points(P_glb, C_glb)

        # 5.9) Push this global chunk into sliding window
        pcd_win = o3d.geometry.PointCloud()
        pcd_win.points = o3d.utility.Vector3dVector(P_glb.astype(np.float64))
        if C_glb is not None:
            pcd_win.colors = o3d.utility.Vector3dVector(
                np.clip(C_glb, 0.0, 1.0).astype(np.float64)
            )

        window_global_pcds.append(pcd_win)
        window_kdtrees.append(build_kdt(pcd_win))

        if len(window_global_pcds) > args.window_kdt:
            window_global_pcds.popleft()
            window_kdtrees.popleft()

        # 5.10) Release temporary objects
        del p_local, P_local, P_glb, pcd_win, views_cur, pts3d_cur

    # 6) Export merged voxelized point cloud
    merged = vox.to_pointcloud()
    o3d.io.write_point_cloud(args.out, merged)
    print(f"\nSaved: {args.out}  (#vox={len(vox.map)})")


if __name__ == "__main__":
    main()
