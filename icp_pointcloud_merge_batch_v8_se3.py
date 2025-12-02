#!/usr/bin/env python3
# Streamed hybrid merge with translation-only ICP options + windowed KD dedup + voxel-hash (coords + colors), float32
import os, sys, math, argparse
from collections import defaultdict, deque
import numpy as np
import open3d as o3d

# =================== DEFAULT CONFIG ===================
BASE_DIR = "/home/ruoyao/Documents/Research_ABE/vggt/1110_M4"
SPARSE_SUBFOLDER = "sparse"
PLY_FILENAME = "points.ply"

# Registration / ICP
MAX_CORR_DIST = 0.05
K_FOR_SVD     = 5000

# Culling radii
MATCH_TO_PLY_RADIUS = 0.0008
GEOM_RADIUS         = 0.0015

# Optional speed/cleanup
PRE_DOWNSAMPLE_INCOMING = 0.003

# Streamed merge knobs
WINDOW_KDT     = 4      # 用于几何去重的滑窗大小（最近多少批的全局点云建 KDTree）
WINDOW_MATCH   = 2      # “匹配种子去重”只看最近多少个父批次（0 关闭）
VOX            = 0.002  # 体素哈希大小（m）
DTYPE          = np.float32

# Save
OUT_PLY = "./1110_M4/merged_all.ply"

# =================== COLMAP I/O (txt/bin) ===================
try:
    from read_write_model import read_model
except Exception:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    from read_write_model import read_model


def load_views_and_points_from_colmap(model_dir, ext=""):
    cameras, images, points3D = read_model(model_dir, ext=ext)
    image_id_to_name, image_points = {}, {}
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
            if img_name is None: continue
            if 0 <= pt_idx < len(image_points[img_name]):
                xy = image_points[img_name][pt_idx]
                view_index[(img_name, xy)].append(p3d.id)
    return points, view_index


# =================== UTILITIES ===================
def sorted_batches(base_dir, sparse_subfolder):
    sparse_root = os.path.join(base_dir, sparse_subfolder)
    if not os.path.isdir(sparse_root):
        raise FileNotFoundError(f"sparse folder not found: {sparse_root}")
    folders = [os.path.join(sparse_root, f)
               for f in os.listdir(sparse_root)
               if os.path.isdir(os.path.join(sparse_root, f))]
    def to_int_or_none(p):
        try: return int(os.path.basename(p))
        except Exception: return None
    infos = [(to_int_or_none(p), p) for p in folders]
    if any(k is not None for k,_ in infos):
        infos_num = sorted([x for x in infos if x[0] is not None], key=lambda t: t[0])
        infos_str = sorted([x for x in infos if x[0] is None], key=lambda t: os.path.basename(t[1]))
        infos = infos_num + infos_str
    else:
        infos = sorted(infos, key=lambda t: os.path.basename(t[1]))
    return infos


def parse_range(arg):
    try:
        start_str, end_str = arg.split(':')
        return int(start_str), int(end_str)
    except Exception:
        raise argparse.ArgumentTypeError("batch_range must be START:END, e.g., 15:75")


def select_batches(all_infos, batch_range=None, max_batches=None):
    selected = []
    if batch_range is None:
        selected = list(all_infos)
    else:
        start, end = batch_range
        if start > end: start, end = end, start
        for k, p in all_infos:
            if k is None: continue
            if start <= k <= end:
                selected.append((k, p))
    if max_batches is not None and max_batches > 0:
        selected = selected[:max_batches]
    return selected


def find_matches_from_views(view_index_1, view_index_2, points1, points2, K=50):
    pairs, matched_B = [], []
    for view in view_index_2:
        if view in view_index_1:
            ids1 = view_index_1[view][:K]
            ids2 = view_index_2[view][:K]
            for id1 in ids1:
                for id2 in ids2:
                    pairs.append((points1[id1], points2[id2]))
            matched_B.extend([points2[idx] for idx in view_index_2[view]])
    if not pairs:
        X1 = np.zeros((0,3)); X2 = np.zeros((0,3))
    else:
        X1, X2 = map(np.array, zip(*pairs))
    MB = np.unique(np.asarray(matched_B), axis=0) if matched_B else np.zeros((0,3))
    return np.array(X1), np.array(X2), MB


def estimate_similarity_transform(A, B, force_unit_scale=True):
    """Umeyama; when force_unit_scale=True => SE(3) (s=1)."""
    assert A.shape == B.shape and A.shape[1] == 3
    muA, muB = A.mean(0), B.mean(0)
    X, Y = A - muA, B - muB
    Sigma = (Y.T @ X) / max(1, len(A))
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[-1, -1] = -1
    R = U @ S @ Vt
    varA = (X**2).sum() / max(1, len(A))
    s = 1.0 if force_unit_scale else (np.trace(np.diag(D) @ S) / (varA + 1e-12))
    t = muB - s * (R @ muA)
    return float(s), R, t


def _intvector_to_list(idx):
    """Open3D CPU/CUDA 的 IntVector 统一转 python list。"""
    try:
        return list(idx)
    except TypeError:
        return np.asarray(idx, dtype=np.int64).tolist()


def xyz_to_indices_in_pcd(pcd, xyz, radius):
    """在 pcd 中查找落在 xyz 每个点半径内的索引集合（CPU/CUDA 兼容）。"""
    if xyz.size == 0 or len(pcd.points) == 0: return np.array([], dtype=int)
    kdt = o3d.geometry.KDTreeFlann(pcd)
    hit = set()
    for q in xyz:
        if not np.all(np.isfinite(q)): continue
        c, idx, _ = kdt.search_radius_vector_3d(q.astype(np.float64), radius)
        if c > 0:
            hit.update(_intvector_to_list(idx))
    return np.fromiter(hit, dtype=int)


# ---------- fixed-correspondence solver (matched_pairs, SE3 only) ----------
def solve_from_fixed_correspondences(X_src, X_tgt, T_init=None):
    """
    固定一一对应 (X_src[k] ↔ X_tgt[k]) -> SE(3) T: src->tgt.
    如果给了 T_init，先把 X_src 用 T_init(SE3) 变换，再估计 ΔT，返回 T = ΔT @ T_init。
    """
    X_src = np.asarray(X_src, dtype=np.float64)
    X_tgt = np.asarray(X_tgt, dtype=np.float64)
    if len(X_src) < 3 or len(X_src) != len(X_tgt):
        return np.eye(4)

    if T_init is not None:
        sR0 = T_init[:3,:3]
        U, _, Vt = np.linalg.svd(sR0)
        R0 = U @ Vt
        if np.linalg.det(R0) < 0: R0[:, -1] *= -1
        t0 = T_init[:3,3]
        X_src0 = (R0 @ X_src.T).T + t0
    else:
        X_src0 = X_src

    pcd_src0 = o3d.geometry.PointCloud()
    pcd_tgt  = o3d.geometry.PointCloud()
    pcd_src0.points = o3d.utility.Vector3dVector(X_src0)
    pcd_tgt.points  = o3d.utility.Vector3dVector(X_tgt)

    N = len(X_src0)
    corr = o3d.utility.Vector2iVector(np.stack([np.arange(N), np.arange(N)], axis=1).astype(np.int32))

    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
    T_delta = est.compute_transformation(pcd_src0, pcd_tgt, corr)

    if T_init is None:
        return T_delta
    else:
        return T_delta @ T_init


# ---------- NEW: translation-only solvers ----------
def translation_only_from_pairs(X_src, X_tgt):
    """
    Given fixed 1-1 pairs (X_src[k] <-> X_tgt[k]), solve:
        min_t sum || (X_src + t) - X_tgt ||^2
    Closed-form: t = mean(X_tgt - X_src).
    Returns 4x4 SE3 with R=I, t solved.
    """
    X_src = np.asarray(X_src, dtype=np.float64)
    X_tgt = np.asarray(X_tgt, dtype=np.float64)
    if len(X_src) == 0 or len(X_src) != len(X_tgt):
        T = np.eye(4); return T
    t = (X_tgt - X_src).mean(axis=0)
    T = np.eye(4)
    T[:3, 3] = t
    return T


def _ensure_normals(pcd, radius=0.02, max_nn=30):
    """
    Ensure normals exist on the given Open3D point cloud (in-place).
    """
    if not hasattr(pcd, "normals") or len(pcd.normals) != len(pcd.points):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        pcd.normalize_normals()
    return pcd


def translation_only_icp_geometry(
    p_src, p_tgt, max_corr=0.05, max_iters=20, huber_delta=0.01, lambda_reg=1e-6,
    symmetric=True, target_normal_radius=None
):
    """
    Translation-only ICP with point-to-plane residuals:
        r_i(t) = n_i^T( (p_i + t) - q_i )
    Solves (A + λI)t = b at each iteration. Optionally symmetric.
    Returns 4x4 SE3 with R=I and the estimated t.
    """
    if len(p_src.points) == 0 or len(p_tgt.points) == 0:
        return np.eye(4)

    p_src = o3d.geometry.PointCloud(p_src)  # shallow copy
    p_tgt = o3d.geometry.PointCloud(p_tgt)

    rad = target_normal_radius if target_normal_radius is not None else max_corr * 2.5
    _ensure_normals(p_tgt, radius=rad)
    if symmetric:
        _ensure_normals(p_src, radius=rad)

    kdt_t = o3d.geometry.KDTreeFlann(p_tgt)
    kdt_s = o3d.geometry.KDTreeFlann(p_src) if symmetric else None

    P = np.asarray(p_src.points, dtype=np.float64)
    Q = np.asarray(p_tgt.points, dtype=np.float64)
    Nt = np.asarray(p_tgt.normals, dtype=np.float64)
    Ns = np.asarray(p_src.normals, dtype=np.float64) if symmetric else None

    t = np.zeros(3, dtype=np.float64)

    def _huber_weight(res, delta):
        a = np.abs(res)
        w = np.ones_like(a)
        mask = a > delta
        w[mask] = (delta / (a[mask] + 1e-12))
        return w

    for _ in range(max_iters):
        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        num_corr = 0

        # src -> tgt
        for i, p in enumerate(P):
            p_t = p + t
            c, idx, _ = kdt_t.search_radius_vector_3d(p_t, max_corr)
            if c == 0:
                continue
            j = int(idx[0])
            q = Q[j]; n = Nt[j]
            r = float(np.dot(n, (p_t - q)))
            w = _huber_weight(r, huber_delta)
            nnT = np.outer(n, n) * w
            A += nnT
            b += nnT @ (q - p)
            num_corr += 1

        # symmetric tgt -> src
        if symmetric and kdt_s is not None and Ns is not None:
            for j, q in enumerate(Q):
                q_minus_t = q - t
                c, idx, _ = kdt_s.search_radius_vector_3d(q_minus_t, max_corr)
                if c == 0:
                    continue
                i = int(idx[0])
                p = P[i]; n = Ns[i]
                r = float(np.dot(n, (p + t - q)))
                w = _huber_weight(r, huber_delta)
                nnT = np.outer(n, n) * w
                A += nnT
                b += nnT @ (q - p)
                num_corr += 1

        if num_corr == 0:
            break

        A_reg = A + (lambda_reg * np.eye(3))
        try:
            dt = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(dt)):
            break

        t += dt
        if np.linalg.norm(dt) < 1e-6:
            break

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = t
    return T


# =================== NEW: yaw-only (R around Y) solvers ===================  # <<< NEW
def yaw_only_from_pairs(X_src, X_tgt):  # <<< NEW
    """
    固定对应，解“仅绕 Y 轴的旋转 + 全向平移”的闭式最小二乘：
        argmin_{θ, t} sum || R_y(θ) X_src[k] + t - X_tgt[k] ||^2
    返回 4x4 SE(3)（R=R_y(θ*), t=t*）。
    """
    X_src = np.asarray(X_src, dtype=np.float64)
    X_tgt = np.asarray(X_tgt, dtype=np.float64)
    N = len(X_src)
    if N < 2 or N != len(X_tgt):
        return np.eye(4)

    muX = X_src.mean(axis=0)
    muY = X_tgt.mean(axis=0)
    Xc = X_src - muX
    Yc = X_tgt - muY

    H = Yc.T @ Xc
    num = H[0,2] - H[2,0]
    den = H[0,0] + H[2,2]
    theta = math.atan2(num, den if abs(den) > 1e-12 else 1e-12)

    c = math.cos(theta); s = math.sin(theta)
    R = np.array([[ c, 0.0,  s],
                  [0.0, 1.0, 0.0],
                  [-s, 0.0,  c]], dtype=np.float64)
    t = muY - R @ muX

    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3,  3] = t
    return T


def yaw_pairs_refined(X_src, X_tgt, T_init=None, max_iter=3):  # <<< NEW
    """
    matched_pairs 风格：先得初值，再在固定对应下只优化“yaw+平移”若干次。
    """
    X_src = np.asarray(X_src, np.float64)
    X_tgt = np.asarray(X_tgt, np.float64)
    if len(X_src) < 3 or len(X_src) != len(X_tgt):
        return np.eye(4)

    # 初值：如果给了 T_init，用它；否则先求全 SE3 初值，再投影成 yaw-only
    if T_init is None:
        _, R0, t0 = estimate_similarity_transform(X_src, X_tgt, force_unit_scale=True)
    else:
        R0 = T_init[:3,:3]; t0 = T_init[:3,3]

    yaw = math.atan2(R0[0,2], R0[0,0])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c,0,s],[0,1,0],[-s,0,c]], np.float64)
    t = t0.copy()

    muX = X_src.mean(0)
    muY = X_tgt.mean(0)

    for _ in range(max_iter):
        # 闭式重估 yaw（固定对应）
        Xc = X_src - muX
        Yc = X_tgt - muY
        H = Yc.T @ Xc
        yaw = math.atan2(H[0,2] - H[2,0], H[0,0] + H[2,2] if abs(H[0,0]+H[2,2])>1e-12 else 1e-12)
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c,0,s],[0,1,0],[-s,0,c]], np.float64)

        # 闭式更新平移
        t = muY - R @ muX

    T = np.eye(4, dtype=np.float64)
    T[:3,:3], T[:3,3] = R, t
    return T
# =================== END yaw-only solvers ===================  # <<< NEW


# =================== STREAMED MERGE HELPERS ===================
def to_f32_pts(arr):
    arr = np.asarray(arr)
    return arr.astype(DTYPE, copy=False)


class VoxelHash:
    """体素级累加：坐标与颜色一起累加到体素，导出为质心与平均色。"""
    def __init__(self, voxel_size):
        self.vs = float(voxel_size)
        # key -> (sx, sy, sz, cr, cg, cb, n)
        self.map = {}

    def add_points(self, pts, cols=None):
        """
        pts: (N,3) float32/64
        cols: (N,3) in [0,1] 或 None
        """
        vs = self.vs
        mp = self.map
        has_color = cols is not None and len(cols) == len(pts)
        if has_color:
            cols = np.asarray(cols, dtype=np.float32, order="C")
        for i, p in enumerate(pts):
            if not np.all(np.isfinite(p)):
                continue
            k = (int(math.floor(p[0]/vs)),
                 int(math.floor(p[1]/vs)),
                 int(math.floor(p[2]/vs)))
            if k in mp:
                sx, sy, sz, cr, cg, cb, n = mp[k]
                sx += float(p[0]); sy += float(p[1]); sz += float(p[2]); n += 1
                if has_color:
                    r, g, b = cols[i]
                    cr += float(r); cg += float(g); cb += float(b)
                mp[k] = (sx, sy, sz, cr, cg, cb, n)
            else:
                if has_color:
                    r, g, b = cols[i]
                    mp[k] = (float(p[0]), float(p[1]), float(p[2]),
                             float(r), float(g), float(b), 1)
                else:
                    mp[k] = (float(p[0]), float(p[1]), float(p[2]),
                             0.0, 0.0, 0.0, 1)

    def to_pointcloud(self):
        nvox = len(self.map)
        pts = np.empty((nvox, 3), dtype=np.float32)
        cols = np.empty((nvox, 3), dtype=np.float32)
        i = 0
        any_color = False
        for (sx, sy, sz, cr, cg, cb, n) in self.map.values():
            inv = 1.0 / n
            pts[i]  = (sx*inv, sy*inv, sz*inv)
            if cr != 0.0 or cg != 0.0 or cb != 0.0:
                cols[i] = (cr*inv, cg*inv, cb*inv)
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
    if len(pcd.points) == 0: return None
    return o3d.geometry.KDTreeFlann(pcd)


def remove_by_match_seed_local(pcd_local, views_cur, pts3d_cur, recent_parents, k_for_svd, radius):
    """
    在“当前 batch 的本地点云（未变换）”上，利用与最近父批次的匹配点（MB，父->当前），
    直接剔除当前局部点中落在 MB 半径内的点。只访问最近若干父批次以控内存。
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
        keep = np.setdiff1d(np.arange(len(pcd_local.points)), np.fromiter(rm_idx_local, dtype=int))
        pcd_local = pcd_local.select_by_index(keep)
    return pcd_local, len(rm_idx_local)


# =================== MAIN (STREAMED) ===================
def main():
    global BASE_DIR, OUT_PLY
    ap = argparse.ArgumentParser(description="Streamed hybrid merge with translation-only ICP options")
    ap.add_argument("--base_dir", type=str, default=BASE_DIR)
    ap.add_argument("--sparse_subfolder", type=str, default=SPARSE_SUBFOLDER)
    ap.add_argument("--ply", type=str, default=PLY_FILENAME)
    ap.add_argument("--out", type=str, default=OUT_PLY)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--batch_range", type=parse_range, default=None)
    ap.add_argument("--model_ext", type=str, default="", choices=["", ".txt", ".bin"])
    ap.add_argument(
        "--icp_mode", type=str, default="matched_pairs",
        choices=[
            "geometry", "matched_pairs",
            "trans_only_geometry", "trans_only_pairs",
            "yaw_pairs", "yaw_pairs_refined"   # <<< NEW
        ],
        help=("geometry: full-cloud NN-ICP (SE3); "
              "matched_pairs: fixed correspondences (SE3); "
              "trans_only_geometry: translation-only NN-ICP (point-to-plane); "
              "trans_only_pairs: translation-only from fixed pairs; "
              "yaw_pairs: fixed correspondences with yaw-only rotation + translation (closed-form); "  # <<< NEW
              "yaw_pairs_refined: yaw-only + translation with few refinement iterations.")              # <<< NEW
    )
    ap.add_argument("--icp_iters", type=int, default=20, help="Max iterations for translation-only ICP (geometry mode) / yaw refinement")
    ap.add_argument("--huber_delta", type=float, default=0.01, help="Huber delta for translation-only ICP")
    ap.add_argument("--tikhonov", type=float, default=1e-6, help="Tikhonov λ for translation-only ICP")
    ap.add_argument("--window_kdt", type=int, default=WINDOW_KDT, help="几何去重滑窗大小")
    ap.add_argument("--window_match", type=int, default=WINDOW_MATCH, help="用于匹配种子去重的父批次数")
    ap.add_argument("--voxel", type=float, default=VOX, help="体素哈希大小（m）")
    ap.add_argument("--pre_voxel", type=float, default=PRE_DOWNSAMPLE_INCOMING, help="输入点云预降采样体素尺寸（m）")
    ap.add_argument("--geom_radius", type=float, default=GEOM_RADIUS, help="全局几何去重半径（m）")
    ap.add_argument("--match_radius", type=float, default=MATCH_TO_PLY_RADIUS, help="匹配种子去重半径（m）")
    ap.add_argument("--k_for_svd", type=int, default=K_FOR_SVD)
    ap.add_argument("--max_corr", type=float, default=MAX_CORR_DIST)
    args = ap.parse_args()

    # === 发现并选择批次 ===
    all_infos = sorted_batches(args.base_dir, args.sparse_subfolder)
    if len(all_infos) == 0:
        print("No batch folders found."); sys.exit(1)

    selected_infos = select_batches(all_infos, batch_range=args.batch_range, max_batches=args.max_batches)
    if len(selected_infos) == 0:
        print("\n[select] No batches selected. Check --batch_range and --max_batches."); sys.exit(1)

    print("\n[select] Batches to reconstruct (in order):")
    for i,(k,p) in enumerate(selected_infos):
        print(f"  ({i:03d}) name={k if k is not None else os.path.basename(p)} path={p}")
    print(f"[select] Total: {len(selected_infos)} batch(es)")

    # === 全局体素哈希 ===
    vox = VoxelHash(args.voxel)

    # === 滑窗：仅保存最近 window_kdt 个“已合并到全局后的子块”用于几何去重 ===
    window_global_pcds = deque()   # [pcd_global_i]
    window_kdtrees      = deque()   # [KDTreeFlann]

    # === 链式全局位姿 ===
    G_prev = np.eye(4, dtype=np.float64)

    # === 逐批处理 ===
    for idx, (_k, path) in enumerate(selected_infos):
        print(f"\n[batch] {idx} -> {path}")

        # 1) 读当前本地点云并预降采样（Open3D 会保留/平均颜色）
        p_local = o3d.io.read_point_cloud(os.path.join(path, args.ply))
        if args.pre_voxel:
            p_local = p_local.voxel_down_sample(args.pre_voxel)

        # 2) 读当前 batch 的 COLMAP（仅当需要匹配时）
        views_cur = pts3d_cur = None
        if idx > 0:
            pts3d_cur, views_cur = load_views_and_points_from_colmap(path, ext=args.model_ext)

        # 3) “匹配种子去重”（可选）
        removed_by_match = 0
        if idx > 0 and args.window_match > 0:
            parent_paths = [selected_infos[j][1] for j in range(max(0, idx - args.window_match), idx)]
            p_local, removed_by_match = remove_by_match_seed_local(
                p_local, views_cur, pts3d_cur, parent_paths, args.k_for_svd, args.match_radius
            )
        print(f"  [seed] removed by matches (local): {removed_by_match} (r={args.match_radius})")

        # 4) 估计与 (idx-1) 或 (idx-2) 的相对位姿（仅连近邻，避免图稠密）
        T_rel = np.eye(4)
        parent_found = False
        if idx > 0:
            for back in (1, 2):
                parent_idx = idx - back
                if parent_idx < 0: 
                    continue
                parent_path = selected_infos[parent_idx][1]

                # -- Find fixed pairs from COLMAP (for either mode)
                try:
                    pts3d_par, views_par = load_views_and_points_from_colmap(parent_path, ext=args.model_ext)
                except Exception:
                    continue
                X_i, X_j, _MB = find_matches_from_views(views_par, views_cur, pts3d_par, pts3d_cur, K=args.k_for_svd)
                del pts3d_par, views_par
                if len(X_i) < 3:
                    continue

                # SE(3) initial guess from SVD
                _, R0, t0 = estimate_similarity_transform(X_j, X_i, force_unit_scale=True)
                T_init = np.eye(4); T_init[:3,:3] = R0; T_init[:3,3] = t0

                if args.icp_mode == "geometry":
                    # Full SE3 ICP against the recent global block
                    if len(window_global_pcds) >= back and len(window_global_pcds[-back].points) > 0:
                        target = window_global_pcds[-back]
                        est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
                        reg = o3d.pipelines.registration.registration_icp(
                            p_local, target,
                            max_correspondence_distance=args.max_corr,
                            init=T_init, estimation_method=est
                        )
                        T_rel = reg.transformation
                        print(f"  [icp] parent={parent_idx} pairs={len(X_i)} fitness={reg.fitness:.4f} rmse={reg.inlier_rmse:.6f}")
                    else:
                        T_rel = T_init
                        print(f"  [icp] parent={parent_idx} target empty -> use T_init")

                elif args.icp_mode == "matched_pairs":
                    # SE(3) from fixed correspondences (existing)
                    T_rel = solve_from_fixed_correspondences(X_src=X_j, X_tgt=X_i, T_init=T_init)
                    X_j_to_i = (T_rel[:3,:3] @ X_j.T).T + T_rel[:3,3]
                    rmse = float(np.sqrt(np.mean(np.sum((X_j_to_i - X_i)**2, axis=1))))
                    print(f"  [pair] parent={parent_idx} pairs={len(X_i)} rmse={rmse:.6f}")

                elif args.icp_mode == "trans_only_pairs":
                    # translation-only from fixed pairs
                    T_rel = translation_only_from_pairs(X_src=X_j, X_tgt=X_i)
                    rmse = float(np.sqrt(np.mean(np.sum(((X_j + T_rel[:3,3]) - X_i)**2, axis=1))))
                    print(f"  [t-only pair] parent={parent_idx} pairs={len(X_i)} rmse={rmse:.6f}")

                elif args.icp_mode == "trans_only_geometry":
                    # translation-only ICP against recent global block
                    if len(window_global_pcds) >= back and len(window_global_pcds[-back].points) > 0:
                        target = window_global_pcds[-back]
                        T_rel = translation_only_icp_geometry(
                            p_src=p_local, p_tgt=target,
                            max_corr=args.max_corr,
                            max_iters=args.icp_iters,
                            huber_delta=args.huber_delta,
                            lambda_reg=args.tikhonov,
                            symmetric=True,
                            target_normal_radius=None  # defaults to 2.5*max_corr
                        )
                        # Simple residual report (src->tgt pt2plane)
                        p_src_trans = o3d.geometry.PointCloud(p_local)
                        P_src = np.asarray(p_src_trans.points, dtype=np.float64) + T_rel[:3,3]
                        kdt_t = o3d.geometry.KDTreeFlann(target)
                        _ensure_normals(target, radius=args.max_corr*2.5)
                        Nt = np.asarray(target.normals, dtype=np.float64)
                        Q = np.asarray(target.points, dtype=np.float64)
                        res, used = [], 0
                        for p in P_src:
                            c, idx, _ = kdt_t.search_radius_vector_3d(p, args.max_corr)
                            if c == 0: 
                                continue
                            j = int(idx[0])
                            q = Q[j]
                            n = Nt[j]
                            res.append(float(np.dot(n, (p - q))))
                            used += 1
                        rmse = float(np.sqrt(np.mean(np.square(res)))) if len(res) else float("nan")
                        print(f"  [t-only icp] parent={parent_idx} used={used} rmse_n={rmse:.6f}")
                    else:
                        T_rel = translation_only_from_pairs(X_src=X_j, X_tgt=X_i)
                        print(f"  [t-only icp] parent={parent_idx} target empty -> use pairs-only translation")

                elif args.icp_mode == "yaw_pairs":   # <<< NEW
                    T_rel = yaw_only_from_pairs(X_src=X_j, X_tgt=X_i)
                    X_j_to_i = (T_rel[:3,:3] @ X_j.T).T + T_rel[:3,3]
                    rmse = float(np.sqrt(np.mean(np.sum((X_j_to_i - X_i)**2, axis=1))))
                    yaw_deg = math.degrees(math.atan2(T_rel[0,2], T_rel[0,0]))
                    print(f"  [yaw+T pair] parent={parent_idx} pairs={len(X_i)} yaw={yaw_deg:.3f}° rmse={rmse:.6f}")

                elif args.icp_mode == "yaw_pairs_refined":  # <<< NEW
                    T_rel = yaw_pairs_refined(X_src=X_j, X_tgt=X_i, T_init=T_init, max_iter=max(1, args.icp_iters))
                    X_j_to_i = (T_rel[:3,:3] @ X_j.T).T + T_rel[:3,3]
                    rmse = float(np.sqrt(np.mean(np.sum((X_j_to_i - X_i)**2, axis=1))))
                    yaw_deg = math.degrees(math.atan2(T_rel[0,2], T_rel[0,0]))
                    print(f"  [yaw+T refine] parent={parent_idx} pairs={len(X_i)} yaw={yaw_deg:.3f}° iters={max(1,args.icp_iters)} rmse={rmse:.6f}")

                parent_found = True
                break

        # 5) 链式全局位姿
        if idx == 0 or not parent_found:
            G_cur = np.eye(4)
        else:
            G_cur = G_prev @ T_rel
        G_prev = G_cur

        # 6) 本地点 -> float32 -> 全局（颜色不变换）
        P_local = to_f32_pts(np.asarray(p_local.points))
        C_local = None
        if hasattr(p_local, "colors") and len(p_local.colors) == len(p_local.points):
            C_local = np.asarray(p_local.colors, dtype=np.float32)

        if P_local.size == 0:
            print("  [skip] empty after seed/pre-voxel")
            del p_local, views_cur, pts3d_cur
            continue

        P_glb = (G_cur[:3,:3] @ P_local.T).T + G_cur[:3,3]
        C_glb = C_local  # 颜色无需随位姿变换

        # 7) 滑窗几何去重（只对最近 window_kdt 全局子块 KDTree 半径剔除）
        if len(window_kdtrees) > 0:
            keep = np.ones(len(P_glb), dtype=bool)
            for kdt in reversed(window_kdtrees):
                if kdt is None: continue
                for i_pt, q in enumerate(P_glb):
                    if not keep[i_pt]: continue
                    c, _, _ = kdt.search_radius_vector_3d(q.astype(np.float64), args.geom_radius)
                    if c > 0: keep[i_pt] = False
            P_glb = P_glb[keep]
            if C_glb is not None:
                C_glb = C_glb[keep]
        print(f"  [geom] kept after window dedup: {len(P_glb)} (r={args.geom_radius})")

        # 8) 体素哈希累加（带颜色）
        vox.add_points(P_glb, C_glb)

        # 9) 把“当前这块（几何去重后的）全局点”放入滑窗
        pcd_win = o3d.geometry.PointCloud()
        pcd_win.points = o3d.utility.Vector3dVector(P_glb.astype(np.float64))
        if C_glb is not None:
            pcd_win.colors = o3d.utility.Vector3dVector(np.clip(C_glb, 0.0, 1.0).astype(np.float64))
        window_global_pcds.append(pcd_win)
        window_kdtrees.append(build_kdt(pcd_win))
        if len(window_global_pcds) > args.window_kdt:
            window_global_pcds.popleft()
            window_kdtrees.popleft()

        # 10) 释放临时对象
        del p_local, P_local, P_glb, pcd_win, views_cur, pts3d_cur

    # === 导出 ===
    merged = vox.to_pointcloud()
    o3d.io.write_point_cloud(args.out, merged)
    print(f"\nSaved: {args.out}  (#vox={len(vox.map)})")


if __name__ == "__main__":
    main()
