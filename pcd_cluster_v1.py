import open3d as o3d
import numpy as np
import cv2
import os

# ================== 参数区 ==================

INPUT_PCD = "/home/ruoyao/Documents/Research_ABE/vggt/1110_M4/merged_50.ply"

USE_AUTO_SCALE = False      # 是否根据点云尺度自动估计 eps / 平面阈值

HEIGHT_PERCENTILE = 50      # 裁掉距离平面“更远的那一侧”中最外层的 X% 点
KEEP_HIGHER_SIDE = False    # True: 保留平面上侧点; False: 保留平面下侧点

# 叶片（绿色）HSV 范围
LEAF_H_MIN, LEAF_H_MAX = 30, 95
LEAF_S_MIN, LEAF_V_MIN = 60, 40

# 茎/叶柄（红/黄棕）HSV 范围
STEM_RED_1_MAX = 20
STEM_RED_2_MIN = 160
STEM_YELLOW_MIN, STEM_YELLOW_MAX = 20, 40
STEM_S_MIN, STEM_V_MIN = 40, 40

# DBSCAN 参数
DBSCAN_EPS_MANUAL = 0.03
DBSCAN_MIN_PTS = 80
DBSCAN_EPS_SCALE = 0.02   # 自动尺度时：eps = scene_diag * scale

# RANSAC 平面拟合参数
PLANE_DIST_THRESHOLD_MANUAL = 0.01
PLANE_DIST_THRESHOLD_SCALE = 0.005
PLANE_RANSAC_N = 3
PLANE_ITER = 1000

# ⭐ 保存 cluster 后点云的开关
SAVE_CLUSTERED_PCD = True                # 保存所有 cluster 点云到一个文件
CLUSTERED_OUTPUT_PATH = "clustered_output.ply"

SAVE_EACH_CLUSTER_SEPARATELY = False      # 每个 cluster 单独保存成一个文件
EACH_CLUSTER_DIR = "clusters_out"        # 单独 cluster 存放目录

# =====================================================

print("加载点云:", INPUT_PCD)
pcd = o3d.io.read_point_cloud(INPUT_PCD)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
N = points.shape[0]
print("点数:", N)

# ---------- 自动按场景尺度设定 eps & 平面距离阈值 ----------
if USE_AUTO_SCALE:
    bbox = points.max(axis=0) - points.min(axis=0)
    scene_diag = np.linalg.norm(bbox)
    print("场景对角线长度:", scene_diag)

    DBSCAN_EPS = scene_diag * DBSCAN_EPS_SCALE
    PLANE_DIST_THRESHOLD = scene_diag * PLANE_DIST_THRESHOLD_SCALE
else:
    DBSCAN_EPS = DBSCAN_EPS_MANUAL
    PLANE_DIST_THRESHOLD = PLANE_DIST_THRESHOLD_MANUAL

print(f"DBSCAN_EPS = {DBSCAN_EPS:.5f}, PLANE_DIST_THRESHOLD = {PLANE_DIST_THRESHOLD:.5f}")

# ---------- 1. 拟合主平面 ----------
print("RANSAC 拟合主平面（地面/地膜）...")
plane_model, plane_inliers = pcd.segment_plane(
    distance_threshold=PLANE_DIST_THRESHOLD,
    ransac_n=PLANE_RANSAC_N,
    num_iterations=PLANE_ITER
)
a, b, c, d = plane_model
print("平面方程: %.4fx + %.4fy + %.4fz + %.4f = 0" % (a, b, c, d))
print("平面内点数量:", len(plane_inliers))

# 点到平面的有符号距离
dist = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d

# ---------- 2. 选择保留哪一侧 ----------
if KEEP_HIGHER_SIDE:
    # 保留平面“上侧”点：dist 大的那部分
    thresh = np.percentile(dist, HEIGHT_PERCENTILE)
    keep_mask = dist >= thresh
    print(f"保留平面上侧: dist >= {thresh:.5f}")
else:
    # 保留平面“下侧”点：dist 小的那部分
    thresh = np.percentile(dist, 100.0 - HEIGHT_PERCENTILE)
    keep_mask = dist <= thresh
    print(f"保留平面下侧: dist <= {thresh:.5f}")

keep_idx = np.where(keep_mask)[0]
print(f"裁剪高度后剩余点数: {keep_idx.size} / {N}")

pcd_high = pcd.select_by_index(keep_idx)
points_high = np.asarray(pcd_high.points)
colors_high = np.asarray(pcd_high.colors)
N_high = points_high.shape[0]

# ---------- 3. 在“保留的点云”上做颜色 -> HSV ----------
rgb_255_high = (colors_high * 255).astype(np.uint8)
hsv_high = cv2.cvtColor(rgb_255_high.reshape(-1, 1, 3),
                        cv2.COLOR_RGB2HSV).reshape(-1, 3)
Hh, Sh, Vh = hsv_high[:, 0], hsv_high[:, 1], hsv_high[:, 2]

# 叶片掩码
leaf_mask_high = (
    (Hh > LEAF_H_MIN) & (Hh < LEAF_H_MAX) &
    (Sh > LEAF_S_MIN) & (Vh > LEAF_V_MIN)
)

# 茎掩码（红/黄棕）
stem_red_high = ((Hh < STEM_RED_1_MAX) | (Hh > STEM_RED_2_MIN))
stem_yellow_high = ((Hh > STEM_YELLOW_MIN) & (Hh < STEM_YELLOW_MAX))
stem_mask_high = (stem_red_high | stem_yellow_high) & \
                 (Sh > STEM_S_MIN) & (Vh > STEM_V_MIN)

plant_mask_high = leaf_mask_high | stem_mask_high
plant_idx_high = np.where(plant_mask_high)[0]

print("保留点云中，植株颜色点数量（叶片+茎）:", plant_idx_high.size)
if plant_idx_high.size == 0:
    print("没有检测到植株点，检查颜色阈值或 HEIGHT_PERCENTILE / KEEP_HIGHER_SIDE。")
    raise SystemExit

plant_pcd_high = pcd_high.select_by_index(plant_idx_high)

# ---------- 4. 对“植株点”做 DBSCAN 聚类 ----------
print("对植株点做 DBSCAN 聚类（按株）...")
labels_high = np.array(
    plant_pcd_high.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_PTS,
        print_progress=True
    )
)

if labels_high.max() < 0:
    print("DBSCAN 没聚出任何簇，检查 EPS / MIN_PTS 参数。")
    raise SystemExit

cluster_ids = np.unique(labels_high[labels_high >= 0])
print("检测到植株簇数量:", cluster_ids.size)

# ---------- 5. 把聚类结果映射回“保留点云”的索引 ----------
cluster_assign_high = -1 * np.ones(N_high, dtype=int)
for local_idx, cid in enumerate(labels_high):
    if cid < 0:
        continue
    global_idx = plant_idx_high[local_idx]
    cluster_assign_high[global_idx] = cid

# ================================
# ⭐ 保存 cluster 点云（新增功能）
# ================================

plant_cluster_mask_high = cluster_assign_high >= 0
cluster_points = points_high[plant_cluster_mask_high]
cluster_colors = colors_high[plant_cluster_mask_high]

# 保存所有 cluster 点云到一个文件
if SAVE_CLUSTERED_PCD:
    pc_all_clusters = o3d.geometry.PointCloud()
    pc_all_clusters.points = o3d.utility.Vector3dVector(cluster_points)
    pc_all_clusters.colors = o3d.utility.Vector3dVector(cluster_colors)
    o3d.io.write_point_cloud(CLUSTERED_OUTPUT_PATH, pc_all_clusters)
    print(f"已保存所有 cluster 点云到：{CLUSTERED_OUTPUT_PATH}")

# 每个 cluster 单独保存
if SAVE_EACH_CLUSTER_SEPARATELY:
    os.makedirs(EACH_CLUSTER_DIR, exist_ok=True)
    unique_clusters = np.unique(cluster_assign_high[cluster_assign_high >= 0])
    print("独立保存每个 cluster 到文件夹:", EACH_CLUSTER_DIR)

    for cid in unique_clusters:
        mask = (cluster_assign_high == cid)
        pts = points_high[mask]
        cols = colors_high[mask]

        pcd_c = o3d.geometry.PointCloud()
        pcd_c.points = o3d.utility.Vector3dVector(pts)
        pcd_c.colors = o3d.utility.Vector3dVector(cols)

        out_path = os.path.join(EACH_CLUSTER_DIR, f"cluster_{cid}.ply")
        o3d.io.write_point_cloud(out_path, pcd_c)
        print("保存:", out_path)

# ---------- 6. 可视化 1：裁剪后点云（背景灰 + 植株簇彩色） ----------
vis_colors_high = np.ones_like(colors_high) * 0.6  # 默认灰色背景
rng = np.random.default_rng(42)
palette = rng.random((cluster_ids.size, 3))

for k_c, cid in enumerate(cluster_ids):
    mask = (cluster_assign_high == cid)
    vis_colors_high[mask] = palette[k_c]

vis_pcd_all = o3d.geometry.PointCloud()
vis_pcd_all.points = o3d.utility.Vector3dVector(points_high)
vis_pcd_all.colors = o3d.utility.Vector3dVector(vis_colors_high)

print("窗口 1：裁剪后点云（灰底 + 植株簇彩色）")
o3d.visualization.draw_geometries(
    [vis_pcd_all],
    window_name="Cropped points (plant clusters colored)",
    width=1280,
    height=720
)

# ---------- 7. 可视化 2：仅植株簇（原始颜色） ----------
vis_pcd_clusters_rgb = o3d.geometry.PointCloud()
vis_pcd_clusters_rgb.points = o3d.utility.Vector3dVector(cluster_points)
vis_pcd_clusters_rgb.colors = o3d.utility.Vector3dVector(cluster_colors)

print("窗口 2：仅植株簇（原始颜色）")
o3d.visualization.draw_geometries(
    [vis_pcd_clusters_rgb],
    window_name="Plant clusters only (original color, after height cropping)",
    width=1280,
    height=720
)

print("完成。")
