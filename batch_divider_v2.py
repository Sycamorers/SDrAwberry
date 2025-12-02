#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGGT Batch Wrapper using Single-Scene Script (multi-camera tree + renaming + summary)
====================================================================================

本版适配的数据结构：
    scene_dir/
      images/
        <cam_serial_1>/
          rgb/
            rgb_<frame_id>.png
        <cam_serial_2>/
          rgb/
            rgb_<frame_id>.png
        ...

主要行为：
- 将 images/ 下的多相机帧，按 frame_id 聚合；每帧的图片按 (cam_serial, 文件名) 稳定排序。
- 按 batch_size / overlap_frames 生成批次，将每个批次作为“迷你场景”放到 scene_dir/batches/<NN>/images/。
- 在拷贝/链接到批次目录时，将文件重命名为：<frame_id_str>_<cam_serial>.png（保留 frame_id 前导零）。
- 跑单场景脚本（默认 demo_colmap_perc.py）并将其输出 sparse/ 收集到 scene_dir/sparse/<NN>/。
- 输出 summary 文件：batch_summary.json 与 batch_summary.txt，包含批次/帧分布、相机集合、缺失告警等。

命名与兼容：
- 批次目录号采用 2 位（01..99），>=100 自然扩展为 3 位（100, 101, ...）。
- 如存在 legacy 的 3 位批次目录（001）且新标准目录（01）不存在，会自动重命名为 2 位以消除重复。
- 若同一编号同时存在 2 位与 3 位目录，会报错让用户手动处理。

假设：
- 源文件名为 rgb_<frame_id>.png，其中 <frame_id> 为十进制整数字符串（可能有前导零，如 000123）。
- 每台相机的帧集合可能不完全一致；脚本会基于检测到的相机数给出“每帧期望张数”的告警，但不强制终止。

CLI:
    --scene_dir        必填，包含 images/ 的场景目录
    --batch_size       每批帧数（默认 10）
    --overlap_frames   批次间重叠帧数（默认 0）
    --demo_script      单场景脚本（默认 demo_colmap_perc.py）
    --symlink          使用软链接代替复制
    --only_batch       只运行该（1-based）批次
    --max_batches      只处理前 N 个批次（与 only_batch 互斥；仅在未指定 only_batch 时生效）
    --demo_args        透传给单场景脚本的附加参数字符串
    --cams_per_frame  （可选）强制期望的每帧相机数；不指定则按目录自动推断

"""

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any


# ---------------------------
# Helpers for new layout
# ---------------------------
def _parse_frame_id_from_name(name: str) -> Tuple[int, str]:
    """
    从文件名解析 frame_id：
      期望 name 形如 'rgb_<frame_id>.png' 或 stem: 'rgb_<frame_id>'
    返回 (frame_id_int, frame_id_str)
    """
    if not name.startswith("rgb_"):
        raise ValueError("not a rgb_<id> name")
    fid_str = name.split("rgb_", 1)[1]
    if "." in fid_str:
        fid_str = fid_str.split(".", 1)[0]
    fid_int = int(fid_str)
    return fid_int, fid_str


def parse_frame_cam_from_path(p: Path) -> Tuple[int, str, str]:
    """
    期望路径: images/<cam_serial>/rgb/rgb_<frame_id>.png
    返回 (frame_id_int, cam_serial, frame_id_str)
    - frame_id_int 用于数值排序
    - frame_id_str 保留原始前导零（用于目标重命名）
    """
    fid_int, fid_str = _parse_frame_id_from_name(p.name)
    cam_serial = p.parent.parent.name
    return fid_int, cam_serial, fid_str


def detect_cameras(images_dir: Path) -> List[str]:
    """
    返回有效相机目录列表：images/<cam_serial>/rgb 存在即视为一个相机
    """
    cams = []
    for cam_dir in sorted(images_dir.iterdir()):
        if cam_dir.is_dir() and (cam_dir / "rgb").is_dir():
            cams.append(cam_dir.name)
    return cams


def group_by_frame(images_dir: Path) -> Dict[int, List[Path]]:
    """
    扫描 images/<cam_serial>/rgb/rgb_<frame_id>.png
    - frame_id 从 'rgb_<frame_id>' 提取
    - grouped[frame_id] = [Path, Path, ...]  (同一帧的多机位图片)
    - 帧内排序使用 (cam_serial, 文件名) 保持稳定
    """
    grouped = defaultdict(list)
    # 只匹配 .../<cam_serial>/rgb/*.png 三级结构
    for f in sorted(images_dir.glob("*/*/*.png")):
        if f.parent.name != "rgb" or f.parent.parent == images_dir:
            continue
        try:
            fid_int, _ = _parse_frame_id_from_name(f.name)
        except Exception:
            continue
        grouped[fid_int].append(f)

    # 统一对每帧内的文件做稳定排序：先 cam_serial，再文件名
    for fid, paths in grouped.items():
        grouped[fid] = sorted(paths, key=lambda p: (p.parent.parent.name, p.name))
    return dict(sorted(grouped.items()))


# ---------------------------
# Batching logic
# ---------------------------
def build_batches(frames_sorted: List[int], grouped: Dict[int, List[Path]],
                  batch_size: int, overlap: int) -> List[List[Path]]:
    step = batch_size - overlap
    if step <= 0:
        raise ValueError("batch_size must be greater than overlap_frames")
    batches: List[List[Path]] = []
    for s in range(0, len(frames_sorted), step):
        batch_frames = frames_sorted[s:s + batch_size]
        items: List[Path] = []
        for fid in batch_frames:
            # 稳定顺序：按 cam_serial，再文件名
            items.extend(sorted(grouped[fid], key=lambda p: (p.parent.parent.name, p.name)))
        batches.append(items)
    return batches


# ---------------------------
# Summary writing
# ---------------------------
def write_summary(scene_dir: Path,
                  cam_serials: List[str],
                  expected_per_frame: int,
                  frames_sorted: List[int],
                  grouped: Dict[int, List[Path]],
                  batches_root: Path,
                  batch_records: List[Dict[str, Any]],
                  issues: Dict[str, Any]) -> None:
    """
    写出 JSON 与 TXT 两份 summary：
    - JSON: 完整结构化数据，便于程序消费
    - TXT : 人类可读版摘要
    """
    now = datetime.now().isoformat(timespec="seconds")
    summary_json = {
        "generated_at": now,
        "scene_dir": str(scene_dir),
        "images_root": str(scene_dir / "images"),
        "batches_root": str(batches_root),
        "cameras": cam_serials,
        "expected_images_per_frame": expected_per_frame,
        "total_frames_detected": len(frames_sorted),
        "frames_range": [frames_sorted[0], frames_sorted[-1]] if frames_sorted else None,
        "batch_count": len(batch_records),
        "batches": batch_records,
        "issues": issues,  # 包含缺图帧等
    }

    # JSON
    json_path = scene_dir / "batch_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    # TXT（简要）
    txt_path = scene_dir / "batch_summary.txt"
    lines = []
    lines.append("VGGT Batch Summary")
    lines.append("=" * 72)
    lines.append(f"Generated at    : {now}")
    lines.append(f"Scene dir       : {scene_dir}")
    lines.append(f"Images root     : {scene_dir / 'images'}")
    lines.append(f"Batches root    : {batches_root}")
    lines.append(f"Cameras ({len(cam_serials)}): {', '.join(cam_serials) if cam_serials else '(none)'}")
    lines.append(f"Expected/frame  : {expected_per_frame}")
    lines.append(f"Total frames    : {len(frames_sorted)}")
    if frames_sorted:
        lines.append(f"Frames range    : {frames_sorted[0]} .. {frames_sorted[-1]}")
    lines.append("")
    lines.append(f"Total batches   : {len(batch_records)}")
    for b in batch_records:
        lines.append(f"- Batch {b['name']}: {b['image_count']} images, frames {b['frame_first']}..{b['frame_last']} "
                     f"(frames={len(b['frames'])}, cams/order={','.join(cam_serials) if cam_serials else '-'})")
    lines.append("")
    if issues.get("bad_frames"):
        lines.append(f"Issues: {len(issues['bad_frames'])} frame(s) not matching expected {expected_per_frame} images:")
        # 仅打印最多前 20 个
        for i, (fid, cnt) in enumerate(sorted(issues["bad_frames"].items())[:20]):
            lines.append(f"  - frame {fid}: {cnt} image(s)")
    else:
        lines.append("Issues: None")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSummary written:\n  - {json_path}\n  - {txt_path}")


# ---------------------------
# Split into per-batch scenes (with renaming)
# ---------------------------
def split_into_scenes(scene_dir: Path, batch_size: int, overlap: int, symlink: bool,
                      max_batches: int | None,
                      cams_per_frame: int | None):
    """
    Create scene_dir/batches/<NN>/images/ (2-digit naming for <100).
    将文件重命名为 <frame_id_str>_<cam_serial>.png 放入对应批次的 images/。
    返回：
      (batches_root, total_batches_created, PAD_WIDTH=2,
       frames_sorted, grouped, cam_serials, expected_per_frame, batch_records, issues)
    """
    images_dir = scene_dir / "images"
    if not images_dir.exists():
        raise RuntimeError(f"images/ folder not found: {images_dir}")

    grouped = group_by_frame(images_dir)
    if not grouped:
        raise RuntimeError("No frames matched under images/<cam_serial>/rgb/rgb_<id>.png")

    # 相机集合 & 每帧期望张数
    cam_serials = detect_cameras(images_dir)
    inferred_expected = len(cam_serials)
    if inferred_expected == 0:
        raise RuntimeError("No camera folders found under images/ (expect images/<cam_serial>/rgb/)")
    expected_per_frame = cams_per_frame if cams_per_frame is not None else inferred_expected

    # 缺失告警统计
    bad_frames = {fid: len(v) for fid, v in grouped.items() if len(v) != expected_per_frame}
    if bad_frames:
        print(f"⚠️  {len(bad_frames)} frame(s) do not have exactly {expected_per_frame} images "
              f"(detected cameras={len(cam_serials)}). Showing up to 5:")
        for i, fid in enumerate(sorted(bad_frames.keys())):
            if i >= 5:
                break
            print(f"   frame {fid}: {bad_frames[fid]} file(s)")
    issues = {"bad_frames": bad_frames}

    frames_sorted = list(grouped.keys())
    batches_all = build_batches(frames_sorted, grouped, batch_size, overlap)

    if max_batches is not None:
        if max_batches < 1:
            raise ValueError("--max_batches must be >= 1")
        batches = batches_all[:max_batches]
    else:
        batches = batches_all

    total = len(batches)

    PAD_WIDTH = 2  # always 2 digits for <100; 100+ will naturally be 3+ with :02d formatting
    batches_root = scene_dir / "batches"
    batches_root.mkdir(exist_ok=True)

    # Guard against legacy 3-digit duplicates (001 vs 01)
    legacy_conflicts = []
    for i in range(1, total + 1):
        name_2 = f"{i:0{PAD_WIDTH}d}"
        name_3 = f"{i:03d}"
        if (batches_root / name_2).exists() and (batches_root / name_3).exists():
            legacy_conflicts.append((name_2, name_3))
    if legacy_conflicts:
        msg = "\n".join([f"  {a}  vs  {b}" for a, b in legacy_conflicts[:10]])
        raise RuntimeError(
            "Duplicate batch folders detected (both 2-digit and 3-digit exist):\n"
            f"{msg}\nPlease delete the legacy 3-digit ones (e.g., '001') or the 2-digit ones, "
            "then re-run. The wrapper standardizes on 2-digit for <100."
        )

    # 批次元数据记录（写 summary 用）
    batch_records: List[Dict[str, Any]] = []

    for i, batch in enumerate(batches, 1):
        bn = f"{i:0{PAD_WIDTH}d}"   # '01'..'99', then '100', '101', ...
        # 如果存在 legacy 3 位且 2 位不存在，先重命名以避免重复
        legacy = batches_root / f"{i:03d}"
        target = batches_root / bn
        if legacy.exists() and not target.exists():
            legacy.rename(target)

        out_images = target / "images"
        out_images.mkdir(parents=True, exist_ok=True)

        # 收集本批次帧集合（用于 summary）
        frame_ids_in_batch: List[int] = []

        for src in batch:
            fid_int, cam_serial, fid_str = parse_frame_cam_from_path(src)
            frame_ids_in_batch.append(fid_int)
            new_name = f"{fid_str}_{cam_serial}.png"
            dst = out_images / new_name

            # 覆盖已有文件（避免重复运行冲突）
            if dst.exists() or dst.is_symlink():
                dst.unlink()

            if symlink:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)

        # 批次帧范围与计数
        fids_sorted = sorted(set(frame_ids_in_batch))
        if fids_sorted:
            print(f"Batch {bn}: {len(batch)} images (frames {fids_sorted[0]}–{fids_sorted[-1]}) → {out_images}")
        else:
            print(f"Batch {bn}: {len(batch)} images (no frames?) → {out_images}")

        batch_records.append({
            "index": i,
            "name": bn,
            "path": str(target),
            "images_dir": str(out_images),
            "image_count": len(batch),
            "frames": fids_sorted,
            "frame_first": fids_sorted[0] if fids_sorted else None,
            "frame_last": fids_sorted[-1] if fids_sorted else None,
        })

    return (batches_root, total, PAD_WIDTH,
            frames_sorted, grouped, cam_serials, expected_per_frame, batch_records, issues)


# ---------------------------
# Run single-scene script & collect
# ---------------------------
def run_single_scene(demo_script: str, scene_path: Path, extra_args: List[str]):
    cmd = ["python", demo_script, "--scene_dir", str(scene_path)] + (extra_args or [])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_batches(scene_dir: Path, batches_root: Path, demo_script: str, pad_width: int,
                only_batch: int | None, max_batches: int | None, extra_demo_args: List[str]):
    sparse_root = scene_dir / "sparse"
    sparse_root.mkdir(exist_ok=True)

    # numeric sort handles '01' and '100' properly
    batch_dirs = sorted([p for p in batches_root.iterdir() if p.is_dir()],
                        key=lambda p: int(p.name))

    # Precedence: --only_batch > --max_batches
    if only_batch is not None:
        target = f"{only_batch:0{pad_width}d}"
        batch_dirs = [b for b in batch_dirs if b.name == target]
        if not batch_dirs:
            raise ValueError(f"--only_batch {only_batch} not found among {len(list(batches_root.iterdir()))} batches.")
    elif max_batches is not None:
        if max_batches < 1:
            raise ValueError("--max_batches must be >= 1")
        batch_dirs = batch_dirs[:max_batches]

    for b in batch_dirs:
        print(f"\n=== Running batch {b.name} ===")
        run_single_scene(demo_script, b, extra_demo_args)

        # Collect results back into scene_dir/sparse/XX/
        src_sparse = b / "sparse"
        dst_sparse = sparse_root / b.name
        if src_sparse.exists():
            if dst_sparse.exists():
                shutil.rmtree(dst_sparse)
            shutil.move(str(src_sparse), str(dst_sparse))
            print(f"→ Collected results to {dst_sparse}")
        else:
            print(f"⚠️  No sparse/ found in {b} (did the single-scene script run successfully?)")


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="VGGT Batch Wrapper (multi-camera tree, renaming, and summary)")
    ap.add_argument("--scene_dir", required=True, help="Path to scene folder containing images/")
    ap.add_argument("--batch_size", type=int, default=10, help="Frames per batch")
    ap.add_argument("--overlap_frames", type=int, default=0, help="Overlapping frames between batches")
    ap.add_argument("--demo_script", type=str, default="demo_colmap_perc.py", help="Single-scene script to run")
    ap.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying (faster, saves space)")
    ap.add_argument("--only_batch", type=int, default=None, help="Run only this (1-based) batch number")
    ap.add_argument("--max_batches", type=int, default=None, help="Only process the first N batches (1-based). Ignored if --only_batch is set")
    ap.add_argument("--demo_args", type=str, default="", help="Extra args for the single-scene script")
    ap.add_argument("--cams_per_frame", type=int, default=None,
                    help="Expected images per frame (= number of cameras). If not set, inferred from images/<cam>/rgb/")
    return ap.parse_args()


def main():
    args = parse_args()
    scene_dir = Path(args.scene_dir).resolve()
    extra_demo_args = args.demo_args.split() if args.demo_args else []

    (batches_root, total_created, pad_width,
     frames_sorted, grouped, cam_serials, expected_per_frame, batch_records, issues) = split_into_scenes(
        scene_dir,
        args.batch_size,
        args.overlap_frames,
        symlink=args.symlink,
        max_batches=args.max_batches if args.only_batch is None else None,
        cams_per_frame=args.cams_per_frame
    )

    print(f"\nTotal batches prepared: {total_created}")

    # 写 summary（在运行单场景脚本之前写，方便预览）
    write_summary(scene_dir, cam_serials, expected_per_frame, frames_sorted,
                  grouped, batches_root, batch_records, issues)

    # 运行/收集
    run_batches(scene_dir, batches_root, args.demo_script, pad_width,
                only_batch=args.only_batch, max_batches=args.max_batches, extra_demo_args=extra_demo_args)


if __name__ == "__main__":
    main()
