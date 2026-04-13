# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Semantic segmentation on video-reconstructed point clouds.
#
# Pipeline:
#   Video -> VGGT (3D reconstruction) -> Utonia encoder -> seg_head -> 20-class prediction
#
# Optional GT evaluation:
#   --gt-scene-dir <path>    : preprocessed scene folder with coord.npy / segment20.npy
#   --align-to-gt            : ICP-align VGGT points to GT before Utonia (test axis effect)
#   --use-gt-coord           : replace VGGT coords with GT coords entirely (upper bound)
#
# This lets you compare:
#   (A) VGGT coord only                        -> mIoU_vggt
#   (B) VGGT coord ICP-aligned to GT           -> mIoU_aligned
#   (C) GT coord directly                      -> mIoU_gt  (upper bound)
# to quantify how much axis/scale misalignment degrades segmentation.

import argparse
import os
import json
import time
import gc
import cv2
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
import utonia
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from einops import rearrange
from tqdm import tqdm
import glob
import shutil

try:
    import flash_attn
except ImportError:
    flash_attn = None

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# ScanNet 20-class metadata (shared with 2_sem_seg_scene.py)
# ---------------------------------------------------------------------------
VALID_CLASS_IDS_20 = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 14, 16, 24, 28, 33, 34, 36, 39,
)
CLASS_LABELS_20 = (
    "wall", "floor", "cabinet", "bed", "chair",
    "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator",
    "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
)
SCANNET_COLOR_MAP_20 = {
    0: (0., 0., 0.), 1: (174., 199., 232.), 2: (152., 223., 138.),
    3: (31., 119., 180.), 4: (255., 187., 120.), 5: (188., 189., 34.),
    6: (140., 86., 75.), 7: (255., 152., 150.), 8: (214., 39., 40.),
    9: (197., 176., 213.), 10: (148., 103., 189.), 11: (196., 156., 148.),
    12: (23., 190., 207.), 14: (247., 182., 210.), 15: (66., 188., 102.),
    16: (219., 219., 141.), 17: (140., 57., 197.), 18: (202., 185., 52.),
    19: (51., 176., 203.), 20: (200., 54., 131.),
}
CLASS_COLOR_20 = np.array([SCANNET_COLOR_MAP_20[i] for i in VALID_CLASS_IDS_20])


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


# ---------------------------------------------------------------------------
# Metrics (same as 2_sem_seg_scene.py)
# ---------------------------------------------------------------------------
def confusion_matrix(pred, gt, num_classes=20):
    valid = (gt >= 0) & (gt < num_classes)
    pred = np.clip(pred[valid].astype(np.int64), 0, num_classes - 1)
    gt = gt[valid].astype(np.int64)
    idx = gt * num_classes + pred
    return np.bincount(idx, minlength=num_classes**2).reshape(
        num_classes, num_classes
    ).astype(np.int64)


def metrics_from_confmat(confmat):
    nc = confmat.shape[0]
    tp = np.diag(confmat).astype(np.float64)
    gt_c = confmat.sum(axis=1).astype(np.float64)
    pr_c = confmat.sum(axis=0).astype(np.float64)
    union = gt_c + pr_c - tp
    ious = np.full(nc, np.nan)
    m = union > 0
    ious[m] = tp[m] / union[m]
    total = confmat.sum()
    acc = float(tp.sum() / total) if total > 0 else float("nan")
    valid_ious = ious[~np.isnan(ious)]
    miou = float(valid_ious.mean()) if valid_ious.size > 0 else float("nan")
    return {
        "num_points_evaluated": int(total),
        "overall_accuracy": acc,
        "mIoU": miou,
        "per_class_iou": {
            CLASS_LABELS_20[i]: (None if np.isnan(ious[i]) else float(ious[i]))
            for i in range(nc)
        },
    }


def print_metrics(metrics, title=""):
    if title:
        print(f"\n=== {title} ===")
    print(f"Points evaluated: {metrics['num_points_evaluated']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"mIoU:             {metrics['mIoU']:.4f}")
    print("Per-class IoU:")
    for label, iou in metrics["per_class_iou"].items():
        print(f"  {label:20s}: {iou:.4f}" if iou is not None else f"  {label:20s}:   n/a")


# ---------------------------------------------------------------------------
# VGGT reconstruction helpers (from 8_pca_video.py)
# ---------------------------------------------------------------------------
def rotx(x, theta=90):
    theta = np.deg2rad(theta)
    rot = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1],
    ])
    return rot @ x


def Coord2zup(points, extrinsics, normals=None):
    pts = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1).T
    pts = rotx(pts, -90)[:3].T
    if normals is not None:
        n4 = np.concatenate([normals, np.ones([normals.shape[0], 1])], axis=1).T
        normals = rotx(n4, -90)[:3].T
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    t = np.min(pts, axis=0)
    pts -= t
    extrinsics = rotx(extrinsics, -90)
    extrinsics[:, :3, 3] -= t.T
    return pts, extrinsics, normals


def extract_and_align_ground_plane(
    pcd, height_percentile=20, ransac_distance_threshold=0.01,
    ransac_n=3, ransac_iterations=1000, max_angle_degree=40, max_trials=6,
):
    points = np.asarray(pcd.points)
    z_vals = points[:, 2]
    z_thresh = np.percentile(z_vals, height_percentile)
    low_indices = np.where(z_vals <= z_thresh)[0]
    remaining_indices = low_indices.copy()
    for _trial in range(max_trials):
        if len(remaining_indices) < ransac_n:
            raise ValueError("Not enough points left to fit a plane.")
        low_pcd = pcd.select_by_index(remaining_indices)
        plane_model, inliers = low_pcd.segment_plane(
            distance_threshold=ransac_distance_threshold,
            ransac_n=ransac_n, num_iterations=ransac_iterations,
        )
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)
        angle = np.arccos(np.clip(np.dot(normal, [0, 0, 1]), -1, 1)) * 180 / np.pi
        if angle <= max_angle_degree:
            inliers_global = remaining_indices[inliers]
            target = np.array([0, 0, 1])
            axis = np.cross(normal, target)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-6:
                rotation_matrix = np.eye(3)
            else:
                axis /= axis_norm
                rot_angle = np.arccos(np.clip(np.dot(normal, target), -1, 1))
                rotation_matrix = R.from_rotvec(axis * rot_angle).as_matrix()
            rotated_points = points @ rotation_matrix.T
            offset = np.mean(rotated_points[inliers_global, 2])
            rotated_points[:, 2] -= offset
            aligned = o3d.geometry.PointCloud()
            aligned.points = o3d.utility.Vector3dVector(rotated_points)
            if pcd.has_colors():
                aligned.colors = pcd.colors
            if pcd.has_normals():
                aligned.normals = o3d.utility.Vector3dVector(
                    np.asarray(pcd.normals) @ rotation_matrix.T
                )
            return aligned, inliers_global, rotation_matrix, offset
        else:
            remaining_indices = np.setdiff1d(remaining_indices, remaining_indices[inliers])
    raise ValueError("Failed to find a valid ground plane within max trials.")


def reconstruct_from_video(
    video_path, conf_thres, frame_interval, prediction_mode, if_TSDF,
    output_dir,
):
    """Run VGGT on video and return (coord, color, normal) numpy arrays.

    Heavily based on 8_pca_video.py handle_uploads + parse_frames, but
    restructured to avoid globals and return raw arrays.
    """
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    import camtools as ct

    target_images = os.path.join(output_dir, "images")
    os.makedirs(target_images, exist_ok=True)

    # Extract frames
    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS)
    skip = int(fps * frame_interval)
    count, idx = 0, 0
    while True:
        ok, frame = vs.read()
        if not ok:
            break
        count += 1
        if count % skip == 0:
            cv2.imwrite(os.path.join(target_images, f"{idx:06d}.png"), frame)
            idx += 1
    vs.release()
    print(f"Extracted {idx} frames from video")

    # Load VGGT
    vggt_model = VGGT().to(device)
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    vggt_model.eval()

    image_names = sorted(glob.glob(os.path.join(target_images, "*")))
    images = load_and_preprocess_images(image_names).to(device)
    print(f"VGGT input: {images.shape}")

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            predictions = vggt_model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    for key in predictions:
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )
    predictions["world_points_from_depth"] = world_points

    Ts = ct.convert.pad_0001(predictions["extrinsic"])
    Ts_inv = np.linalg.inv(Ts)
    Cs = np.array([ct.convert.T_to_C(T) for T in Ts])

    world_points_data = predictions["world_points"]
    view_dirs = world_points_data - rearrange(Cs, "n c -> n 1 1 c")
    view_dirs = rearrange(view_dirs, "n h w c -> (n h w) c")
    view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=-1, keepdims=True)

    images_np = predictions["images"]
    points = rearrange(world_points_data, "n h w c -> (n h w) c")
    colors = rearrange(images_np, "n c h w -> (n h w) c")

    if prediction_mode == "Pointmap Branch":
        conf = predictions["world_points_conf"].reshape(-1)
        thr = np.percentile(conf, conf_thres) if conf_thres > 0 else 0.0
        mask = (conf >= thr) & (conf > 1e-5)
        points, colors = points[mask], colors[mask]
        view_dirs = view_dirs[mask]
        points, Ts_inv, _ = Coord2zup(points, Ts_inv)
        scale = 3 / (points[:, 2].max() - points[:, 2].min())
        points *= scale
    else:  # Depthmap and Camera Branch
        im_colors = rearrange(images_np, "n c h w -> (n) h w c")
        im_dists = world_points_data - rearrange(Cs, "n c -> n 1 1 c")
        im_dists = np.linalg.norm(im_dists, axis=-1)
        Ks = predictions["intrinsic"]
        im_depths = np.stack([
            ct.convert.im_distance_to_im_depth(d, K)
            for d, K in zip(im_dists, Ks)
        ])
        if if_TSDF:
            from demo_utils import integrate_rgbd_to_mesh  # noqa: avoid if unused
            raise NotImplementedError(
                "TSDF mode is not yet wired for sem_seg_video. "
                "Use --prediction_mode 'Pointmap Branch' or disable --if_TSDF."
            )
        else:
            pts_list = []
            for K, T, d in zip(Ks, Ts, im_depths):
                pt = ct.project.im_depth_to_point_cloud(
                    im_depth=d, K=K, T=T, to_image=False, ignore_invalid=False
                )
                pts_list.append(pt)
            points = np.vstack(pts_list)
            colors = im_colors.reshape(-1, 3)
            conf = predictions["depth_conf"].reshape(-1)
            thr = np.percentile(conf, conf_thres) if conf_thres > 0 else 0.0
            mask = (conf >= thr) & (conf > 1e-5)
            points, colors = points[mask], colors[mask]
            view_dirs_filtered = view_dirs[mask] if view_dirs.shape[0] == mask.shape[0] else None
            points, Ts_inv, _ = Coord2zup(points, Ts_inv)
            scale = 3 / (points[:, 2].max() - points[:, 2].min())
            points *= scale

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals()

    try:
        pcd, _, _, _ = extract_and_align_ground_plane(pcd)
    except Exception as e:
        print(f"Ground plane alignment failed: {e}")

    # Flip normals toward camera
    normals = np.asarray(pcd.normals)
    if view_dirs.shape[0] == normals.shape[0]:
        dot = np.sum(normals * view_dirs[:normals.shape[0]], axis=-1)
        normals[dot > 0] *= -1
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        pcd.normals = o3d.utility.Vector3dVector(normals)

    coord = np.asarray(pcd.points).astype(np.float64)
    color_out = np.asarray(pcd.colors).astype(np.float64)
    normal_out = np.asarray(pcd.normals).astype(np.float64)

    # Color from VGGT is [0,1] float; Utonia expects [0,255]
    if color_out.max() <= 1.0:
        color_out *= 255.0

    # Cleanup
    del predictions, vggt_model
    gc.collect()
    torch.cuda.empty_cache()

    return coord, color_out, normal_out


# ---------------------------------------------------------------------------
# GT alignment and label transfer
# ---------------------------------------------------------------------------
def load_gt_scene(scene_dir):
    """Load GT scene data: coord, segment20, (optional) color, normal."""
    coord = np.load(os.path.join(scene_dir, "coord.npy")).astype(np.float64)
    seg_path = os.path.join(scene_dir, "segment20.npy")
    segment = np.load(seg_path).astype(np.int64) if os.path.isfile(seg_path) else None
    color_path = os.path.join(scene_dir, "color.npy")
    color = np.load(color_path).astype(np.float64) if os.path.isfile(color_path) else None
    normal_path = os.path.join(scene_dir, "normal.npy")
    normal = np.load(normal_path).astype(np.float64) if os.path.isfile(normal_path) else None
    return {"coord": coord, "segment": segment, "color": color, "normal": normal}


def icp_align(source_pts, target_pts, max_correspondence_distance=0.5):
    """Align source_pts to target_pts using ICP. Returns (aligned_pts, transform_4x4).

    The function first does a coarse scale normalization, then ICP, then applies
    the combined transformation.
    """
    # Coarse scale alignment: match bounding box diagonals
    src_diag = np.linalg.norm(source_pts.max(0) - source_pts.min(0))
    tgt_diag = np.linalg.norm(target_pts.max(0) - target_pts.min(0))
    scale = tgt_diag / max(src_diag, 1e-8)

    # Center both
    src_center = source_pts.mean(0)
    tgt_center = target_pts.mean(0)
    src_scaled = (source_pts - src_center) * scale

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_scaled)
    pcd_tgt = o3d.geometry.PointCloud()
    tgt_centered = target_pts - tgt_center
    pcd_tgt.points = o3d.utility.Vector3dVector(tgt_centered)

    # ICP (point-to-point)
    reg = o3d.pipelines.registration.registration_icp(
        pcd_src, pcd_tgt,
        max_correspondence_distance=max_correspondence_distance * tgt_diag,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200,
        ),
    )
    T_icp = reg.transformation  # 4x4
    print(f"ICP fitness={reg.fitness:.4f}  inlier_rmse={reg.inlier_rmse:.6f}")

    # Apply full transform: center -> scale -> ICP -> un-center
    ones = np.ones((src_scaled.shape[0], 1))
    src_h = np.hstack([src_scaled, ones])
    aligned_h = (T_icp @ src_h.T).T[:, :3]
    aligned = aligned_h + tgt_center

    # Build composite 4x4 for reference
    T_full = np.eye(4)
    T_full[:3, :3] = T_icp[:3, :3] * scale
    T_full[:3, 3] = T_icp[:3, 3] + tgt_center - T_icp[:3, :3] @ (src_center * scale)

    return aligned, T_full, {
        "scale": float(scale),
        "fitness": float(reg.fitness),
        "inlier_rmse": float(reg.inlier_rmse),
    }


def transfer_gt_labels(
    query_pts, gt_pts, gt_labels, max_distance=0.1,
):
    """Transfer GT labels to query points via nearest-neighbor.

    Points farther than max_distance from any GT point get label -1 (ignore).
    Returns (transferred_labels, match_stats_dict).
    """
    tree = KDTree(gt_pts)
    dists, indices = tree.query(query_pts, k=1)
    labels = gt_labels[indices].copy()
    too_far = dists > max_distance
    labels[too_far] = -1

    stats = {
        "num_query": int(query_pts.shape[0]),
        "num_matched": int((~too_far).sum()),
        "num_unmatched": int(too_far.sum()),
        "match_rate": float((~too_far).mean()),
        "mean_dist": float(dists[~too_far].mean()) if (~too_far).any() else float("nan"),
        "median_dist": float(np.median(dists[~too_far])) if (~too_far).any() else float("nan"),
        "max_distance_threshold": float(max_distance),
    }
    return labels, stats


# ---------------------------------------------------------------------------
# Utonia inference
# ---------------------------------------------------------------------------
def run_utonia_semseg(coord, color, normal, model, seg_head, scale=0.5):
    """Run Utonia encoder + seg_head, return per-point predictions at original
    resolution (mapped back via inverse indices).
    """
    point = {"coord": coord.copy(), "color": color.copy(), "normal": normal.copy()}
    transform = utonia.transform.default(scale=scale)
    point = transform(point)

    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor) and device == "cuda":
                point[key] = point[key].cuda(non_blocking=True)
        point = model(point)
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        logits = seg_head(point.feat)
        pred_sampled = logits.argmax(dim=-1).cpu().numpy()

    # Map back to original resolution
    if "inverse" in point.keys():
        inv = point.inverse.cpu().numpy()
        pred_full = pred_sampled[inv]
    else:
        pred_full = pred_sampled

    coord_out = point.coord.cpu().numpy()
    return pred_full, pred_sampled, coord_out


# ---------------------------------------------------------------------------
# Visualization (matplotlib, headless-safe)
# ---------------------------------------------------------------------------
def render_comparison_png(
    coord, color_rgb, pred_labels, gt_labels,
    out_dir, point_size=1.5, max_points=120_000,
):
    """Render input / prediction / GT side-by-side from multiple views."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    def _sub(pts, *arrs):
        if pts.shape[0] > max_points:
            idx = rng.choice(pts.shape[0], max_points, replace=False)
            return (pts[idx],) + tuple(a[idx] for a in arrs)
        return (pts,) + arrs

    # Prepare colors
    pred_color = CLASS_COLOR_20[pred_labels] / 255.0
    if gt_labels is not None:
        gt_color = np.zeros((gt_labels.shape[0], 3))
        valid = gt_labels >= 0
        gt_color[valid] = CLASS_COLOR_20[gt_labels[valid]] / 255.0
    input_color = np.clip(color_rgb / 255.0, 0, 1) if color_rgb.max() > 1 else np.clip(color_rgb, 0, 1)

    panels = [("input_RGB", input_color), ("prediction", pred_color)]
    if gt_labels is not None:
        panels.append(("GT_segment20", gt_color))

    views = {
        "top":   (90, -90),
        "front": (0, -90),
        "side":  (0, 0),
        "iso":   (30, -45),
    }

    # Center
    center = coord.mean(axis=0)
    pts_c = coord - center

    def _set_equal(ax, pts):
        lim = np.abs(pts).max() * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

    for view_name, (elev, azim) in views.items():
        n = len(panels)
        fig = plt.figure(figsize=(7 * n, 7), dpi=120)
        fig.patch.set_facecolor("white")
        for i, (title, col) in enumerate(panels):
            pts_s, col_s = _sub(pts_c, col)
            ax = fig.add_subplot(1, n, i + 1, projection="3d")
            ax.set_facecolor("white")
            ax.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
                       c=col_s, s=point_size, marker=".", linewidths=0, depthshade=False)
            _set_equal(ax, pts_s)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            ax.set_title(title, fontsize=11)
        fig.tight_layout(pad=0.5)
        path = os.path.join(out_dir, f"comparison_{view_name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"  saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic segmentation on video-reconstructed point clouds"
    )

    # Video / VGGT args
    parser.add_argument("--input-video", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--conf-thres", type=float, default=10.0,
                        help="VGGT confidence threshold percentile")
    parser.add_argument("--frame-interval", type=float, default=1.0,
                        help="Frame interval in seconds")
    parser.add_argument("--prediction-mode", type=str,
                        choices=["Pointmap Branch", "Depthmap and Camera Branch"],
                        default="Depthmap and Camera Branch")
    parser.add_argument("--if-TSDF", action="store_true")

    # GT evaluation args
    parser.add_argument("--gt-scene-dir", type=str, default=None,
                        help="Path to preprocessed scene folder with coord.npy/segment20.npy "
                             "for GT label transfer and mIoU evaluation")
    parser.add_argument("--align-to-gt", action="store_true",
                        help="ICP-align VGGT points to GT before Utonia inference. "
                             "Tests whether axis/scale misalignment is the bottleneck.")
    parser.add_argument("--use-gt-coord", action="store_true",
                        help="Replace VGGT coords entirely with GT coords (keep VGGT "
                             "color/normal). Upper-bound test for coord quality.")
    parser.add_argument("--nn-max-dist", type=float, default=0.1,
                        help="Max distance for nearest-neighbor GT label transfer. "
                             "Points farther than this from any GT point get label -1.")
    parser.add_argument("--icp-max-corr", type=float, default=0.5,
                        help="ICP max correspondence distance (fraction of bbox diagonal)")

    # Utonia args
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Utonia transform scale (default 0.5)")
    parser.add_argument("--wo-color", action="store_true")
    parser.add_argument("--wo-normal", action="store_true")

    # Output args
    parser.add_argument("--out-dir", "-o", type=str, default="./sem_seg_video_out")
    parser.add_argument("--save-png", action="store_true",
                        help="Save comparison PNG (matplotlib, headless-safe)")
    parser.add_argument("--save-ply", action="store_true",
                        help="Save PLY files (input/pred/gt)")
    parser.add_argument("--max-points", type=int, default=120_000,
                        help="Max points for matplotlib rendering")
    parser.add_argument("--point-size", type=float, default=1.5)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Step 1: VGGT Reconstruction ----
    print("=" * 60)
    print("Step 1: VGGT 3D reconstruction from video")
    print("=" * 60)
    t0 = time.time()
    vggt_coord, vggt_color, vggt_normal = reconstruct_from_video(
        video_path=args.input_video,
        conf_thres=args.conf_thres,
        frame_interval=args.frame_interval,
        prediction_mode=args.prediction_mode,
        if_TSDF=args.if_TSDF,
        output_dir=args.out_dir,
    )
    t_vggt = time.time() - t0
    print(f"VGGT reconstruction: {vggt_coord.shape[0]} points ({t_vggt:.1f}s)")

    # ---- Step 2: Load GT (optional) ----
    gt = None
    if args.gt_scene_dir:
        print(f"\nLoading GT scene: {args.gt_scene_dir}")
        gt = load_gt_scene(args.gt_scene_dir)
        print(f"  GT points: {gt['coord'].shape[0]}")
        if gt["segment"] is not None:
            valid = gt["segment"] >= 0
            print(f"  GT labels:  {valid.sum()} valid / {len(valid)} total")

    # ---- Step 3: Coordinate alignment ----
    results = {}
    coord_for_utonia = vggt_coord
    color_for_utonia = vggt_color
    normal_for_utonia = vggt_normal

    if args.use_gt_coord and gt is not None:
        # Mode C: use GT coords, keep VGGT color/normal via NN
        print("\n[Mode C] Using GT coordinates directly")
        coord_for_utonia = gt["coord"]
        # Transfer VGGT color/normal to GT points via NN
        tree = KDTree(vggt_coord)
        _, nn_idx = tree.query(gt["coord"], k=1)
        color_for_utonia = vggt_color[nn_idx]
        normal_for_utonia = vggt_normal[nn_idx]
        if gt.get("color") is not None:
            # Actually use GT color too for the true upper bound
            color_for_utonia = gt["color"]
        if gt.get("normal") is not None:
            normal_for_utonia = gt["normal"]
        results["alignment_mode"] = "use_gt_coord"

    elif args.align_to_gt and gt is not None:
        # Mode B: ICP align VGGT -> GT
        print("\n[Mode B] ICP-aligning VGGT points to GT")
        aligned, T_icp, icp_stats = icp_align(
            vggt_coord, gt["coord"],
            max_correspondence_distance=args.icp_max_corr,
        )
        print(f"  scale={icp_stats['scale']:.4f}  "
              f"fitness={icp_stats['fitness']:.4f}  "
              f"rmse={icp_stats['inlier_rmse']:.6f}")
        coord_for_utonia = aligned
        results["alignment_mode"] = "icp_to_gt"
        results["icp_stats"] = icp_stats
    else:
        results["alignment_mode"] = "none (VGGT native)"

    if args.wo_color:
        color_for_utonia = np.zeros_like(coord_for_utonia)
    if args.wo_normal:
        normal_for_utonia = np.zeros_like(coord_for_utonia)

    # ---- Step 4: Utonia Semantic Segmentation ----
    print("\n" + "=" * 60)
    print("Step 4: Utonia semantic segmentation")
    print("=" * 60)

    utonia.utils.set_seed(46647087)

    # Load models
    print("Loading Utonia model + seg head...")
    if flash_attn is not None:
        model = utonia.load("utonia", repo_id="Pointcept/Utonia").to(device)
    else:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],
            enable_flash=False,
        )
        model = utonia.load(
            "utonia", repo_id="Pointcept/Utonia", custom_config=custom_config
        ).to(device)
    ckpt = utonia.load("utonia_linear_prob_head_sc", repo_id="Pointcept/Utonia", ckpt_only=True)
    seg_head = SegHead(**ckpt["config"]).to(device)
    seg_head.load_state_dict(ckpt["state_dict"])
    model.eval()
    seg_head.eval()

    t0 = time.time()
    pred_full, pred_sampled, coord_sampled = run_utonia_semseg(
        coord_for_utonia, color_for_utonia, normal_for_utonia,
        model, seg_head, scale=args.scale,
    )
    t_utonia = time.time() - t0
    print(f"Inference: {coord_for_utonia.shape[0]} pts -> {pred_full.shape[0]} predictions ({t_utonia:.1f}s)")

    # Class distribution
    unique, counts = np.unique(pred_full, return_counts=True)
    print("\nPredicted class distribution:")
    for cls_idx, cnt in zip(unique, counts):
        pct = 100.0 * cnt / pred_full.shape[0]
        print(f"  {CLASS_LABELS_20[cls_idx]:20s}: {cnt:8d} ({pct:5.1f}%)")

    results["num_points_vggt"] = int(vggt_coord.shape[0])
    results["num_points_utonia_input"] = int(coord_for_utonia.shape[0])
    results["vggt_time_s"] = float(t_vggt)
    results["utonia_time_s"] = float(t_utonia)

    # ---- Step 5: GT label transfer + mIoU ----
    gt_labels_transferred = None
    if gt is not None and gt["segment"] is not None:
        print("\n" + "=" * 60)
        print("Step 5: GT label transfer + evaluation")
        print("=" * 60)

        gt_labels_transferred, match_stats = transfer_gt_labels(
            query_pts=coord_for_utonia,
            gt_pts=gt["coord"],
            gt_labels=gt["segment"],
            max_distance=args.nn_max_dist,
        )
        print(f"Label transfer: {match_stats['num_matched']}/{match_stats['num_query']} "
              f"matched ({match_stats['match_rate']:.1%}), "
              f"mean_dist={match_stats['mean_dist']:.4f}")
        results["label_transfer"] = match_stats

        confmat = confusion_matrix(pred_full, gt_labels_transferred, num_classes=20)
        metrics = metrics_from_confmat(confmat)
        print_metrics(metrics, title=f"mIoU ({results['alignment_mode']})")
        results["metrics"] = metrics

        np.save(os.path.join(args.out_dir, "confmat.npy"), confmat)
    else:
        print("\nNo GT available; skipping mIoU evaluation.")

    # ---- Step 6: Save outputs ----
    # metrics.json
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved: {os.path.join(args.out_dir, 'metrics.json')}")

    # PLY
    if args.save_ply:
        ply_dir = os.path.join(args.out_dir, "ply")
        os.makedirs(ply_dir, exist_ok=True)
        # Input
        pc = trimesh.PointCloud(vertices=coord_for_utonia,
                                colors=np.clip(color_for_utonia / 255.0, 0, 1))
        pc.export(os.path.join(ply_dir, "input.ply"))
        # Prediction
        pred_rgb = CLASS_COLOR_20[pred_full] / 255.0
        pc = trimesh.PointCloud(vertices=coord_for_utonia, colors=pred_rgb)
        pc.export(os.path.join(ply_dir, "pred_seg20.ply"))
        # GT
        if gt_labels_transferred is not None:
            gt_rgb = np.zeros((gt_labels_transferred.shape[0], 3))
            valid = gt_labels_transferred >= 0
            gt_rgb[valid] = CLASS_COLOR_20[gt_labels_transferred[valid]] / 255.0
            pc = trimesh.PointCloud(vertices=coord_for_utonia, colors=gt_rgb)
            pc.export(os.path.join(ply_dir, "gt_seg20.ply"))
        print(f"PLY saved: {ply_dir}")

    # PNG
    if args.save_png:
        png_dir = os.path.join(args.out_dir, "png")
        render_comparison_png(
            coord_for_utonia, color_for_utonia, pred_full,
            gt_labels_transferred, png_dir,
            point_size=args.point_size, max_points=args.max_points,
        )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Alignment mode:  {results['alignment_mode']}")
    print(f"VGGT points:     {results['num_points_vggt']}")
    print(f"Utonia input:    {results['num_points_utonia_input']}")
    print(f"VGGT time:       {results['vggt_time_s']:.1f}s")
    print(f"Utonia time:     {results['utonia_time_s']:.1f}s")
    if "metrics" in results:
        print(f"Overall acc:     {results['metrics']['overall_accuracy']:.4f}")
        print(f"mIoU:            {results['metrics']['mIoU']:.4f}")
    print(f"Output dir:      {args.out_dir}")
