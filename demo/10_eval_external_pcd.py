# Evaluate external point cloud segmentation results against GT.
#
# Accepts an external point cloud with per-point labels (e.g. from another
# model's output) and compares them to ScanNet GT via:
#   1. Optional ICP alignment to GT coordinate frame
#   2. KDTree nearest-neighbor label transfer for GT association
#   3. Confusion matrix / mIoU / accuracy / per-class IoU
#   4. Multi-view matplotlib visualization (headless-safe)
#
# Supported input formats:
#   - .npy  : (N, 4+) array where columns are [x, y, z, label, ...]
#   - .npz  : keys "coord" (N,3) + "label" (N,), optionally "color" (N,3)
#   - .txt / .csv : space/comma-separated, columns [x, y, z, label]
#   - .ply  : Open3D readable, with scalar_Label or label field
#
# GT format: preprocessed scene folder with coord.npy, segment20.npy,
#            optionally color.npy, normal.npy

import argparse
import os
import json
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


# ---------------------------------------------------------------------------
# ScanNet 20-class metadata
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
    12: (23., 190., 207.), 14: (247., 182., 210.), 16: (219., 219., 141.),
    24: (255., 127., 14.), 28: (158., 218., 229.), 33: (44., 160., 44.),
    34: (112., 128., 144.), 36: (227., 119., 194.), 39: (82., 84., 163.),
}
CLASS_COLOR_20 = np.array([SCANNET_COLOR_MAP_20[i] for i in VALID_CLASS_IDS_20])


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def load_external_pcd(path):
    """Load external point cloud. Returns (coord (N,3), labels (N,), color (N,3) or None)."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
        coord = arr[:, :3].astype(np.float64)
        labels = arr[:, 3].astype(np.int64)
        color = arr[:, 4:7].astype(np.float64) if arr.shape[1] >= 7 else None
        return coord, labels, color

    elif ext == ".npz":
        data = dict(np.load(path))
        coord = data["coord"].astype(np.float64)
        labels = data["label"].astype(np.int64)
        color = data.get("color", None)
        if color is not None:
            color = color.astype(np.float64)
        return coord, labels, color

    elif ext in (".txt", ".csv"):
        arr = np.loadtxt(path, delimiter=None)  # auto-detect whitespace/comma
        coord = arr[:, :3].astype(np.float64)
        labels = arr[:, 3].astype(np.int64)
        color = arr[:, 4:7].astype(np.float64) if arr.shape[1] >= 7 else None
        return coord, labels, color

    elif ext == ".ply":
        pcd = o3d.io.read_point_cloud(path)
        coord = np.asarray(pcd.points, dtype=np.float64)
        color = np.asarray(pcd.colors, dtype=np.float64) * 255.0 if pcd.has_colors() else None
        # Try reading label from PLY as custom property via trimesh
        try:
            import trimesh
            mesh = trimesh.load(path, process=False)
            if hasattr(mesh, "metadata") and "ply_raw" in mesh.metadata:
                vdata = mesh.metadata["ply_raw"]["vertex"]["data"]
                for key in ("label", "scalar_Label", "class", "semantic"):
                    if key in vdata.dtype.names:
                        labels = np.asarray(vdata[key], dtype=np.int64).ravel()
                        return coord, labels, color
            # fallback: vertex_data
            if hasattr(mesh, "vertices") and hasattr(mesh, "visual"):
                pass  # no label found
        except Exception:
            pass
        raise ValueError(
            f"Could not find label field in PLY file: {path}. "
            "Expected a 'label' or 'scalar_Label' vertex property."
        )

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def load_gt_scene(scene_dir, gt_segment="segment20"):
    """Load GT from preprocessed scene folder."""
    coord = np.load(os.path.join(scene_dir, "coord.npy")).astype(np.float64)
    labels = np.load(os.path.join(scene_dir, f"{gt_segment}.npy")).astype(np.int64)
    color_path = os.path.join(scene_dir, "color.npy")
    color = np.load(color_path).astype(np.float64) if os.path.exists(color_path) else None
    return coord, labels, color


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------
def remove_statistical_outliers(coord, labels, color=None, nb_neighbors=20, std_ratio=2.0):
    """Remove statistical outliers using Open3D SOR.

    Points whose mean distance to k-nearest neighbors exceeds
    (global_mean + std_ratio * global_std) are removed.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    _, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio,
    )
    inlier_idx = np.asarray(inlier_idx)
    n_before = coord.shape[0]
    coord = coord[inlier_idx]
    labels = labels[inlier_idx]
    if color is not None:
        color = color[inlier_idx]
    n_removed = n_before - coord.shape[0]
    print(f"  SOR: {n_before} -> {coord.shape[0]} ({n_removed} removed, "
          f"nb={nb_neighbors}, std={std_ratio})")
    return coord, labels, color


def clip_to_bbox(coord, labels, ref_coord, margin=0.1, color=None):
    """Remove points outside the bounding box of ref_coord (+ margin).

    margin is a fraction of the bbox diagonal added on each side.
    """
    ref_min = ref_coord.min(axis=0)
    ref_max = ref_coord.max(axis=0)
    diag = np.linalg.norm(ref_max - ref_min)
    pad = diag * margin
    lo = ref_min - pad
    hi = ref_max + pad
    mask = np.all((coord >= lo) & (coord <= hi), axis=1)
    n_before = coord.shape[0]
    coord = coord[mask]
    labels = labels[mask]
    if color is not None:
        color = color[mask]
    n_removed = n_before - coord.shape[0]
    print(f"  BBox clip: {n_before} -> {coord.shape[0]} ({n_removed} removed, "
          f"margin={margin})")
    return coord, labels, color


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def _umeyama(src, dst):
    """Estimate similarity transform (scale, R, t) from corresponding points.

    Solves:  dst = scale * R @ src + t   (least-squares, closed-form).
    Returns scale (float), R (3x3), t (3,).
    Reference: Umeyama, PAMI 1991.
    """
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = (src_c ** 2).sum() / n
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = (D * np.diag(S)).sum() / var_s
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def _make_pcd(pts, voxel_size=0.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def robust_align(source_pts, target_pts, downsample_voxel=0.05,
                 ransac_distance_multiplier=0.05, icp_refine=True,
                 percentile_scale=True):
    """Robust alignment: FPFH+RANSAC global registration → Umeyama scale → ICP refine.

    Steps:
      1. Coarse scale via percentile-based bbox (robust to outliers)
      2. Downsample + FPFH feature extraction
      3. RANSAC-based global registration (robust to outliers/ghosts)
      4. Umeyama on RANSAC inlier correspondences → refine scale+R+t
      5. Optional ICP refinement on the Umeyama-aligned result

    Returns (aligned_pts, stats_dict).
    """
    # --- Step 1: Robust scale estimation ---
    if percentile_scale:
        # Use 5th-95th percentile extents instead of min/max
        src_lo = np.percentile(source_pts, 5, axis=0)
        src_hi = np.percentile(source_pts, 95, axis=0)
        tgt_lo = np.percentile(target_pts, 5, axis=0)
        tgt_hi = np.percentile(target_pts, 95, axis=0)
    else:
        src_lo, src_hi = source_pts.min(0), source_pts.max(0)
        tgt_lo, tgt_hi = target_pts.min(0), target_pts.max(0)

    src_diag = np.linalg.norm(src_hi - src_lo)
    tgt_diag = np.linalg.norm(tgt_hi - tgt_lo)
    coarse_scale = tgt_diag / max(src_diag, 1e-8)

    src_center = np.median(source_pts, axis=0)
    tgt_center = np.median(target_pts, axis=0)
    src_prescaled = (source_pts - src_center) * coarse_scale + tgt_center

    print(f"  Coarse scale: {coarse_scale:.4f} "
          f"(src_diag={src_diag:.3f}, tgt_diag={tgt_diag:.3f})")

    # --- Step 2: Downsample + FPFH ---
    # Use a voxel size relative to target diagonal for consistent feature extraction
    if downsample_voxel > 0:
        fpfh_voxel = downsample_voxel
    else:
        fpfh_voxel = tgt_diag * 0.01  # fallback: 1% of target diagonal

    pcd_src = _make_pcd(src_prescaled, fpfh_voxel)
    pcd_tgt = _make_pcd(target_pts, fpfh_voxel)
    print(f"  Downsampled: source {source_pts.shape[0]}->{len(pcd_src.points)}, "
          f"target {target_pts.shape[0]}->{len(pcd_tgt.points)} "
          f"(voxel={fpfh_voxel:.4f})")

    radius_normal = fpfh_voxel * 2
    radius_feature = fpfh_voxel * 5

    pcd_src.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_tgt.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    fpfh_src = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_src, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    fpfh_tgt = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # --- Step 3: RANSAC global registration ---
    ransac_dist = tgt_diag * ransac_distance_multiplier
    print(f"  RANSAC (max_corr_dist={ransac_dist:.4f})...")
    reg_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_src, pcd_tgt, fpfh_src, fpfh_tgt,
        mutual_filter=True,
        max_correspondence_distance=ransac_dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    T_ransac = reg_ransac.transformation
    print(f"  RANSAC fitness={reg_ransac.fitness:.4f}, "
          f"inlier_rmse={reg_ransac.inlier_rmse:.6f}")

    # Apply RANSAC transform to prescaled source
    src_ransac = (T_ransac[:3, :3] @ src_prescaled.T).T + T_ransac[:3, 3]

    # --- Step 4: Umeyama on correspondences → refine scale ---
    # Find correspondences within ransac_dist
    pcd_src_ransac = _make_pcd(src_ransac, fpfh_voxel)
    pcd_tgt_for_corr = _make_pcd(target_pts, fpfh_voxel)

    src_tree = KDTree(np.asarray(pcd_tgt_for_corr.points))
    src_ransac_down = np.asarray(pcd_src_ransac.points)
    dists, indices = src_tree.query(src_ransac_down, k=1)
    inlier_mask = dists < ransac_dist
    n_inliers = inlier_mask.sum()
    print(f"  Correspondence inliers: {n_inliers}/{len(src_ransac_down)}")

    umeyama_scale = 1.0
    if n_inliers >= 10:
        corr_src = src_ransac_down[inlier_mask]
        corr_tgt = np.asarray(pcd_tgt_for_corr.points)[indices[inlier_mask]]
        u_scale, u_R, u_t = _umeyama(corr_src, corr_tgt)
        print(f"  Umeyama refinement: scale={u_scale:.4f}")
        src_final = u_scale * (u_R @ src_ransac.T).T + u_t
        umeyama_scale = u_scale
    else:
        print("  Too few inliers for Umeyama, using RANSAC transform only")
        src_final = src_ransac

    # --- Step 5: Optional ICP refinement ---
    icp_stats = {}
    if icp_refine:
        pcd_src_final = _make_pcd(src_final)
        pcd_tgt_full = _make_pcd(target_pts)
        if downsample_voxel > 0:
            pcd_src_icp = pcd_src_final.voxel_down_sample(downsample_voxel)
            pcd_tgt_icp = pcd_tgt_full.voxel_down_sample(downsample_voxel)
        else:
            pcd_src_icp = pcd_src_final
            pcd_tgt_icp = pcd_tgt_full

        icp_dist = tgt_diag * 0.02  # tight distance for refinement
        reg_icp = o3d.pipelines.registration.registration_icp(
            pcd_src_icp, pcd_tgt_icp,
            max_correspondence_distance=icp_dist,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )
        T_icp = reg_icp.transformation
        src_final = (T_icp[:3, :3] @ src_final.T).T + T_icp[:3, 3]
        icp_stats = {
            "icp_fitness": float(reg_icp.fitness),
            "icp_inlier_rmse": float(reg_icp.inlier_rmse),
        }
        print(f"  ICP refine: fitness={reg_icp.fitness:.4f}, "
              f"rmse={reg_icp.inlier_rmse:.6f}")

    total_scale = coarse_scale * umeyama_scale
    stats = {
        "coarse_scale": float(coarse_scale),
        "umeyama_scale": float(umeyama_scale),
        "total_scale": float(total_scale),
        "ransac_fitness": float(reg_ransac.fitness),
        "ransac_inlier_rmse": float(reg_ransac.inlier_rmse),
        "n_correspondence_inliers": int(n_inliers),
        **icp_stats,
    }
    return src_final, stats


def icp_align(source_pts, target_pts, max_correspondence_distance=0.5,
              icp_downsample_voxel=0.05):
    """Legacy ICP alignment (bbox-diagonal scale + rigid ICP). Returns (aligned_pts, T_4x4, stats)."""
    src_diag = np.linalg.norm(source_pts.max(0) - source_pts.min(0))
    tgt_diag = np.linalg.norm(target_pts.max(0) - target_pts.min(0))
    scale = tgt_diag / max(src_diag, 1e-8)

    src_center = source_pts.mean(0)
    tgt_center = target_pts.mean(0)
    src_scaled = (source_pts - src_center) * scale
    tgt_centered = target_pts - tgt_center

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_scaled)
    pcd_tgt = o3d.geometry.PointCloud()
    pcd_tgt.points = o3d.utility.Vector3dVector(tgt_centered)

    if icp_downsample_voxel > 0:
        n_src = len(pcd_src.points)
        n_tgt = len(pcd_tgt.points)
        pcd_src_down = pcd_src.voxel_down_sample(icp_downsample_voxel)
        pcd_tgt_down = pcd_tgt.voxel_down_sample(icp_downsample_voxel)
        print(f"  ICP downsample: source {n_src}->{len(pcd_src_down.points)}, "
              f"target {n_tgt}->{len(pcd_tgt_down.points)} "
              f"(voxel={icp_downsample_voxel})")
    else:
        pcd_src_down = pcd_src
        pcd_tgt_down = pcd_tgt

    reg = o3d.pipelines.registration.registration_icp(
        pcd_src_down, pcd_tgt_down,
        max_correspondence_distance=max_correspondence_distance * tgt_diag,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    T_icp = reg.transformation
    print(f"  ICP fitness={reg.fitness:.4f}  inlier_rmse={reg.inlier_rmse:.6f}")

    ones = np.ones((src_scaled.shape[0], 1))
    src_h = np.hstack([src_scaled, ones])
    aligned_h = (T_icp @ src_h.T).T[:, :3]
    aligned = aligned_h + tgt_center

    T_full = np.eye(4)
    T_full[:3, :3] = T_icp[:3, :3] * scale
    T_full[:3, 3] = T_icp[:3, 3] + tgt_center - T_icp[:3, :3] @ (src_center * scale)

    return aligned, T_full, {
        "scale": float(scale),
        "fitness": float(reg.fitness),
        "inlier_rmse": float(reg.inlier_rmse),
    }


# ---------------------------------------------------------------------------
# Label transfer via nearest neighbor
# ---------------------------------------------------------------------------
def transfer_gt_labels(query_pts, gt_pts, gt_labels, max_distance=0.1):
    """Transfer GT labels to query points via KDTree NN. Returns (labels, stats)."""
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
# Metrics
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
# Visualization (matplotlib, headless-safe)
# ---------------------------------------------------------------------------
def render_comparison_png(
    pred_coord, pred_labels, gt_coord, gt_labels,
    out_dir, pred_color=None, gt_color=None,
    gt_transferred_labels=None,
    point_size=1.5, max_points=120_000,
):
    """Render multi-panel comparison from multiple views.

    Panels:
      1. Prediction (pred geometry + predicted seg colors)
      2. GT reference (GT geometry + GT seg colors)
      3. Pred RGB (pred geometry + original color)       [if pred_color given]
      4. GT RGB (GT geometry + original color)            [if gt_color given]
      5. GT transferred (pred geometry + transferred GT labels) [if gt_transferred_labels given]
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    def _sub(pts, *arrs):
        if pts.shape[0] > max_points:
            idx = rng.choice(pts.shape[0], max_points, replace=False)
            return (pts[idx],) + tuple(a[idx] for a in arrs)
        return (pts,) + arrs

    def _labels_to_color(labels):
        c = np.zeros((labels.shape[0], 3))
        valid = (labels >= 0) & (labels < 20)
        c[valid] = CLASS_COLOR_20[labels[valid]] / 255.0
        return c

    def _norm_rgb(c):
        return np.clip(c / 255.0, 0, 1) if c.max() > 1 else np.clip(c, 0, 1)

    panels = [
        ("prediction", pred_coord, _labels_to_color(pred_labels)),
        ("GT reference", gt_coord, _labels_to_color(gt_labels)),
    ]
    if pred_color is not None:
        panels.append(("pred RGB", pred_coord, _norm_rgb(pred_color)))
    if gt_color is not None:
        panels.append(("GT RGB", gt_coord, _norm_rgb(gt_color)))
    if gt_transferred_labels is not None:
        panels.append(("GT transferred\n(on pred geom)", pred_coord,
                        _labels_to_color(gt_transferred_labels)))

    views = {
        "top":   (90, -90),
        "front": (0, -90),
        "side":  (0, 0),
        "iso":   (30, -45),
    }

    def _set_equal(ax, pts):
        lim = np.abs(pts).max() * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

    for view_name, (elev, azim) in views.items():
        n = len(panels)
        fig = plt.figure(figsize=(6 * n, 6), dpi=120)
        fig.patch.set_facecolor("white")
        for i, (title, pts, col) in enumerate(panels):
            pts_c = pts - pts.mean(axis=0)
            pts_s, col_s = _sub(pts_c, col)
            ax = fig.add_subplot(1, n, i + 1, projection="3d")
            ax.set_facecolor("white")
            ax.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
                       c=col_s, s=point_size, marker=".", linewidths=0,
                       depthshade=False)
            _set_equal(ax, pts_s)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            ax.set_title(title, fontsize=10)
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
        description="Evaluate external point cloud segmentation results against GT"
    )
    parser.add_argument("input", type=str,
                        help="Path to external pcd file (.npy/.npz/.txt/.csv/.ply). "
                             "Expected columns: x, y, z, label [, r, g, b]")
    parser.add_argument("--gt-scene-dir", type=str, required=True,
                        help="Path to GT scene folder (coord.npy, segment20.npy, ...)")
    parser.add_argument("--gt-segment", type=str, default="segment20",
                        help="GT segment file name without .npy (default: segment20)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: next to input file)")

    # Outlier removal
    parser.add_argument("--sor", action="store_true",
                        help="Apply Statistical Outlier Removal before ICP")
    parser.add_argument("--sor-neighbors", type=int, default=20,
                        help="SOR: number of neighbors to consider (default: 20)")
    parser.add_argument("--sor-std", type=float, default=2.0,
                        help="SOR: std ratio threshold (default: 2.0, lower=stricter)")
    parser.add_argument("--clip-bbox", action="store_true",
                        help="Clip points to GT bounding box after ICP alignment")
    parser.add_argument("--clip-margin", type=float, default=0.1,
                        help="BBox clip: margin as fraction of GT diagonal (default: 0.1)")

    # Alignment
    align_group = parser.add_mutually_exclusive_group()
    align_group.add_argument("--no-align", action="store_true",
                             help="Skip alignment (assume coords are already aligned)")
    align_group.add_argument("--legacy-icp", action="store_true",
                             help="Use legacy ICP (bbox-scale + rigid ICP) instead of "
                                  "robust alignment")
    parser.add_argument("--align-downsample-voxel", type=float, default=0.05,
                        help="Voxel size for alignment downsampling (default: 0.05)")
    parser.add_argument("--ransac-dist-mult", type=float, default=0.05,
                        help="RANSAC max correspondence distance as fraction of "
                             "target bbox diagonal (default: 0.05)")
    parser.add_argument("--no-icp-refine", action="store_true",
                        help="Skip ICP refinement after RANSAC+Umeyama")

    # Post-alignment outlier removal
    parser.add_argument("--post-align-nn-filter", type=float, default=0,
                        help="After alignment, remove points farther than this distance "
                             "from any GT point. 0 = disabled (default). "
                             "Recommended: 0.1-0.3")

    # Label transfer
    parser.add_argument("--nn-max-dist", type=float, default=0.1,
                        help="Max distance for NN label transfer (in GT coord units)")

    # Visualization
    parser.add_argument("--save-png", action="store_true",
                        help="Save comparison PNG visualizations")
    parser.add_argument("--max-points", type=int, default=120_000,
                        help="Max points for visualization subsampling")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(args.input),
            os.path.splitext(os.path.basename(args.input))[0] + "_eval",
        )
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load ----
    print(f"Loading external pcd: {args.input}")
    pred_coord, pred_labels, pred_color = load_external_pcd(args.input)
    print(f"  {pred_coord.shape[0]} points, "
          f"label range [{pred_labels.min()}, {pred_labels.max()}]")

    print(f"Loading GT: {args.gt_scene_dir}")
    gt_coord, gt_labels, gt_color = load_gt_scene(args.gt_scene_dir, args.gt_segment)
    print(f"  {gt_coord.shape[0]} points, "
          f"label range [{gt_labels.min()}, {gt_labels.max()}]")

    # ---- Outlier removal (before ICP) ----
    if args.sor:
        print("\nStatistical Outlier Removal...")
        pred_coord, pred_labels, pred_color = remove_statistical_outliers(
            pred_coord, pred_labels, pred_color,
            nb_neighbors=args.sor_neighbors, std_ratio=args.sor_std,
        )

    # ---- Alignment ----
    align_stats = None
    if args.no_align:
        pred_coord_aligned = pred_coord
        print("\nSkipping alignment (--no-align)")
    elif args.legacy_icp:
        print("\nRunning legacy ICP alignment...")
        pred_coord_aligned, _, align_stats = icp_align(
            pred_coord, gt_coord,
            max_correspondence_distance=args.ransac_dist_mult,
            icp_downsample_voxel=args.align_downsample_voxel,
        )
    else:
        print("\nRunning robust alignment (RANSAC + Umeyama + ICP)...")
        pred_coord_aligned, align_stats = robust_align(
            pred_coord, gt_coord,
            downsample_voxel=args.align_downsample_voxel,
            ransac_distance_multiplier=args.ransac_dist_mult,
            icp_refine=not args.no_icp_refine,
        )

    # ---- Post-alignment outlier removal (NN distance to GT) ----
    if args.post_align_nn_filter > 0:
        print(f"\nPost-alignment NN filter (max_dist={args.post_align_nn_filter})...")
        tree = KDTree(gt_coord)
        dists, _ = tree.query(pred_coord_aligned, k=1)
        keep = dists <= args.post_align_nn_filter
        n_before = pred_coord_aligned.shape[0]
        pred_coord_aligned = pred_coord_aligned[keep]
        pred_labels = pred_labels[keep]
        if pred_color is not None:
            pred_color = pred_color[keep]
        print(f"  {n_before} -> {pred_coord_aligned.shape[0]} "
              f"({n_before - pred_coord_aligned.shape[0]} removed)")

    # ---- BBox clip (after alignment, in GT coordinate frame) ----
    if args.clip_bbox:
        print("\nClipping to GT bounding box...")
        pred_coord_aligned, pred_labels, pred_color = clip_to_bbox(
            pred_coord_aligned, pred_labels, gt_coord,
            margin=args.clip_margin, color=pred_color,
        )

    # ---- Transfer GT labels to prediction points via NN ----
    print(f"\nTransferring GT labels (nn_max_dist={args.nn_max_dist})...")
    gt_transferred, transfer_stats = transfer_gt_labels(
        pred_coord_aligned, gt_coord, gt_labels,
        max_distance=args.nn_max_dist,
    )
    print(f"  matched: {transfer_stats['num_matched']}/{transfer_stats['num_query']} "
          f"({transfer_stats['match_rate']:.1%})")
    print(f"  mean_dist={transfer_stats['mean_dist']:.4f}, "
          f"median_dist={transfer_stats['median_dist']:.4f}")

    # ---- Compute metrics ----
    print("\nComputing metrics...")
    confmat = confusion_matrix(pred_labels, gt_transferred, num_classes=20)
    metrics = metrics_from_confmat(confmat)
    print_metrics(metrics, title="Evaluation Results")

    # ---- Save results ----
    results = {
        "input": os.path.abspath(args.input),
        "gt_scene_dir": os.path.abspath(args.gt_scene_dir),
        "gt_segment": args.gt_segment,
        "align_mode": "none" if args.no_align else ("legacy_icp" if args.legacy_icp else "robust"),
        "align_stats": align_stats,
        "transfer_stats": transfer_stats,
        "metrics": metrics,
    }
    results_path = os.path.join(args.out_dir, "metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics: {results_path}")

    confmat_path = os.path.join(args.out_dir, "confmat.npy")
    np.save(confmat_path, confmat)
    print(f"Saved confusion matrix: {confmat_path}")

    # ---- Visualization ----
    if args.save_png:
        print("\nRendering comparison PNGs...")
        render_comparison_png(
            pred_coord=pred_coord_aligned,
            pred_labels=pred_labels,
            gt_coord=gt_coord,
            gt_labels=gt_labels,
            out_dir=args.out_dir,
            pred_color=pred_color,
            gt_color=gt_color,
            gt_transferred_labels=gt_transferred,
            max_points=args.max_points,
        )

    print(f"\nDone. Results saved to: {args.out_dir}")
