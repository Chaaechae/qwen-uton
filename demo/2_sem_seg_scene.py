# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Semantic segmentation demo that loads a preprocessed ScanNet-style scene
# folder (containing color.npy, coord.npy, normal.npy, segment20.npy, ...)
# instead of the bundled sample1.npz, and visualizes prediction vs GT.

import os
import argparse
import numpy as np
import utonia
import torch
import torch.nn as nn
import open3d as o3d

try:
    import flash_attn
except ImportError:
    flash_attn = None

device = "cuda" if torch.cuda.is_available() else "cpu"


# ScanNet Meta data (same as demo/2_sem_seg.py)
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
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

CLASS_COLOR_20 = np.array([SCANNET_COLOR_MAP_20[i] for i in VALID_CLASS_IDS_20])
IGNORE_COLOR = np.array([0.0, 0.0, 0.0])  # for label == -1 (unlabeled)


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


def load_scene_folder(scene_dir: str) -> dict:
    """Load a preprocessed ScanNet-style scene folder into a Utonia point dict.

    Expected files:
        coord.npy      (N, 3) float
        color.npy      (N, 3) float [0-255]
        normal.npy     (N, 3) float
        segment20.npy  (N,)   int   (ScanNet 20-class, -1 = ignore)
        segment200.npy (N,)   int   (ScanNet 200-class, -1 = ignore) - optional
        instance.npy   (N,)   int   - optional
    """
    def _load(name, required=True):
        p = os.path.join(scene_dir, name)
        if os.path.isfile(p):
            return np.load(p)
        if required:
            raise FileNotFoundError(f"Missing required file: {p}")
        return None

    coord = _load("coord.npy").astype(np.float64)
    color = _load("color.npy").astype(np.float64)
    normal = _load("normal.npy").astype(np.float64)

    point = {
        "coord": coord,
        "color": color,
        "normal": normal,
    }

    seg20 = _load("segment20.npy", required=False)
    seg200 = _load("segment200.npy", required=False)
    if seg20 is not None:
        point["segment20"] = seg20.astype(np.int64)
    if seg200 is not None:
        point["segment200"] = seg200.astype(np.int64)

    print(f"Loaded scene: {scene_dir}")
    print(f"  coord:  {coord.shape}  range=[{coord.min():.2f}, {coord.max():.2f}]")
    print(f"  color:  {color.shape}  range=[{color.min():.1f}, {color.max():.1f}]")
    print(f"  normal: {normal.shape}")
    if seg20 is not None:
        print(f"  segment20:  {seg20.shape}  "
              f"unique={len(np.unique(seg20))}  "
              f"valid={(seg20 >= 0).sum()}/{len(seg20)}")
    if seg200 is not None:
        print(f"  segment200: {seg200.shape}")
    return point


def labels_to_rgb(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Convert integer labels to RGB colors. label == -1 -> black."""
    out = np.zeros((labels.shape[0], 3), dtype=np.float64)
    valid = labels >= 0
    out[valid] = palette[labels[valid]]
    out[~valid] = IGNORE_COLOR
    return out


def make_pcd(coord: np.ndarray, rgb: np.ndarray, offset=(0.0, 0.0, 0.0)):
    pcd = o3d.geometry.PointCloud()
    pts = coord + np.asarray(offset, dtype=coord.dtype)
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb / 255.0, 0.0, 1.0))
    return pcd


# --- Matplotlib / Open3D rendering view presets ---------------------------
# Keys are used as filename suffixes; values are (elev, azim) for matplotlib
# and (front, up) vectors for Open3D.
MPL_VIEW_PRESETS = {
    "top":   (90.0,  -90.0),
    "front": (0.0,   -90.0),
    "back":  (0.0,    90.0),
    "left":  (0.0,     0.0),
    "right": (0.0,   180.0),
    "iso":   (30.0,  -45.0),
}

O3D_VIEW_PRESETS = {
    "top":   ([0.0,  0.0, -1.0], [0.0,  1.0, 0.0]),
    "front": ([0.0, -1.0,  0.0], [0.0,  0.0, 1.0]),
    "back":  ([0.0,  1.0,  0.0], [0.0,  0.0, 1.0]),
    "left":  ([1.0,  0.0,  0.0], [0.0,  0.0, 1.0]),
    "right": ([-1.0, 0.0,  0.0], [0.0,  0.0, 1.0]),
    "iso":   ([-0.7, -0.7, -0.7], [0.0, 0.0, 1.0]),
}


def _safe_name(title: str) -> str:
    return title.replace(" ", "_").replace("(", "").replace(")", "")


def render_views_to_png_open3d(
    pcd_list,
    titles,
    out_dir: str,
    width: int = 1280,
    height: int = 960,
    point_size: float = 2.0,
    bg_color=(1.0, 1.0, 1.0),
):
    """Open3D offscreen renderer. Requires OpenGL/X. May fail headless."""
    os.makedirs(out_dir, exist_ok=True)

    def _render(geoms, filename):
        vis = o3d.visualization.Visualizer()
        created = vis.create_window(
            window_name="render", width=width, height=height, visible=False
        )
        if not created:
            raise RuntimeError("Open3D failed to create a rendering window")
        for g in geoms:
            vis.add_geometry(g)
        opt = vis.get_render_option()
        if opt is not None:
            opt.background_color = np.asarray(bg_color)
            opt.point_size = point_size
        for view_name, (front, up) in O3D_VIEW_PRESETS.items():
            ctr = vis.get_view_control()
            ctr.set_front(front)
            ctr.set_up(up)
            ctr.set_lookat(
                np.mean(
                    np.concatenate([np.asarray(g.points) for g in geoms], axis=0),
                    axis=0,
                ).tolist()
            )
            ctr.set_zoom(0.7)
            vis.poll_events()
            vis.update_renderer()
            out_path = os.path.join(out_dir, f"{filename}_{view_name}.png")
            vis.capture_screen_image(out_path, do_render=True)
            print(f"  saved: {out_path}")
        vis.destroy_window()

    for pcd, title in zip(pcd_list, titles):
        pts = np.asarray(pcd.points)
        centered = o3d.geometry.PointCloud()
        centered.points = o3d.utility.Vector3dVector(pts - pts.mean(axis=0))
        centered.colors = pcd.colors
        _render([centered], _safe_name(title))

    _render(list(pcd_list), "all")


def render_views_to_png_matplotlib(
    pcd_list,
    titles,
    out_dir: str,
    width: int = 1280,
    height: int = 960,
    point_size: float = 2.0,
    bg_color=(1.0, 1.0, 1.0),
    max_points: int = 150_000,
):
    """Pure-CPU renderer using matplotlib. No OpenGL / display needed.

    Large point clouds are randomly subsampled to `max_points` to keep
    rendering tractable (matplotlib's 3D scatter is ~O(N)).
    """
    # Force non-interactive backend BEFORE importing pyplot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Registering the 3D projection import is required for side-effects.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    os.makedirs(out_dir, exist_ok=True)
    dpi = 100
    figsize = (width / dpi, height / dpi)

    rng = np.random.default_rng(0)

    def _subsample(pts, cols):
        if pts.shape[0] > max_points:
            idx = rng.choice(pts.shape[0], size=max_points, replace=False)
            return pts[idx], cols[idx]
        return pts, cols

    def _set_equal_aspect(ax, pts):
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        half = (maxs - mins).max() / 2.0 * 1.05
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)

    def _render_single(pts, cols, filename, title):
        pts, cols = _subsample(pts, cols)
        for view_name, (elev, azim) in MPL_VIEW_PRESETS.items():
            fig = plt.figure(figsize=figsize, dpi=dpi)
            fig.patch.set_facecolor(bg_color)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor(bg_color)
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=np.clip(cols, 0.0, 1.0),
                s=point_size, marker=".", linewidths=0, depthshade=False,
            )
            _set_equal_aspect(ax, pts)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            ax.set_title(f"{title} — {view_name}")
            fig.tight_layout(pad=0)
            out_path = os.path.join(out_dir, f"{_safe_name(title)}_{view_name}.png")
            fig.savefig(out_path, dpi=dpi, facecolor=bg_color,
                        bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            print(f"  saved: {out_path}")

    def _render_combined(pcd_list, titles):
        """One figure per view containing all point clouds as subplots."""
        n = len(pcd_list)
        for view_name, (elev, azim) in MPL_VIEW_PRESETS.items():
            fig = plt.figure(figsize=(figsize[0] * n, figsize[1]), dpi=dpi)
            fig.patch.set_facecolor(bg_color)
            for i, (pcd, title) in enumerate(zip(pcd_list, titles)):
                pts = np.asarray(pcd.points)
                cols = np.asarray(pcd.colors)
                pts_c = pts - pts.mean(axis=0)  # center each
                pts_c, cols_c = _subsample(pts_c, cols)
                ax = fig.add_subplot(1, n, i + 1, projection="3d")
                ax.set_facecolor(bg_color)
                ax.scatter(
                    pts_c[:, 0], pts_c[:, 1], pts_c[:, 2],
                    c=np.clip(cols_c, 0.0, 1.0),
                    s=point_size, marker=".", linewidths=0, depthshade=False,
                )
                _set_equal_aspect(ax, pts_c)
                ax.view_init(elev=elev, azim=azim)
                ax.set_axis_off()
                ax.set_title(title, fontsize=10)
            fig.tight_layout(pad=0.5)
            out_path = os.path.join(out_dir, f"all_{view_name}.png")
            fig.savefig(out_path, dpi=dpi, facecolor=bg_color,
                        bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            print(f"  saved: {out_path}")

    for pcd, title in zip(pcd_list, titles):
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors)
        pts_c = pts - pts.mean(axis=0)
        _render_single(pts_c, cols, _safe_name(title), title)

    _render_combined(pcd_list, titles)


def render_views_to_png(
    pcd_list,
    titles,
    out_dir: str,
    backend: str = "auto",
    **kwargs,
):
    """Dispatch to the requested rendering backend.

    backend:
        "open3d"     - Open3D offscreen (needs OpenGL / X / xvfb)
        "matplotlib" - Matplotlib Agg (pure CPU, no display needed)
        "auto"       - try Open3D, fall back to matplotlib on failure
    """
    # open3d backend does not accept `max_points`; strip it when dispatching.
    o3d_kwargs = {k: v for k, v in kwargs.items() if k != "max_points"}

    if backend == "open3d":
        render_views_to_png_open3d(pcd_list, titles, out_dir, **o3d_kwargs)
        return "open3d"
    if backend == "matplotlib":
        render_views_to_png_matplotlib(pcd_list, titles, out_dir, **kwargs)
        return "matplotlib"
    if backend == "auto":
        try:
            render_views_to_png_open3d(pcd_list, titles, out_dir, **o3d_kwargs)
            return "open3d"
        except Exception as e:
            print(f"[render] Open3D backend failed ({e}); "
                  f"falling back to matplotlib.")
            render_views_to_png_matplotlib(pcd_list, titles, out_dir, **kwargs)
            return "matplotlib"
    raise ValueError(f"Unknown backend: {backend}")


def compute_miou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 20):
    """Compute mean IoU and per-class IoU, ignoring gt == -1."""
    valid = gt >= 0
    pred = pred[valid]
    gt = gt[valid]
    ious = []
    for c in range(num_classes):
        p_mask = pred == c
        g_mask = gt == c
        inter = np.logical_and(p_mask, g_mask).sum()
        union = np.logical_or(p_mask, g_mask).sum()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(float(inter) / float(union))
    acc = (pred == gt).mean() if len(gt) > 0 else float("nan")
    return ious, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Utonia semantic segmentation on a preprocessed scene folder"
    )
    parser.add_argument(
        "scene_dir",
        help="Path to scene folder containing coord.npy/color.npy/normal.npy/segment20.npy",
    )
    parser.add_argument(
        "--gt-segment", choices=["segment20", "segment200"], default="segment20",
        help="Which ground-truth segment file to compare against (default: segment20). "
             "Utonia linear-prob head is trained on ScanNet20, so metrics only make "
             "sense for segment20.",
    )
    parser.add_argument(
        "--wo_color", action="store_true", help="disable the color."
    )
    parser.add_argument(
        "--wo_normal", action="store_true", help="disable the normal."
    )
    parser.add_argument(
        "--save-ply-dir", default=None,
        help="If set, save pred/gt/input PLY files to this directory instead of "
             "opening an interactive viewer.",
    )
    parser.add_argument(
        "--save-png-dir", default=None,
        help="If set, render PNG snapshots from multiple viewpoints "
             "(top/front/back/left/right/iso) for input/pred/gt and a combined "
             "side-by-side 'all' view. Can be combined with --save-ply-dir.",
    )
    parser.add_argument(
        "--png-width", type=int, default=1280, help="PNG width (default 1280)"
    )
    parser.add_argument(
        "--png-height", type=int, default=960, help="PNG height (default 960)"
    )
    parser.add_argument(
        "--point-size", type=float, default=2.0,
        help="Point size for PNG rendering (default 2.0)",
    )
    parser.add_argument(
        "--backend", choices=["auto", "open3d", "matplotlib"], default="auto",
        help="PNG rendering backend. 'matplotlib' works on headless servers "
             "with no OpenGL/display; 'open3d' is higher quality but needs "
             "OpenGL; 'auto' tries open3d first and falls back to matplotlib.",
    )
    parser.add_argument(
        "--max-points", type=int, default=150_000,
        help="Max points for matplotlib rendering (subsample above this). "
             "Matplotlib 3D scatter is slow; lower this for faster PNG output. "
             "Ignored for open3d backend. Default 150000.",
    )
    args = parser.parse_args()

    utonia.utils.set_seed(46647087)

    # ---- Load model ----
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

    ckpt = utonia.load(
        "utonia_linear_prob_head_sc",
        repo_id="Pointcept/Utonia",
        ckpt_only=True,
    )
    seg_head = SegHead(**ckpt["config"]).to(device)
    seg_head.load_state_dict(ckpt["state_dict"])

    # ---- Load data ----
    point = load_scene_folder(args.scene_dir)

    if args.wo_color:
        point["color"] = np.zeros_like(point["coord"])
    if args.wo_normal:
        point["normal"] = np.zeros_like(point["coord"])

    # Keep a copy of originals for visualization against GT
    original_coord = point["coord"].copy()
    original_color = point["color"].copy()
    gt_segment_full = point.get(args.gt_segment, None)
    if gt_segment_full is None:
        print(f"Warning: {args.gt_segment}.npy not found; skipping GT comparison.")

    # Utonia transform expects a 'segment' key.  Use segment20 since the
    # linear-probing head is trained on ScanNet20 labels.
    seg20_for_transform = point.pop("segment20", None)
    point.pop("segment200", None)
    if seg20_for_transform is not None:
        point["segment"] = seg20_for_transform

    transform = utonia.transform.default(0.5)
    point = transform(point)

    # ---- Inference ----
    model.eval()
    seg_head.eval()
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor) and device == "cuda":
                point[key] = point[key].cuda(non_blocking=True)

        point = model(point)
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        seg_logits = seg_head(point.feat)
        pred = seg_logits.argmax(dim=-1).data.cpu().numpy()  # (M,)

    # point.coord is the grid-sampled coord (M, 3); the transform also stores
    # `inverse` so we can map predictions back to the original N points.
    coord_sampled = point.coord.detach().cpu().numpy()
    pred_color_sampled = CLASS_COLOR_20[pred]

    if "inverse" in point.keys():
        inverse = point.inverse.detach().cpu().numpy()  # (N,) maps orig -> sampled
        pred_full = pred[inverse]                       # (N,)
        pred_color_full = CLASS_COLOR_20[pred_full]
    else:
        inverse = None
        pred_full = None
        pred_color_full = None

    # ---- Compare against GT (at original resolution if possible) ----
    if gt_segment_full is not None and pred_full is not None:
        ious, acc = compute_miou(pred_full, gt_segment_full, num_classes=20)
        valid_ious = [v for v in ious if not np.isnan(v)]
        miou = float(np.mean(valid_ious)) if valid_ious else float("nan")
        print(f"\n=== Evaluation against {args.gt_segment} (ScanNet20) ===")
        print(f"Overall point accuracy: {acc:.4f}")
        print(f"mIoU:                   {miou:.4f}")
        print("Per-class IoU:")
        for label, iou in zip(CLASS_LABELS_20, ious):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "  n/a "
            print(f"  {label:20s}: {iou_str}")

    # ---- Build visualization geometries ----
    geometries = []

    # 1. Input RGB (original resolution)
    pcd_input = make_pcd(original_coord, original_color, offset=(0.0, 0.0, 0.0))
    geometries.append(pcd_input)

    # Shift along +x by the scene width so the three views sit side by side
    extent_x = float(original_coord[:, 0].max() - original_coord[:, 0].min())
    shift = extent_x * 1.2

    # 2. Prediction (use full-resolution if available, otherwise sampled)
    if pred_color_full is not None:
        pcd_pred = make_pcd(original_coord, pred_color_full, offset=(shift, 0.0, 0.0))
    else:
        pcd_pred = make_pcd(coord_sampled, pred_color_sampled, offset=(shift, 0.0, 0.0))
    geometries.append(pcd_pred)

    # 3. Ground truth (if available)
    if gt_segment_full is not None and args.gt_segment == "segment20":
        gt_rgb = labels_to_rgb(gt_segment_full, CLASS_COLOR_20)
        pcd_gt = make_pcd(original_coord, gt_rgb, offset=(2 * shift, 0.0, 0.0))
        geometries.append(pcd_gt)
    elif gt_segment_full is not None:
        # segment200: we can't map to the 20-class palette, just show it as
        # a randomized palette so the structure is visible.
        rng = np.random.default_rng(0)
        max_lbl = int(gt_segment_full.max()) + 1 if (gt_segment_full >= 0).any() else 1
        palette200 = (rng.random((max_lbl, 3)) * 255.0)
        gt_rgb = labels_to_rgb(gt_segment_full, palette200)
        pcd_gt = make_pcd(original_coord, gt_rgb, offset=(2 * shift, 0.0, 0.0))
        geometries.append(pcd_gt)

    titles = ["input RGB", f"prediction (ScanNet20)"]
    if len(geometries) == 3:
        titles.append(f"GT ({args.gt_segment})")
    print("\nLayout (left -> right):", " | ".join(titles))

    # ---- Save / visualize ----
    did_save = False

    if args.save_ply_dir:
        os.makedirs(args.save_ply_dir, exist_ok=True)
        o3d.io.write_point_cloud(
            os.path.join(args.save_ply_dir, "input_rgb.ply"), pcd_input
        )
        o3d.io.write_point_cloud(
            os.path.join(args.save_ply_dir, "pred_seg20.ply"), pcd_pred
        )
        if len(geometries) == 3:
            o3d.io.write_point_cloud(
                os.path.join(args.save_ply_dir, f"gt_{args.gt_segment}.ply"),
                geometries[2],
            )
        print(f"Saved PLY files to: {args.save_ply_dir}")
        did_save = True

    if args.save_png_dir:
        print(f"Rendering PNG snapshots to: {args.save_png_dir} "
              f"(backend={args.backend})")
        used = render_views_to_png(
            geometries,
            titles,
            out_dir=args.save_png_dir,
            backend=args.backend,
            width=args.png_width,
            height=args.png_height,
            point_size=args.point_size,
            max_points=args.max_points,
        )
        print(f"PNG rendering done (backend used: {used})")
        did_save = True

    if not did_save:
        o3d.visualization.draw_geometries(
            geometries, window_name="Utonia semseg: input | pred | gt"
        )
