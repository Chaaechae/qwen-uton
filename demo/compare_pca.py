# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Side-by-side PCA visualization comparing the published Utonia checkpoint
# against two locally-trained variants (typically v1m3-A and v1m3-B). All
# three models run the same scene through the same transform + upcast
# protocol so the resulting PCA-colored point clouds are directly
# comparable.
#
# Usage:
#   # Interactive (Open3D windows for RGB + each PCA, opened sequentially):
#   python demo/compare_pca.py \
#       --v1m3a exp/utonia_q35_align_only/model/model_last.pth \
#       --v1m3b exp/utonia_q35_align_ssl/model/model_last.pth
#
#   # Single PNG via matplotlib 3-D scatter (headless-friendly):
#   python demo/compare_pca.py \
#       --v1m3a exp/.../model_last.pth \
#       --v1m3b exp/.../model_last.pth \
#       --save compare.png
#
#   # Interactive HTML via plotly (rotate/zoom each panel in a browser):
#   python demo/compare_pca.py \
#       --v1m3a exp/.../model_last.pth \
#       --v1m3b exp/.../model_last.pth \
#       --save compare.html
#
#   # Different sample / no-color / no-normal:
#   python demo/compare_pca.py --sample sample2 --wo_color --v1m3a ... --v1m3b ...
#
# --utonia defaults to /group-volume/Utonia/utonia.pth (cluster default).
# Pass any other local path, or `--utonia ""` to skip that slot.
# HuggingFace-hub download path is NOT used.

import argparse
import copy
import os
import sys

import numpy as np
import torch

# Make the local `utonia` package importable regardless of cwd / PYTHONPATH.
# Existing demo scripts assume `export PYTHONPATH=./` from the repo root;
# this auto-prepends the qwen-uton repo root (= this file's parent dir's
# parent) so `python demo/compare_pca.py` works from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import utonia
from utonia.model import PointTransformerV3

try:
    import flash_attn  # noqa: F401
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False


# Hard-fail early if CUDA isn't available. PTv3 + spconv only run on GPU
# (CPU-build of spconv raises a cryptic "CPU ONLY build" RuntimeError far
# downstream); checking up front gives a clearer error and pinpoints
# whether the issue is environment / node / package selection.
if not torch.cuda.is_available():
    raise SystemExit(
        "[compare_pca] CUDA is not available in this Python environment.\n"
        "  PTv3 + spconv require a GPU. Likely causes:\n"
        "    - Running on a non-GPU node (check nvidia-smi)\n"
        "    - Conda env without CUDA-enabled torch / spconv-cuXXX\n"
        "  Use demo/run_compare_pca.sh (auto-activates the training env)\n"
        "  or activate ~/anaconda3/envs/pointcept manually."
    )
DEVICE = "cuda"
print(f"[setup] CUDA: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.device_count()} device(s), torch={torch.__version__})")


# Default architecture for our v1m3 trained checkpoints. Matches what we use
# in training/configs/utonia/distill-utonia-v1m3-{A,B}-...py.
V1M3_BACKBONE_CONFIG = dict(
    in_channels=9,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(3, 3, 3, 12, 3),
    enc_channels=(54, 108, 216, 432, 576),
    enc_num_head=(3, 6, 12, 24, 32),
    enc_patch_size=(1024,) * 5,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    shuffle_orders=True,
    pre_norm=True,
    enable_rpe=False,
    enable_flash=HAS_FLASH,
    upcast_attention=False,
    upcast_softmax=False,
    enc_mode=True,
    mask_token=True,
    rope_base=10,
    shift_coords=None,
    jitter_coords=1.1,
    rescale_coords=1.2,
)


def _strip_backbone_prefix(state_dict):
    """
    Pointcept training-format ckpts wrap weights as
        module.student.backbone.<param>
    or
        student.backbone.<param>
    depending on whether DDP was used. Strip whichever prefix is present
    and return the resulting PTv3-native state_dict.
    """
    prefixes = ("module.student.backbone.", "student.backbone.")
    out = {}
    for k, v in state_dict.items():
        for p in prefixes:
            if k.startswith(p):
                out[k[len(p):]] = v
                break
    return out


def load_model(ckpt_path, label=""):
    """
    Load a PTv3 checkpoint from a LOCAL file path. Two formats supported:
      (a) Pointcept training format (module.student.backbone.*)
      (b) Published Utonia HF format ({"config": ..., "state_dict": ...})

    No HuggingFace-hub download path — provide --utonia <local_path>.
    """
    if not ckpt_path:
        raise ValueError(
            f"[{label}] No checkpoint path provided. "
            "Pass a local file path (HF download is disabled here)."
        )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # (b) Published HF format.
    if isinstance(ckpt, dict) and "config" in ckpt and "state_dict" in ckpt:
        print(f"[{label}] Loading {ckpt_path} (HF format) ...")
        cfg = dict(ckpt["config"])
        if not HAS_FLASH:
            cfg["enable_flash"] = False
        model = PointTransformerV3(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model

    # (a) Pointcept training format.
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    stripped = _strip_backbone_prefix(state_dict)
    if not stripped:
        raise ValueError(
            f"[{label}] {ckpt_path}: no `*student.backbone.*` keys found. "
            f"Is this a Pointcept training checkpoint of v1m3-A / v1m3-B?"
        )
    print(f"[{label}] Loading {ckpt_path} (Pointcept fmt) — "
          f"{len(stripped)} backbone tensors ...")
    cfg = dict(V1M3_BACKBONE_CONFIG)
    model = PointTransformerV3(**cfg)
    info = model.load_state_dict(stripped, strict=False)
    if info.missing_keys:
        print(f"  missing keys: {len(info.missing_keys)} "
              f"(first 3: {info.missing_keys[:3]})")
    if info.unexpected_keys:
        print(f"  unexpected keys: {len(info.unexpected_keys)} "
              f"(first 3: {info.unexpected_keys[:3]})")
    return model


def get_pca_color(feat, brightness=1.25, center=True):
    """Same PCA mapping as demo/0_pca_indoor.py for consistency."""
    u, s, v = torch.pca_lowrank(feat, center=center, q=6, niter=5)
    projection = feat @ v
    projection = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    return color.clamp(0.0, 1.0)


def forward_pca(model, raw_point, transform):
    """
    Run a single point cloud through `model`, apply the same upcast
    protocol as demo/0_pca_indoor.py (2 cat-upcasts then drain via
    inverse scatter back to original grid), and produce per-point RGB
    from PCA at the ORIGINAL (pre-GridSample) resolution.
    """
    point = transform(copy.deepcopy(raw_point))
    for k in point:
        if isinstance(point[k], torch.Tensor):
            point[k] = point[k].to(DEVICE, non_blocking=True)

    with torch.inference_mode():
        point = model(point)
        # 2 cat-upcasts (matches demo/0_pca_indoor.py)
        for _ in range(2):
            assert "pooling_parent" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat(
                [parent.feat, point.feat[inverse]], dim=-1
            )
            point = parent
        # Drain any remaining levels with bare scatter (no cat).
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent

        pca = get_pca_color(point.feat, brightness=1.2, center=True)

    # Lift back to original (pre-GridSample) resolution.
    return pca[point.inverse].cpu().numpy()


def show_open3d(coord, color_rgb, pcas_by_name):
    """Open one window per (RGB + each PCA). Interactive."""
    import open3d as o3d

    def _show(name, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name=name)

    _show("RGB", color_rgb)
    for name, pca in pcas_by_name.items():
        _show(f"PCA: {name}", pca)


def save_matplotlib(coord, color_rgb, pcas_by_name, out_path, elev=20, azim=-60):
    """Render N+1 panels (RGB + each PCA) to a single PNG via matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    panels = [("RGB", color_rgb)] + list(pcas_by_name.items())
    n = len(panels)
    fig = plt.figure(figsize=(5 * n, 5))
    for i, (title, colors) in enumerate(panels):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2],
                   c=colors, s=0.5, marker=".", linewidths=0)
        ax.set_title(title)
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        # Equal axis ratio.
        ranges = coord.max(axis=0) - coord.min(axis=0)
        center = (coord.max(axis=0) + coord.min(axis=0)) / 2
        r = ranges.max() / 2
        for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
            setter(c - r, c + r)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")


def _rgb_to_str(rgb_float):
    """(N, 3) float in [0,1] → list of "rgb(r,g,b)" strings for plotly."""
    rgb_int = (np.asarray(rgb_float) * 255).clip(0, 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb_int]


def save_plotly(coord, color_rgb, pcas_by_name, out_path, marker_size=1.2):
    """
    Render N+1 panels (RGB + each PCA) as an interactive HTML via plotly.
    Cameras across subplots are kept independent (rotate one without
    moving the others). Use share_camera=True to sync them if desired.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panels = [("RGB", color_rgb)] + list(pcas_by_name.items())
    n = len(panels)

    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "scene"}] * n],
        subplot_titles=[t for t, _ in panels],
        horizontal_spacing=0.02,
    )

    for i, (_title, colors) in enumerate(panels):
        fig.add_trace(
            go.Scatter3d(
                x=coord[:, 0], y=coord[:, 1], z=coord[:, 2],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=_rgb_to_str(colors),
                    opacity=1.0,
                ),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=i + 1,
        )

    # Tight, equal-aspect scenes; hide axes.
    scene_layout = dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
    )
    layout_updates = {f"scene{(i+1) if i else ''}": scene_layout for i in range(n)}
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        width=400 * n,
        **layout_updates,
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--utonia", default="/group-volume/Utonia/utonia.pth",
        help="Path to local utonia.pth (cluster default shown). "
             "Set to empty string '' to skip this slot.",
    )
    parser.add_argument(
        "--v1m3a", default=None,
        help="Path to v1m3-A trained checkpoint (model_last.pth).",
    )
    parser.add_argument(
        "--v1m3b", default=None,
        help="Path to v1m3-B trained checkpoint (model_last.pth).",
    )
    parser.add_argument(
        "--sample", default="sample1",
        help="Built-in sample name (sample1/2/...). See utonia.data.",
    )
    parser.add_argument(
        "--scale", type=float, default=0.5,
        help="Scale for the default transform (smaller = coarser).",
    )
    parser.add_argument(
        "--save", default=None,
        help="Save side-by-side PNG (matplotlib). If omitted, opens Open3D windows.",
    )
    parser.add_argument("--wo_color", action="store_true",
                        help="Zero out input color before encoding.")
    parser.add_argument("--wo_normal", action="store_true",
                        help="Zero out input normal before encoding.")
    args = parser.parse_args()

    utonia.utils.set_seed(37)

    # Load all requested models. Skip a slot if its path is empty or missing.
    models = {}
    if args.utonia and os.path.isfile(args.utonia):
        models["utonia.pth"] = load_model(args.utonia, "utonia.pth")
    elif args.utonia:
        print(f"[skip] --utonia path not found: {args.utonia}")
    if args.v1m3a:
        models["v1m3-A"] = load_model(args.v1m3a, "v1m3-A")
    if args.v1m3b:
        models["v1m3-B"] = load_model(args.v1m3b, "v1m3-B")
    if not models:
        raise SystemExit(
            "No models loaded. Provide at least one of "
            "--utonia / --v1m3a / --v1m3b (local file paths)."
        )
    for m in models.values():
        m.to(DEVICE).eval()

    # Load sample once and prep coord/color for visualization.
    raw_point = utonia.data.load(args.sample)
    if args.wo_color:
        raw_point["color"] = np.zeros_like(raw_point["coord"])
    if args.wo_normal:
        raw_point["normal"] = np.zeros_like(raw_point["coord"])
    raw_point.pop("segment200", None)
    if "segment20" in raw_point:
        raw_point["segment"] = raw_point.pop("segment20")
    original_coord = raw_point["coord"].copy()
    original_color = raw_point["color"].copy() / 255.0

    transform = utonia.transform.default(args.scale)

    pcas = {}
    for name, m in models.items():
        print(f"--- forward: {name}")
        pcas[name] = forward_pca(m, raw_point, transform)

    if args.save:
        ext = os.path.splitext(args.save)[1].lower()
        if ext in (".html", ".htm"):
            save_plotly(original_coord, original_color, pcas, args.save)
        else:
            save_matplotlib(original_coord, original_color, pcas, args.save)
    else:
        show_open3d(original_coord, original_color, pcas)


if __name__ == "__main__":
    main()
