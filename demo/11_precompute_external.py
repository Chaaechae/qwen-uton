# Precompute Utonia per-patch features from an externally extracted pseudo
# point cloud (e.g. 32 frames x 14x14 patch-center samples) and save as .pkl.
#
# This mirrors Video-3D-LLM's
#   scripts/3d/preprocessing/precompute_utonia_patch_features.py
# but skips the frame-wise depth unprojection / normal estimation / patch
# center sampling — the caller has already produced a (T, P, 10) array with
# columns [x, y, z, r, g, b, nx, ny, nz, confidence].
#
# Key design note (matching the reference):
#   - invalid points are NOT deleted; they stay in the (T, P, ...) layout
#     as zero-filled slots with patch_mask=False.
#   - only the VALID points are fed through Utonia as one pseudo cloud,
#     and the output features are scattered back into (T, P, D) so the
#     frame / patch structure is preserved for downstream loaders.
#
# Supported input formats (one scene per file):
#   - .npy:  (T, H, W, 10), (T, P, 10), or (N, 10).
#            The (N, 10) case requires --frame-shape to recover (T, P) or
#            (T, H, W); otherwise every point is treated as a single frame.
#   - .npz:  either a single (T, *, 10) / (N, 10) array under key "data" /
#            "arr_0", or separate keys
#            {coord, color, normal, confidence} with matching leading dims.
#
# Output per scene (.pkl) — same field names as the reference patch-feature
# precompute so downstream loaders need no changes:
#   frame_ids:     list[str]          — ["0000", "0001", ...] (sequential)
#   patch_coords:  (T, P, 3) float32  — world coords, 0 at invalid slots
#   patch_colors:  (T, P, 3) uint8    — 0 at invalid slots
#   patch_normals: (T, P, 3) float32  — 0 at invalid slots
#   patch_feat:    (T, P, D) float32  — Utonia feature, 0 at invalid slots
#   patch_mask:    (T, P)    bool     — True iff the slot is valid
#   confidence:    (T, P)    float32  — raw per-patch confidence (optional)
#   grid_size:     (Ph, Pw) | None    — patch grid, None if P is not a square
#   feat_dim:      int                — D
#
# Usage:
#   python demo/11_precompute_external.py scene.npy -o scene.pkl
#   python demo/11_precompute_external.py scenes_dir/ -o out_dir/ \
#       --conf-threshold 0.5
#   python demo/11_precompute_external.py flat.npy --frame-shape 32,14,14 \
#       --conf-threshold 50 --conf-percentile

import os
import argparse
import glob
import pickle
import numpy as np
import torch
from tqdm import tqdm

import utonia


# ---------------------------------------------------------------------------
# Input loading & shape handling
# ---------------------------------------------------------------------------
def parse_frame_shape(s):
    """Parse '--frame-shape' like '32,14,14' or '32,196' -> tuple of ints."""
    if s is None:
        return None
    parts = [int(x) for x in s.split(",") if x.strip()]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "--frame-shape must be 'T,P' or 'T,H,W'"
        )
    return tuple(parts)


def infer_layout(arr, frame_shape_override=None):
    """Return (T, P, Ph, Pw) for an input whose last axis is the channel axis.

    Priority: frame_shape_override > arr.shape.
    Ph/Pw are None unless P can be expressed as H*W with both known
    (either given as (T,H,W) or P is a perfect square).
    """
    if frame_shape_override is not None:
        fs = frame_shape_override
        if len(fs) == 3:
            T, H, W = fs
            expected = T * H * W
            if arr.shape[0] != expected and arr.size // arr.shape[-1] != expected:
                raise ValueError(
                    f"frame-shape {fs} expects {expected} points, got array {arr.shape}"
                )
            return T, H * W, H, W
        # len(fs) == 2
        T, P = fs
        expected = T * P
        if arr.shape[0] != expected and arr.size // arr.shape[-1] != expected:
            raise ValueError(
                f"frame-shape {fs} expects {expected} points, got array {arr.shape}"
            )
        root = int(round(P ** 0.5))
        if root * root == P:
            return T, P, root, root
        return T, P, None, None

    # No override — use array shape.
    if arr.ndim == 4:              # (T, H, W, C)
        T, H, W, _ = arr.shape
        return T, H * W, H, W
    if arr.ndim == 3:              # (T, P, C)
        T, P, _ = arr.shape
        root = int(round(P ** 0.5))
        if root * root == P:
            return T, P, root, root
        return T, P, None, None
    if arr.ndim == 2:              # (N, C) -> treat as a single frame
        N, _ = arr.shape
        root = int(round(N ** 0.5))
        if root * root == N:
            return 1, N, root, root
        return 1, N, None, None
    raise ValueError(f"Unsupported array ndim: {arr.ndim} (shape={arr.shape})")


def load_external_pcd(path, frame_shape_override=None):
    """Return dict with (T, P, ...) tensors plus Ph, Pw.

    Keys:
      coord:      (T, P, 3) float32
      color:      (T, P, 3) raw (dtype preserved until color normalization)
      normal:     (T, P, 3) float32 (zeros if missing)
      confidence: (T, P)    float32 or None
      Ph, Pw:     int or None
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        raw = np.load(path)
        T, P, Ph, Pw = infer_layout(raw, frame_shape_override)
        if raw.shape[-1] < 9:
            raise ValueError(
                f"{path}: expected >=9 channels [x,y,z,r,g,b,nx,ny,nz] "
                f"(+ optional confidence), got {raw.shape}"
            )
        flat = raw.reshape(T * P, raw.shape[-1])
        coord = flat[:, 0:3].astype(np.float32).reshape(T, P, 3)
        color = flat[:, 3:6].reshape(T, P, 3)
        normal = flat[:, 6:9].astype(np.float32).reshape(T, P, 3)
        confidence = None
        if flat.shape[1] >= 10:
            confidence = flat[:, 9].astype(np.float32).reshape(T, P)
        return dict(coord=coord, color=color, normal=normal,
                    confidence=confidence, T=T, P=P, Ph=Ph, Pw=Pw)

    if ext == ".npz":
        data = dict(np.load(path))
        # Case A: a single (..., 10) array under "data" / "arr_0".
        for key in ("data", "arr_0"):
            if key in data and data[key].shape[-1] >= 9:
                raw = data[key]
                T, P, Ph, Pw = infer_layout(raw, frame_shape_override)
                flat = raw.reshape(T * P, raw.shape[-1])
                coord = flat[:, 0:3].astype(np.float32).reshape(T, P, 3)
                color = flat[:, 3:6].reshape(T, P, 3)
                normal = flat[:, 6:9].astype(np.float32).reshape(T, P, 3)
                confidence = None
                if flat.shape[1] >= 10:
                    confidence = flat[:, 9].astype(np.float32).reshape(T, P)
                return dict(coord=coord, color=color, normal=normal,
                            confidence=confidence, T=T, P=P, Ph=Ph, Pw=Pw)

        # Case B: separate keys.
        coord = np.asarray(data["coord"])
        T, P, Ph, Pw = infer_layout(coord, frame_shape_override)
        coord = coord.astype(np.float32).reshape(T, P, 3)
        color = np.asarray(data["color"]).reshape(T, P, -1) if "color" in data \
            else np.zeros((T, P, 3), dtype=np.uint8)
        if "normal" in data:
            normal = np.asarray(data["normal"]).astype(np.float32).reshape(T, P, 3)
        else:
            normal = np.zeros((T, P, 3), dtype=np.float32)
        confidence = None
        if "confidence" in data:
            confidence = np.asarray(data["confidence"]).astype(np.float32).reshape(T, P)
        return dict(coord=coord, color=color, normal=normal,
                    confidence=confidence, T=T, P=P, Ph=Ph, Pw=Pw)

    raise ValueError(f"Unsupported input format: {ext}")


def normalize_color_to_uint8(color, color_scale):
    """Coerce color to uint8 in [0, 255]."""
    arr = np.asarray(color)
    if color_scale == "auto":
        scale = 255.0 if arr.max() <= 1.0 + 1e-5 else 1.0
    else:
        scale = 255.0 / float(color_scale)
    arr = arr.astype(np.float32) * scale
    return np.clip(arr, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Validity mask
# ---------------------------------------------------------------------------
def build_valid_mask(coord, confidence, conf_threshold, conf_percentile):
    """Build a (T, P) bool mask.

    Rules (match the reference's patch_mask semantics — 'True iff usable'):
      - If confidence is provided and --conf-threshold > 0:
          * absolute mode: mask = confidence >= threshold
          * percentile mode: threshold is computed from confidence quantile
      - Additionally always reject all-zero coords (placeholder slots).
    """
    T, P, _ = coord.shape
    mask = np.ones((T, P), dtype=bool)

    nonzero_coord = np.any(coord != 0, axis=-1)
    mask &= nonzero_coord

    applied_thr = None
    if confidence is not None and conf_threshold > 0:
        if conf_percentile:
            applied_thr = float(np.percentile(confidence, conf_threshold))
        else:
            applied_thr = float(conf_threshold)
        mask &= (confidence >= applied_thr)

    return mask, applied_thr


# ---------------------------------------------------------------------------
# Utonia transform + encoder (no CenterShift → world coords preserved)
# ---------------------------------------------------------------------------
def build_utonia_transform(grid_size):
    from utonia.transform import Compose
    config = [
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True,
        ),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "color", "inverse"),
            feat_keys=("coord", "color", "normal"),
        ),
    ]
    return Compose(config)


def run_utonia_pseudo_cloud(model, transform, coord, color, normal, device):
    """Encode a valid-only pseudo cloud, return (N_valid, D) features."""
    point_dict = {
        "coord": coord.astype(np.float32),
        "color": color.astype(np.uint8),
        "normal": normal.astype(np.float32),
    }
    point_dict = transform(point_dict)
    inverse = point_dict["inverse"].clone()

    for key in list(point_dict.keys()):
        if isinstance(point_dict[key], torch.Tensor):
            point_dict[key] = point_dict[key].to(device)

    if "batch" not in point_dict or point_dict["batch"] is None:
        point_dict["batch"] = torch.zeros(
            point_dict["coord"].shape[0], dtype=torch.long, device=device
        )
    if "offset" not in point_dict or point_dict["offset"] is None:
        point_dict["offset"] = torch.tensor(
            [point_dict["coord"].shape[0]], dtype=torch.long, device=device
        )

    point = utonia.structure.Point(point_dict)
    with torch.inference_mode():
        point = model(point)
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inv = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inv]], dim=-1)
            point = parent

    grid_feat = point.feat.cpu().numpy()             # (N_grid, D)
    inverse_np = inverse.cpu().numpy()               # (N_valid,) → grid idx
    return grid_feat[inverse_np].astype(np.float32)  # (N_valid, D)


# ---------------------------------------------------------------------------
# Per-scene pipeline
# ---------------------------------------------------------------------------
def process_one(input_path, output_path, model, transform, device, args):
    parsed = load_external_pcd(input_path, args.frame_shape)
    T, P, Ph, Pw = parsed["T"], parsed["P"], parsed["Ph"], parsed["Pw"]
    coord = parsed["coord"]           # (T, P, 3) float32
    color_raw = parsed["color"]       # (T, P, 3)
    normal = parsed["normal"]         # (T, P, 3) float32
    confidence = parsed["confidence"] # (T, P) float32 | None

    # Normalize color to uint8 (matches NormalizeColor inside the transform).
    color_u8 = normalize_color_to_uint8(color_raw, args.color_scale)
    assert color_u8.shape == (T, P, 3), color_u8.shape

    print(f"  layout: T={T}, P={P}"
          + (f", grid=({Ph}x{Pw})" if Ph is not None else "")
          + (f", confidence=yes" if confidence is not None else ""))

    # ---- Validity mask (NOT a filter — keeps (T, P) layout intact) ----
    mask, applied_thr = build_valid_mask(
        coord, confidence,
        conf_threshold=args.conf_threshold,
        conf_percentile=args.conf_percentile,
    )
    n_valid = int(mask.sum())
    print(f"  valid: {n_valid}/{T * P}"
          + (f" (conf thr={applied_thr:.4f}"
             f"{' [pct]' if args.conf_percentile else ''})"
             if applied_thr is not None else ""))

    if n_valid == 0:
        print("  WARNING: no valid points; writing zero-filled output")

    # ---- Gather the VALID pseudo cloud ----
    flat_coord = coord.reshape(T * P, 3)
    flat_color = color_u8.reshape(T * P, 3)
    flat_normal = normal.reshape(T * P, 3)
    flat_mask = mask.reshape(T * P)
    valid_idx = np.where(flat_mask)[0]

    # ---- Run Utonia on the pseudo cloud only ----
    if n_valid > 0:
        span = flat_coord[valid_idx].max(axis=0) - flat_coord[valid_idx].min(axis=0)
        print(f"  coord span (valid): {span}")
        per_valid_feat = run_utonia_pseudo_cloud(
            model, transform,
            flat_coord[valid_idx],
            flat_color[valid_idx],
            flat_normal[valid_idx],
            device,
        )
        D = int(per_valid_feat.shape[1])
    else:
        # Fall back to a sentinel D so the file still loads. Use a 1-D dummy
        # forward on a single zero point to discover the true D, so downstream
        # code that concatenates features across scenes still aligns.
        D = _probe_feat_dim(model, transform, device)
        per_valid_feat = np.zeros((0, D), dtype=np.float32)

    # ---- Scatter features back into (T*P, D), zeros at invalid slots ----
    patch_feat = np.zeros((T * P, D), dtype=np.float32)
    if n_valid > 0:
        patch_feat[valid_idx] = per_valid_feat
    patch_feat = patch_feat.reshape(T, P, D)

    # Zero out coord/color/normal at invalid slots (match the reference, makes
    # downstream bugs obvious if patch_mask is accidentally ignored).
    coord_out = coord.copy()
    color_out = color_u8.copy()
    normal_out = normal.copy()
    coord_out[~mask] = 0.0
    color_out[~mask] = 0
    normal_out[~mask] = 0.0

    # Sequential frame ids since we don't have real filenames here.
    frame_ids = [f"{i:04d}" for i in range(T)]

    out = {
        "frame_ids":     frame_ids,
        "patch_coords":  coord_out.astype(np.float32),
        "patch_colors":  color_out.astype(np.uint8),
        "patch_normals": normal_out.astype(np.float32),
        "patch_feat":    patch_feat,
        "patch_mask":    mask.astype(bool),
        "grid_size":     (Ph, Pw) if Ph is not None else None,
        "feat_dim":      D,
    }
    if confidence is not None:
        out["confidence"] = confidence.astype(np.float32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"    saved: {output_path}")
    print(f"      patch_coords{out['patch_coords'].shape}, "
          f"patch_feat{out['patch_feat'].shape}, "
          f"patch_mask{out['patch_mask'].shape} ({n_valid} valid), "
          f"feat_dim={D}")


def _probe_feat_dim(model, transform, device):
    """Run a tiny dummy forward to discover D when a scene has 0 valid points."""
    dummy_coord = np.zeros((8, 3), dtype=np.float32)
    dummy_coord[:, 0] = np.linspace(0, 0.1, 8)  # non-degenerate
    dummy_color = np.zeros((8, 3), dtype=np.uint8)
    dummy_normal = np.tile(np.array([0, 0, 1], dtype=np.float32), (8, 1))
    feat = run_utonia_pseudo_cloud(
        model, transform, dummy_coord, dummy_color, dummy_normal, device
    )
    return int(feat.shape[1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def discover_inputs(path):
    if os.path.isfile(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return [(path, stem)]
    if os.path.isdir(path):
        files = sorted(
            glob.glob(os.path.join(path, "*.npy"))
            + glob.glob(os.path.join(path, "*.npz"))
        )
        return [(f, os.path.splitext(os.path.basename(f))[0]) for f in files]
    raise FileNotFoundError(f"Input not found: {path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Utonia model (grid_size={args.grid_size})...")
    custom_config = dict(enable_flash=False)
    model = utonia.load(
        name="utonia", repo_id="Pointcept/Utonia", custom_config=custom_config,
    ).to(device)
    model.eval()

    transform = build_utonia_transform(args.grid_size)

    inputs = discover_inputs(args.input)
    if not inputs:
        raise RuntimeError(f"No .npy/.npz files found under {args.input}")

    if os.path.isdir(args.input) or len(inputs) > 1:
        out_dir = args.output or (args.input.rstrip("/") + "_utonia_feat")
        os.makedirs(out_dir, exist_ok=True)
        def resolve_out(sid): return os.path.join(out_dir, f"{sid}.pkl")
    else:
        if args.output is None:
            out_default = os.path.splitext(args.input)[0] + "_utonia_feat.pkl"
            def resolve_out(sid): return out_default
        elif args.output.endswith(".pkl"):
            def resolve_out(sid): return args.output
        else:
            os.makedirs(args.output, exist_ok=True)
            def resolve_out(sid): return os.path.join(args.output, f"{sid}.pkl")

    print(f"Found {len(inputs)} input(s)")

    for input_path, scene_id in tqdm(inputs, desc="Encoding"):
        out_path = resolve_out(scene_id)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  skip (exists): {out_path}")
            continue
        print(f"\n[{scene_id}] {input_path}")
        try:
            process_one(input_path, out_path, model, transform, device, args)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute Utonia patch features from an externally "
                    "extracted (T, P, 10) point cloud and save as .pkl "
                    "(matches precompute_utonia_patch_features.py layout)."
    )
    parser.add_argument(
        "input",
        help="Input .npy / .npz file, or directory containing many of them. "
             "Each array has >=9 channels [x,y,z,r,g,b,nx,ny,nz] with an "
             "optional 10th confidence channel. Supported shapes: "
             "(T, H, W, C), (T, P, C), (N, C).",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output .pkl (single) or directory (batch). Defaults to "
             "<input_stem>_utonia_feat.pkl or <input>_utonia_feat/.",
    )
    parser.add_argument(
        "--frame-shape", type=parse_frame_shape, default=None,
        help="Override frame layout for flat (N, C) input, e.g. '32,14,14' "
             "or '32,196'. Ignored when the array already has the dims.",
    )
    parser.add_argument(
        "--grid-size", type=float, default=0.02,
        help="Utonia GridSample voxel size in input world units (default 0.02).",
    )
    parser.add_argument(
        "--color-scale", default="255",
        help="'255' = color is in [0,255] (default); '1' = [0,1] float; "
             "'auto' = decide by max value.",
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=0.0,
        help="Confidence threshold that controls patch_mask (not a filter — "
             "the (T, P) layout is preserved and invalid slots stay zero). "
             "0 disables confidence masking. Absolute by default.",
    )
    parser.add_argument(
        "--conf-percentile", action="store_true",
        help="Treat --conf-threshold as a percentile (0-100) over the "
             "per-scene confidence array rather than an absolute value.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing .pkl outputs (default: skip).",
    )
    args = parser.parse_args()

    if args.color_scale != "auto":
        try:
            args.color_scale = float(args.color_scale)
        except ValueError:
            parser.error("--color-scale must be a number or 'auto'")

    main(args)
