"""
Side-by-side visualization of ScanNet segmentation predictions from
two probe checkpoints (e.g. v1m3-H probe vs published utonia.pth probe).

What this assumes
-----------------
You have already run Pointcept's `tools/test.py` (or the SemSegTester
that the linear-probe configs invoke at end of training) for BOTH
probe checkpoints, so each probe's save_path now contains a
`{scene_name}_pred.npy` file with per-point class labels for every
val scene.

  exp/probe_H/result/scene0011_00_pred.npy
  exp/probe_utonia/result/scene0011_00_pred.npy
  exp/probe_H/result/scene0046_00_pred.npy
  ...

This script does NOT run any forward pass; it only consumes the
already-saved predictions plus the source ScanNet scene data.

What it produces
----------------
Per scene, four PLY point-clouds you can open in MeshLab /
CloudCompare / Blender / open3d:

  <scene>_rgb.ply      — original colors (for orientation)
  <scene>_gt.ply       — ground-truth ScanNet20 colors
  <scene>_<labA>.ply   — probe A prediction (e.g. "H")
  <scene>_<labB>.ply   — probe B prediction (e.g. "utonia")
  <scene>_diff.ply     — agreement map.  Green = both match GT.
                         Yellow = only A matches GT.  Cyan = only B
                         matches GT.  Red = both wrong.  Grey = GT
                         unlabeled (ignore_index).

ScanNet20 palette is the same one Pointcept's tools use so colors
match any other Pointcept visualization in the literature.

Usage
-----
  python tools/compare_seg_predictions.py \
      --scene-data-root /group-volume/3Ddataset/data/scannet/val \
      --pred-a-dir exp/probe_H/result \
      --pred-b-dir exp/probe_utonia/result \
      --label-a H \
      --label-b utonia \
      --out-dir exp/vis_compare \
      --scenes scene0011_00 scene0046_00 scene0207_00
      # or omit --scenes to render every scene with both preds available
"""

import argparse
import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# ScanNet20 palette — pulled from Pointcept's preprocessing/meta_data so the
# colors match any other ScanNet visualization in the literature.  The map is
# nominally indexed by raw ScanNet class id, but our probe outputs the dense
# [0, 19] index space (ScanNetDataset maps via class2id = VALID_CLASS_IDS_20).
# So we build a dense [0, 19] palette here.
# ---------------------------------------------------------------------------
SCANNET20_PALETTE = np.array(
    [
        [174, 199, 232],  #  0 wall
        [152, 223, 138],  #  1 floor
        [ 31, 119, 180],  #  2 cabinet
        [255, 187, 120],  #  3 bed
        [188, 189,  34],  #  4 chair
        [140,  86,  75],  #  5 sofa
        [255, 152, 150],  #  6 table
        [214,  39,  40],  #  7 door
        [197, 176, 213],  #  8 window
        [148, 103, 189],  #  9 bookshelf
        [196, 156, 148],  # 10 picture
        [ 23, 190, 207],  # 11 counter
        [247, 182, 210],  # 12 desk
        [219, 219, 141],  # 13 curtain
        [255, 127,  14],  # 14 refrigerator
        [158, 218, 229],  # 15 shower curtain
        [ 44, 160,  44],  # 16 toilet
        [112, 128, 144],  # 17 sink
        [227, 119, 194],  # 18 bathtub
        [ 82,  84, 163],  # 19 otherfurniture
    ],
    dtype=np.uint8,
)
IGNORE_COLOR = np.array([80, 80, 80], dtype=np.uint8)   # grey for unlabeled
IGNORE_INDEX = -1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-data-root", required=True,
                   help="Pointcept-preprocessed ScanNet val dir, "
                        "e.g. /group-volume/3Ddataset/data/scannet/val")
    p.add_argument("--pred-a-dir", required=True,
                   help="dir containing {scene}_pred.npy from probe A")
    p.add_argument("--pred-b-dir", required=True,
                   help="dir containing {scene}_pred.npy from probe B")
    p.add_argument("--label-a", default="A",
                   help="short tag for probe A, used in output filenames")
    p.add_argument("--label-b", default="B",
                   help="short tag for probe B")
    p.add_argument("--scenes", nargs="+", default=None,
                   help="scene names to render. Omit to render every scene "
                        "that has both predictions available.")
    p.add_argument("--max-scenes", type=int, default=10,
                   help="cap on # scenes when --scenes is not given "
                        "(rendering is fast but the dir grows). "
                        "Set 0 for unlimited.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-diff", action="store_true",
                   help="skip the diff PLY (saves time / disk).")
    p.add_argument("--no-rgb", action="store_true",
                   help="skip the rgb PLY (saves time / disk).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# PLY writer — minimal, no open3d / trimesh dependency.  Pure ASCII or
# binary_little_endian; we use binary for size since point clouds get big.
# ---------------------------------------------------------------------------
def write_ply(path, coord, color):
    """coord: (N, 3) float32. color: (N, 3) uint8."""
    coord = np.asarray(coord, dtype=np.float32)
    color = np.asarray(color, dtype=np.uint8)
    assert coord.ndim == 2 and coord.shape[1] == 3
    assert color.shape == coord.shape
    n = coord.shape[0]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    rec = np.empty(
        n,
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    rec["x"], rec["y"], rec["z"] = coord[:, 0], coord[:, 1], coord[:, 2]
    rec["r"], rec["g"], rec["b"] = color[:, 0], color[:, 1], color[:, 2]

    with open(path, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())


# ---------------------------------------------------------------------------
# Label → color
# ---------------------------------------------------------------------------
def labels_to_colors(labels):
    """labels: int array, values in [0, 19] or -1 (ignore). -> (N, 3) uint8."""
    labels = np.asarray(labels, dtype=np.int64)
    out = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    ignore = labels < 0
    valid = ~ignore
    # Clamp into the palette range to be safe against the rare out-of-bound
    # value (e.g. predictions might return a value that wasn't in training).
    idx = np.clip(labels[valid], 0, len(SCANNET20_PALETTE) - 1)
    out[valid] = SCANNET20_PALETTE[idx]
    out[ignore] = IGNORE_COLOR
    return out


def diff_colors(gt, pred_a, pred_b):
    """
    Per-point agreement RGB:
      grey   = GT ignored
      green  = both A and B agree with GT
      yellow = only A agrees with GT
      cyan   = only B agrees with GT
      red    = neither agrees with GT
    """
    n = gt.shape[0]
    out = np.zeros((n, 3), dtype=np.uint8)
    ignore = gt < 0
    a_ok = (pred_a == gt) & ~ignore
    b_ok = (pred_b == gt) & ~ignore
    both = a_ok & b_ok
    only_a = a_ok & ~b_ok
    only_b = b_ok & ~a_ok
    neither = ~a_ok & ~b_ok & ~ignore
    out[ignore] = IGNORE_COLOR
    out[both]    = (60, 200,  60)   # green
    out[only_a]  = (220, 200,  40)  # yellow
    out[only_b]  = ( 50, 180, 220)  # cyan
    out[neither] = (210,  40,  40)  # red
    return out


# ---------------------------------------------------------------------------
# Scene I/O
# ---------------------------------------------------------------------------
def load_scene(scene_dir):
    """
    Pointcept-preprocessed ScanNet scene dir.  Expects coord.npy, color.npy,
    segment20.npy (or segment200.npy — we only render 20 here).
    """
    coord_path = os.path.join(scene_dir, "coord.npy")
    color_path = os.path.join(scene_dir, "color.npy")
    seg20_path = os.path.join(scene_dir, "segment20.npy")
    seg200_path = os.path.join(scene_dir, "segment200.npy")

    if not os.path.isfile(coord_path):
        raise FileNotFoundError(f"missing {coord_path}")
    coord = np.load(coord_path).astype(np.float32)
    color = (
        np.load(color_path).astype(np.uint8)
        if os.path.isfile(color_path) else
        np.full((coord.shape[0], 3), 200, dtype=np.uint8)
    )
    if os.path.isfile(seg20_path):
        gt = np.load(seg20_path).reshape(-1).astype(np.int64)
    elif os.path.isfile(seg200_path):
        # If we only have 200-class GT, downstream comparison won't be
        # meaningful — skip this scene.
        raise RuntimeError(
            f"{scene_dir} only has segment200.npy; expected segment20.npy "
            "for ScanNet20 comparison."
        )
    else:
        gt = np.full(coord.shape[0], IGNORE_INDEX, dtype=np.int64)
    return coord, color, gt


def discover_scenes(scene_root, pred_a_dir, pred_b_dir, requested):
    """Return sorted list of scene names with all 3 sources present."""
    have_a = {
        f[:-len("_pred.npy")]
        for f in os.listdir(pred_a_dir) if f.endswith("_pred.npy")
    }
    have_b = {
        f[:-len("_pred.npy")]
        for f in os.listdir(pred_b_dir) if f.endswith("_pred.npy")
    }
    on_disk = {d for d in os.listdir(scene_root)
               if os.path.isdir(os.path.join(scene_root, d))}
    common = sorted(have_a & have_b & on_disk)
    if requested:
        missing = [s for s in requested if s not in common]
        if missing:
            print(f"[warn] requested scenes missing from intersection: "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        common = [s for s in requested if s in common]
    return common


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def render_one(scene_name, args):
    scene_dir = os.path.join(args.scene_data_root, scene_name)
    coord, color, gt = load_scene(scene_dir)

    pred_a = np.load(os.path.join(args.pred_a_dir, f"{scene_name}_pred.npy"))
    pred_b = np.load(os.path.join(args.pred_b_dir, f"{scene_name}_pred.npy"))
    pred_a = pred_a.reshape(-1).astype(np.int64)
    pred_b = pred_b.reshape(-1).astype(np.int64)

    n = coord.shape[0]
    if pred_a.shape[0] != n or pred_b.shape[0] != n or gt.shape[0] != n:
        # When predictions came from a voxelized loader and weren't unpooled
        # back to the original points, the lengths will not match.  Bail out
        # with a clear message rather than silently re-indexing.
        print(f"  [{scene_name}] length mismatch — "
              f"coord={n} gt={gt.shape[0]} pred_a={pred_a.shape[0]} "
              f"pred_b={pred_b.shape[0]} — skipping.")
        return False

    out_dir = os.path.join(args.out_dir, scene_name)
    os.makedirs(out_dir, exist_ok=True)

    if not args.no_rgb:
        write_ply(os.path.join(out_dir, f"{scene_name}_rgb.ply"), coord, color)
    write_ply(os.path.join(out_dir, f"{scene_name}_gt.ply"),
              coord, labels_to_colors(gt))
    write_ply(os.path.join(out_dir, f"{scene_name}_{args.label_a}.ply"),
              coord, labels_to_colors(pred_a))
    write_ply(os.path.join(out_dir, f"{scene_name}_{args.label_b}.ply"),
              coord, labels_to_colors(pred_b))
    if not args.no_diff:
        write_ply(os.path.join(out_dir, f"{scene_name}_diff.ply"),
                  coord, diff_colors(gt, pred_a, pred_b))

    # Cheap per-scene mIoU print so we know which scenes are hardest.
    def per_scene_acc(pred):
        mask = gt >= 0
        if not mask.any():
            return float("nan")
        return float((pred[mask] == gt[mask]).mean())
    acc_a = per_scene_acc(pred_a)
    acc_b = per_scene_acc(pred_b)
    print(f"  [{scene_name}] N={n}  "
          f"acc({args.label_a})={acc_a:.3f}  "
          f"acc({args.label_b})={acc_b:.3f}  "
          f"Δ={(acc_a - acc_b):+.3f}")
    return True


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isdir(args.scene_data_root):
        sys.exit(f"[error] scene-data-root not found: {args.scene_data_root}")
    if not os.path.isdir(args.pred_a_dir):
        sys.exit(f"[error] pred-a-dir not found: {args.pred_a_dir}")
    if not os.path.isdir(args.pred_b_dir):
        sys.exit(f"[error] pred-b-dir not found: {args.pred_b_dir}")

    scenes = discover_scenes(
        args.scene_data_root, args.pred_a_dir, args.pred_b_dir,
        args.scenes,
    )
    if not scenes:
        sys.exit("[error] no scenes have predictions from BOTH probes.")
    if args.max_scenes and len(scenes) > args.max_scenes and not args.scenes:
        scenes = scenes[: args.max_scenes]
        print(f"[info] capping to first {args.max_scenes} scenes "
              "(--max-scenes to change, 0 = unlimited)")

    print(f"[info] rendering {len(scenes)} scenes -> {args.out_dir}")
    ok = 0
    for s in scenes:
        try:
            if render_one(s, args):
                ok += 1
        except Exception as e:
            print(f"  [{s}] ERROR: {type(e).__name__}: {e}")
    print(f"[done] {ok}/{len(scenes)} scenes rendered.")


if __name__ == "__main__":
    main()
