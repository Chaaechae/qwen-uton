"""
Debug why `total_img_num == 0` (i.e. why the enc2d alignment branch is
falling back to the SSL average) in the ScanNet-only distill recipe.

Usage (run from the Pointcept submodule so `data/scannet` resolves):
    cd third_party/Pointcept
    python ../../tools/debug_scannet_img_num.py [--limit 50] [--config <path>]

What it does (read-only):
  1. Builds the same ConcatDataset the training recipe builds (single ScanNet
     entry, indoor_transform, crop=512/patch=16).
  2. Iterates `--limit` samples and reports, per sample:
       - raw `images` count read from the splits.json entry
       - `img_num` AFTER the MultiViewGenerator's match_point_image filter
       - count of valid (non -1) entries in `global_correspondence`
  3. Prints aggregate counts at the end so you can see if (a) images aren't
     loaded at all, (b) MultiViewGenerator is dropping everything via
     match_point_image, or (c) correspondences are universally -1.

This hits ONLY the dataset (no model load), so it runs in seconds and
doesn't need the Qwen3.5 weight or any GPU.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np


def _build_dataset(config_path):
    # Pointcept's Config loader has the side effect of registering datasets,
    # so import it inside the function.
    from pointcept.utils.config import Config
    from pointcept.datasets.builder import build_dataset

    cfg = Config.fromfile(config_path)
    train_cfg = cfg.data.train
    if train_cfg.get("type") == "ConcatDataset":
        # Take the first inner dataset (ScanNet) so we can iterate it directly.
        ds_cfg = dict(train_cfg.datasets[0])
    else:
        ds_cfg = dict(train_cfg)
    print(f"[info] dataset cfg: type={ds_cfg.get('type')} "
          f"data_root={ds_cfg.get('data_root')} split={ds_cfg.get('split')}")
    return build_dataset(ds_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/utonia/distill-utonia-v1m3-1-scannet-only-qwen3_5-4b.py",
        help="Pointcept-side path to the recipe (relative to the Pointcept "
             "submodule cwd).",
    )
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[fatal] config not found at {args.config}. "
              f"Run from third_party/Pointcept.")
        sys.exit(1)

    ds = _build_dataset(args.config)
    print(f"[info] dataset length = {len(ds)}")

    raw_image_counts = Counter()
    img_num_after = Counter()
    valid_corr_counts = []
    raw_split_lookup_examples = 0
    samples_with_zero_imgs_raw = 0

    # Peek at the raw splits.json once to see what dataset says it has.
    splits_dir = os.path.join(ds.data_root, "splits")
    print(f"[info] reading raw splits from {splits_dir}")
    raw_image_per_scene = []
    import json
    for fn in sorted(os.listdir(splits_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(splits_dir, fn)) as f:
            d = json.load(f)
        for k, v in d.items():
            n = len(v.get("images", []))
            raw_image_per_scene.append(n)
            if raw_split_lookup_examples < 3:
                print(f"  [example] {fn} entry '{k}': {n} images, "
                      f"{len(v.get('correspondences', []))} correspondences")
                raw_split_lookup_examples += 1
    if raw_image_per_scene:
        print(f"[info] raw images-per-entry across splits: "
              f"min={min(raw_image_per_scene)} "
              f"mean={np.mean(raw_image_per_scene):.2f} "
              f"max={max(raw_image_per_scene)} "
              f"zero={sum(1 for n in raw_image_per_scene if n == 0)}/"
              f"{len(raw_image_per_scene)}")

    print(f"\n[info] iterating {min(args.limit, len(ds))} samples through "
          f"the full transform pipeline...")
    for i in range(min(args.limit, len(ds))):
        try:
            sample = ds[i]
        except Exception as e:
            print(f"  [sample {i}] FAILED: {e}")
            continue

        # `sample` is a dict already collected by the Collect transform. The
        # multi-view pipeline replaces top-level "images" / "img_num" with
        # global-view-filtered versions.
        img_num = sample.get("img_num", None)
        if img_num is not None:
            try:
                # img_num may be tensor or ndarray of len-1 batch.
                v = img_num.tolist() if hasattr(img_num, "tolist") else img_num
                # The Collect produces a 1-element list.
                v = v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v
                img_num_after[int(v)] += 1
            except Exception:
                img_num_after[str(img_num)] += 1

        gc = sample.get("global_correspondence", None)
        if gc is not None:
            try:
                arr = gc.numpy() if hasattr(gc, "numpy") else np.asarray(gc)
                # global_correspondence is (N_pts, V, 2). Count entries that are
                # NOT [-1,-1].
                valid = int(np.sum(np.any(arr != -1, axis=-1)))
                valid_corr_counts.append(valid)
            except Exception:
                pass

        if i < 5:
            print(f"  [sample {i}] img_num_after_mvg={img_num_after} "
                  f"valid_corr_so_far={valid_corr_counts[-1] if valid_corr_counts else 'n/a'}")

    print("\n=== summary ===")
    print(f"img_num histogram (post MultiViewGenerator):")
    for k, n in sorted(img_num_after.items()):
        print(f"  img_num={k}: {n} samples")
    if valid_corr_counts:
        print(f"valid global_correspondence rows per sample: "
              f"min={min(valid_corr_counts)} "
              f"mean={np.mean(valid_corr_counts):.0f} "
              f"max={max(valid_corr_counts)}")
    zero_after = img_num_after.get(0, 0)
    total = sum(img_num_after.values())
    print(f"\nimg_num==0 fraction: {zero_after}/{total} "
          f"({100.0 * zero_after / max(total, 1):.1f}%)")
    if zero_after == total:
        print("  → ALL samples drop their images. This is what triggers the "
              "enc2d-loss SSL fallback in concerto_v1m2_distill forward.")
        print("  Likely culprit: match_point_image filters out every image "
              "because the major_view crop has no overlap with any image's "
              "correspondences. Causes are usually:")
        print("   (a) the preprocessed correspondence/<frame>.npy files were "
              "       not extracted (check data/scannet/images/train/<scene>/"
              "       correspondence/ exists and is non-empty)")
        print("   (b) the dataset preprocessor produced correspondence files "
              "       with absolute paths that don't survive the symlink "
              "       (splits.py hardcodes 'data/scannet' — your cwd at "
              "       train time must be the directory containing the "
              "       'data/scannet' dir).")


if __name__ == "__main__":
    main()
