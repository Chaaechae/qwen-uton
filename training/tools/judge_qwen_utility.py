"""
judge_qwen_utility.py — Is Qwen3.5 ViT worth using as a 2D teacher despite
DINOv2 looking richer per-patch?

What this measures
------------------
DINOv2 was trained with iBOT + multi-crop DINO, so each patch is a
self-contained semantic unit (high rank, well-spread, locally discriminative).
Qwen3.5 ViT was trained for LLM consumption — patches go through a 2x2
merger and the "real" output is at the merged 16x16 / LLM-hidden-dim scale.
Our experiments confirmed Qwen's per-patch features are anisotropic and
rank-collapsed at the raw-block level, BUT that doesn't necessarily mean
Qwen is useless — it might still carry text-aligned semantic that DINOv2
cannot ever produce.

So this script asks three concrete yes/no questions:

  Q1. Does the merged Qwen output (the representation H/I actually targets)
      separate basic indoor classes at all?  Measure: linear-probe top-1
      accuracy on per-image mean-pooled features, classes derived from
      folder names.  Compare to a DINOv2 baseline on the same data.

  Q2. Does it cluster classes (intra/inter cosine ratio)?  Cheap unsupervised
      proxy that doesn't depend on classifier training noise.

  Q3. Is there a text-alignment angle DINOv2 inherently lacks?  We feed each
      class name through Qwen's token embedding table and check whether the
      max-cosine class for each image's mean-pooled vision feature matches
      the ground-truth label.  This is a *minimal* CLIP-style retrieval
      probe — Qwen's text path isn't trained contrastively, so this is a
      lower bound on its zero-shot ability, not an upper bound.

Verdict logic
-------------
  - If Q1 (Qwen) >= 0.7 * Q1 (DINOv2):  "Qwen carries usable semantics"
  - Else if Q3 (Qwen) > random baseline by >= 2x:  "Qwen wins on text"
  - Else:                                          "Stick with DINOv2"

Input modes (one required)
--------------------------
  --image-root <dir>
      Class-folder layout: <dir>/<class>/*.{jpg,png,...}
      Most natural when you already have per-image class labels.

  --scannet-root <dir>  [--per-scene N]  [--scenes scene0011_00 ...]
                        [--color-subdir color]
      ScanNet / ARKitScenes / S3DIS-style layout:
          <dir>/<scene_id>/<color-subdir>/<frame>.{jpg,png}
      Each scene becomes one "class".  The Q1 linear probe + Q2 cluster
      stats then answer "do Qwen features cleanly separate rooms?"
      — weaker than per-class semseg labels but needs zero annotation.
      Q3 text-retrieval is meaningless for scene-ids (text embeddings
      of "scene0011_00" carry no semantic), so expect NaN there.

  --image-list <file>  [--image-list-root <dir>]
      Free-form: each line is "<path><sep><class>".  Use this when you
      have your own labels (e.g. ScanNet 2D semantic GT majority-class
      per image, or room-type metadata).

Usage
-----
  # ScanNet val set, 8 images per scene from the 'color' subdir,
  # whitelist of 20 scenes:
  python tools/judge_qwen_utility.py \
      --qwen        /group-volume/chaewon.yun/Qwen3.5-4B \
      --dinov2      facebook/dinov2-with-registers-giant \
      --scannet-root /group-volume/3Ddataset/data/scannet/val \
      --per-scene   8 \
      --scenes      scene0011_00 scene0050_00 scene0084_00 ... \
      --crop        512

  # Or auto-pick whatever scenes exist + cap total images:
  python tools/judge_qwen_utility.py \
      --qwen        Qwen/Qwen3.5-4B \
      --scannet-root /group-volume/3Ddataset/data/scannet/val \
      --per-scene   8 --max-images 400

  # Class-folder layout:
  python tools/judge_qwen_utility.py \
      --qwen        Qwen/Qwen3.5-4B \
      --image-root  /path/to/labeled_images
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def list_labeled_images(image_root: Path):
    """Class-folder layout: <image_root>/<class>/*.{jpg,png,...}"""
    pairs = []
    classes = sorted(
        d.name for d in image_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    for cls in classes:
        for p in sorted((image_root / cls).iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                pairs.append((p, cls))
    return pairs, classes


def list_scannet_images(scannet_root: Path, per_scene: int,
                        scenes: list[str] | None = None,
                        color_subdir: str = "color"):
    """ScanNet / ARKitScenes / S3DIS-style layout:
        <scannet_root>/<scene_id>/<color_subdir>/<frame>.{jpg,png,...}

    Each scene becomes one "class".  Linear probe / cluster-separability then
    answer: do Qwen features distinguish ROOMS from each other?  That's
    weaker than ScanNet 20-class semseg, but it's a clean unsupervised
    signal that doesn't require per-image labels.

    per_scene: cap N images per scene with even striding (diverse views,
               not just the first N adjacent frames).
    scenes:    optional whitelist of scene ids to include.
    """
    pairs = []
    scene_dirs = sorted(
        d for d in scannet_root.iterdir()
        if d.is_dir() and (d / color_subdir).is_dir()
    )
    if scenes:
        wanted = set(scenes)
        scene_dirs = [d for d in scene_dirs if d.name in wanted]
        missing = wanted - {d.name for d in scene_dirs}
        if missing:
            print(f"[warn] requested scenes missing from {scannet_root}: "
                  f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    for sd in scene_dirs:
        imgs = sorted(p for p in (sd / color_subdir).iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if per_scene and len(imgs) > per_scene:
            step = max(1, len(imgs) // per_scene)
            imgs = imgs[::step][:per_scene]
        for p in imgs:
            pairs.append((p, sd.name))
    classes = sorted({c for _, c in pairs})
    return pairs, classes


def load_splits_file(path: Path) -> list[str]:
    """Read a ScanNet/ARKit-style splits file: one scene id per line
    (comments / blank lines ignored).  Also accepts whitespace-separated
    ids on a single line, which some legacy splits files use.
    """
    raw = path.read_text()
    out = []
    for tok in raw.replace(",", " ").split():
        tok = tok.strip()
        if tok and not tok.startswith("#"):
            out.append(tok)
    if not out:
        raise ValueError(f"No scene ids found in splits file {path}")
    return out


def list_from_file(image_list: Path, root: Path | None = None):
    """Each line: <path>\\t<class>   (also accepts <path>,<class> or <path> <class>)
    `root` is prepended to relative paths.
    """
    pairs = []
    for raw in image_list.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("\t", ",", " "):
            if sep in line:
                path_str, cls = line.split(sep, 1)
                break
        else:
            raise ValueError(f"Cannot split path/class in line: {line!r}")
        path = Path(path_str.strip())
        if not path.is_absolute() and root is not None:
            path = root / path
        pairs.append((path, cls.strip()))
    classes = sorted({c for _, c in pairs})
    return pairs, classes


def build_transform(crop: int, mean, std):
    return transforms.Compose([
        transforms.Resize(crop, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Qwen3.5 ViT forward — mirrors the path utonia_v1m3b uses with
# use_full_merger=True so we measure exactly what H/I distills from.
# ---------------------------------------------------------------------------
@torch.no_grad()
def qwen_merged_features(model, x: torch.Tensor) -> torch.Tensor:
    """Return [B, h/2 * w/2, D_llm] full-merger output."""
    B, C, Hpx, Wpx = x.shape
    P = model.config.patch_size
    T = model.config.temporal_patch_size
    assert Hpx % P == 0 and Wpx % P == 0
    h, w = Hpx // P, Wpx // P
    assert h % 2 == 0 and w % 2 == 0, \
        f"merger needs h,w even; got {h}x{w} (crop must be a multiple of 2*P={2*P})"

    x_t = x.unsqueeze(1).repeat(1, T, 1, 1, 1)
    patches = x_t.view(B, T, C, h, P, w, P)
    patches = patches.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
    patches = patches.view(B * h * w, T * C * P * P)

    grid_thw = torch.tensor([[1, h, w]] * B, device=x.device, dtype=torch.long)
    hidden = model.patch_embed(patches)
    rotary = model.rot_pos_emb(grid_thw)
    emb = torch.cat((rotary, rotary), dim=-1)
    pos_emb = (emb.cos(), emb.sin())
    cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(0, dtype=torch.int32)
    cu = F.pad(cu, (1, 0), value=0)

    for blk in model.blocks:
        hidden = blk(hidden, cu_seqlens=cu, position_embeddings=pos_emb)

    # Block-major reorder before handing the (N, D) buffer to merger.
    # The merger module owns its own norm + spatial-2x2 merge + MLP path
    # (different Qwen3.5 builds package it as `mlp` or `linear_fc1/act/
    # linear_fc2` or even-newer variants).  Calling the module directly
    # avoids hard-coding any specific internal layout — same approach
    # utonia_v1m3b uses in `ENC2D_forward` (`self.enc2d_model.merger(h_blk)`).
    D = hidden.shape[-1]
    h_blk = (
        hidden.view(B, h, w, D)
              .view(B, h // 2, 2, w // 2, 2, D)
              .permute(0, 1, 3, 2, 4, 5)
              .contiguous()
              .view(B * (h // 2) * (w // 2) * 4, D)
    )
    merged = model.merger(h_blk)
    return merged.view(B, (h // 2) * (w // 2), -1)


@torch.no_grad()
def qwen_text_embeddings(full_model, tokenizer, prompts: list[str], device):
    """Cheap text-side probe: average input token embedding per prompt.

    Qwen3.5 is not trained contrastively, so there is no canonical
    `encode_text`.  Mean over the LM's input embedding table is the
    closest model-internal vector we can pull without running the LLM
    end-to-end — good enough to ask 'does the text-token space at all
    align with vision tokens via cosine?'.
    """
    embed = full_model.get_input_embeddings()
    out = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        out.append(embed(ids).mean(dim=1).squeeze(0).float().cpu())
    return torch.stack(out)


# ---------------------------------------------------------------------------
# DINOv2 forward (HF)
# ---------------------------------------------------------------------------
@torch.no_grad()
def dinov2_features(model, x: torch.Tensor) -> torch.Tensor:
    """Return [B, N_patch, D] patch features (drops CLS + register tokens)."""
    out = model(pixel_values=x)
    last = out.last_hidden_state  # [B, 1 + R + N, D]
    n_special = 1 + getattr(model.config, "num_register_tokens", 0)
    return last[:, n_special:, :]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def linear_probe(features: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    """Stratified-K-fold linear probe top-1.  Robust to sparse classes:
    drops classes with <2 samples (they can't be K-folded) and clamps
    n_splits to the smallest surviving class count, never below 2.
    """
    if not HAS_SKLEARN:
        print("[warn] sklearn missing — skipping linear probe")
        return float("nan")
    counts = np.bincount(labels)
    keep_mask = np.isin(labels, np.where(counts >= 2)[0])
    n_drop = int((~keep_mask).sum())
    if n_drop:
        print(f"[probe] dropping {n_drop} sample(s) from classes with <2 images "
              "(StratifiedKFold can't fold a singleton).")
    feats_k = features[keep_mask]
    labels_k = labels[keep_mask]
    if len(np.unique(labels_k)) < 2:
        print("[probe] <2 classes have >=2 samples; skipping linear probe.")
        return float("nan")
    min_count = int(np.bincount(labels_k)[np.unique(labels_k)].min())
    n_splits = max(2, min(5, min_count))
    print(f"[probe] StratifiedKFold(n_splits={n_splits})  "
          f"surviving classes={len(np.unique(labels_k))}  "
          f"samples={len(labels_k)}  min_per_class={min_count}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(feats_k, labels_k):
        clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
        clf.fit(feats_k[tr], labels_k[tr])
        accs.append(clf.score(feats_k[te], labels_k[te]))
    return float(np.mean(accs))


def class_centroid_stats(features: np.ndarray, labels: np.ndarray):
    feats = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    cents = []
    intra = []
    for c in np.unique(labels):
        sel = feats[labels == c]
        cent = sel.mean(0)
        cent /= np.linalg.norm(cent) + 1e-8
        cents.append(cent)
        intra.append(float((sel @ cent).mean()))
    cents = np.stack(cents)
    K = cents.shape[0]
    inter = (cents @ cents.T)[np.triu_indices(K, k=1)]
    return float(np.mean(intra)), float(np.mean(inter))


def text_retrieval_acc(image_feats: torch.Tensor, text_feats: torch.Tensor,
                       labels: np.ndarray, classes: list[str]) -> float:
    img_n = F.normalize(image_feats.float(), dim=-1)
    txt_n = F.normalize(text_feats.float(), dim=-1)
    # Project text into image-feature dim if they differ.  Simplest correct
    # answer when dims don't match: skip this metric.
    if img_n.shape[1] != txt_n.shape[1]:
        return float("nan")
    sims = img_n @ txt_n.T
    pred = sims.argmax(dim=-1).cpu().numpy()
    return float((pred == labels).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--qwen", type=str, default=None, help="Qwen3.5 model path or HF id")
    ap.add_argument("--dinov2", type=str, default=None, help="DINOv2 model path or HF id")

    # Three mutually-exclusive input modes:
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image-root", type=Path,
                     help="Class-folder layout: <root>/<class>/*.{jpg,png,...}")
    src.add_argument("--scannet-root", type=Path,
                     help="ScanNet-style layout: <root>/<scene>/<color-subdir>/*.{jpg,png}. "
                          "Each scene becomes one class (tests inter-scene separability).")
    src.add_argument("--image-list", type=Path,
                     help="Text file with one '<path><sep><class>' per line "
                          "(sep = tab / comma / space).")

    ap.add_argument("--per-scene", type=int, default=8,
                    help="(--scannet-root only) max images per scene; "
                         "evenly strided over the frame range.")
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="(--scannet-root only) whitelist of scene ids "
                         "(e.g. scene0011_00 scene0050_00 ...). "
                         "Merged with --splits-file if both given.")
    ap.add_argument("--splits-file", type=Path, default=None,
                    help="(--scannet-root only) text file with scene ids "
                         "(one per line or whitespace-separated). "
                         "Example: /.../scannet/splits/scannetv2_val.txt")
    ap.add_argument("--color-subdir", default="color",
                    help="(--scannet-root only) subdir inside each scene "
                         "holding the RGB frames. Default 'color' "
                         "(ScanNet); ARKit uses 'wide'.")
    ap.add_argument("--image-list-root", type=Path, default=None,
                    help="(--image-list only) prepended to relative paths.")

    ap.add_argument("--max-images", type=int, default=None,
                    help="Cap total images across all classes (random subsample).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ---- Resolve input mode -----------------------------------------------
    if args.image_root:
        pairs, classes = list_labeled_images(args.image_root)
        mode_label = f"class-folders @ {args.image_root}"
    elif args.scannet_root:
        scenes_filter = list(args.scenes) if args.scenes else []
        if args.splits_file:
            from_split = load_splits_file(args.splits_file)
            print(f"[splits] loaded {len(from_split)} scene ids from {args.splits_file}")
            scenes_filter = sorted(set(scenes_filter + from_split))
        pairs, classes = list_scannet_images(
            args.scannet_root,
            per_scene=args.per_scene,
            scenes=scenes_filter or None,
            color_subdir=args.color_subdir,
        )
        mode_label = (f"scannet-style @ {args.scannet_root} "
                      f"(per_scene={args.per_scene}, "
                      f"color_subdir={args.color_subdir!r}"
                      + (f", splits={args.splits_file.name}"
                         if args.splits_file else "")
                      + ")")
    else:
        pairs, classes = list_from_file(args.image_list, root=args.image_list_root)
        mode_label = f"image-list @ {args.image_list}"

    if args.max_images and len(pairs) > args.max_images:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(pairs), size=args.max_images, replace=False)
        pairs = [pairs[i] for i in idx]
        classes = sorted({c for _, c in pairs})
        print(f"[data] subsampled to {len(pairs)} images / {len(classes)} classes")

    if len(classes) < 2:
        raise SystemExit(
            f"Need >= 2 classes (got {classes}). "
            "For --scannet-root, supply >= 2 scene folders. "
            "For --image-list, ensure >= 2 distinct class labels."
        )
    label_to_idx = {c: i for i, c in enumerate(classes)}
    labels = np.array([label_to_idx[c] for _, c in pairs])
    print(f"[data] {len(pairs)} images across {len(classes)} classes "
          f"({mode_label})")
    if len(classes) <= 25:
        print(f"[data] classes: {classes}")
    else:
        print(f"[data] classes: {classes[:5]} ... {classes[-5:]} "
              f"(showing 10/{len(classes)})")
    cnts = np.bincount(labels)
    n_single = int((cnts == 1).sum())
    print(f"[data] per-class images: min={cnts[cnts > 0].min()}  "
          f"median={int(np.median(cnts[cnts > 0]))}  max={cnts.max()}  "
          f"singletons={n_single}")
    if n_single:
        print(f"[warn] {n_single} class(es) have only 1 image — they will be "
              "dropped from the linear probe (need >=2 for K-fold). "
              "Raise --per-scene or use --max-images higher to keep more "
              "images per scene.")

    summary = {}

    # ---- Qwen --------------------------------------------------------------
    if args.qwen:
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        print(f"[qwen] loading {args.qwen}")
        full = AutoModelForImageTextToText.from_pretrained(args.qwen, trust_remote_code=True).to(args.device).eval()
        tok = AutoTokenizer.from_pretrained(args.qwen, trust_remote_code=True)
        visual = full.model.visual

        tfm = build_transform(args.crop, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        feats = []
        for i in range(0, len(pairs), args.batch):
            batch = pairs[i:i + args.batch]
            x = torch.stack([tfm(Image.open(p).convert("RGB")) for p, _ in batch]).to(args.device)
            f = qwen_merged_features(visual, x)
            feats.append(f.mean(dim=1).float().cpu())
        feats = torch.cat(feats, dim=0).numpy()

        lp = linear_probe(feats, labels)
        intra, inter = class_centroid_stats(feats, labels)
        text_emb = qwen_text_embeddings(full.model, tok, classes, args.device)
        tr_acc = text_retrieval_acc(torch.from_numpy(feats), text_emb, labels, classes)

        summary["qwen"] = dict(linear_probe=lp, intra_cos=intra, inter_cos=inter,
                               separability=intra - inter, text_acc=tr_acc)
        del full, visual
        torch.cuda.empty_cache() if args.device == "cuda" else None

    # ---- DINOv2 ------------------------------------------------------------
    if args.dinov2:
        from transformers import AutoModel
        print(f"[dinov2] loading {args.dinov2}")
        dino = AutoModel.from_pretrained(args.dinov2).to(args.device).eval()
        # DINOv2's processor expects ImageNet stats; patch_size=14 in giant.
        patch = dino.config.patch_size
        crop = (args.crop // patch) * patch
        tfm = build_transform(crop, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        feats = []
        for i in range(0, len(pairs), args.batch):
            batch = pairs[i:i + args.batch]
            x = torch.stack([tfm(Image.open(p).convert("RGB")) for p, _ in batch]).to(args.device)
            f = dinov2_features(dino, x)
            feats.append(f.mean(dim=1).float().cpu())
        feats = torch.cat(feats, dim=0).numpy()

        lp = linear_probe(feats, labels)
        intra, inter = class_centroid_stats(feats, labels)
        summary["dinov2"] = dict(linear_probe=lp, intra_cos=intra, inter_cos=inter,
                                 separability=intra - inter, text_acc=float("nan"))

    # ---- Report ------------------------------------------------------------
    print()
    print(f"{'metric':<22} {'qwen':>10} {'dinov2':>10}")
    print("-" * 46)
    metrics = ["linear_probe", "intra_cos", "inter_cos", "separability", "text_acc"]
    q, d = summary.get("qwen", {}), summary.get("dinov2", {})
    for m in metrics:
        print(f"{m:<22} {q.get(m, float('nan')):>10.4f} {d.get(m, float('nan')):>10.4f}")
    print()

    # ---- Verdict -----------------------------------------------------------
    random_baseline = 1.0 / len(classes)
    print(f"random baseline: {random_baseline:.4f}")
    if "qwen" in summary and "dinov2" in summary:
        ratio = summary["qwen"]["linear_probe"] / max(summary["dinov2"]["linear_probe"], 1e-6)
        text_lift = summary["qwen"]["text_acc"] / random_baseline if summary["qwen"]["text_acc"] == summary["qwen"]["text_acc"] else 0
        print(f"qwen / dinov2 linear-probe ratio:   {ratio:.2f}")
        print(f"qwen text-retrieval lift over rand: {text_lift:.2f}x")
        if ratio >= 0.7:
            verdict = "Qwen carries usable semantics — alignment target is workable"
        elif text_lift >= 2.0:
            verdict = "Qwen has a text-alignment niche DINOv2 inherently lacks — use for open-vocab"
        else:
            verdict = "Qwen offers no measurable advantage on this data — stick with DINOv2"
        print()
        print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
