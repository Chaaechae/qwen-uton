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

Usage
-----
  python tools/judge_qwen_utility.py \
      --qwen        /group-volume/chaewon.yun/Qwen3.5-4B \
      --dinov2      facebook/dinov2-with-registers-giant \
      --image-root  /path/to/labeled_images \
      --crop        512

  <image-root> layout:  <image-root>/<class_name>/*.{jpg,png}
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
    pairs = []
    classes = sorted(
        d.name for d in image_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    for cls in classes:
        for p in sorted((image_root / cls).iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                pairs.append((p, cls))
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

    merger = model.merger
    hidden = merger.norm(hidden)
    # block-major reorder: rows 0,1 cols 0,1 -> 2x2 block per merged token
    hidden = hidden.view(B, h, w, -1)
    hidden = hidden.view(B, h // 2, 2, w // 2, 2, -1).permute(0, 1, 3, 2, 4, 5).contiguous()
    hidden = hidden.view(B * (h // 2) * (w // 2), 4 * hidden.shape[-1])
    hidden = merger.mlp(hidden) if hasattr(merger, "mlp") else \
             merger.linear_fc2(merger.act(merger.linear_fc1(hidden)))
    return hidden.view(B, (h // 2) * (w // 2), -1)


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
    if not HAS_SKLEARN:
        print("[warn] sklearn missing — skipping linear probe")
        return float("nan")
    skf = StratifiedKFold(n_splits=min(5, np.bincount(labels).min()), shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(features, labels):
        clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
        clf.fit(features[tr], labels[tr])
        accs.append(clf.score(features[te], labels[te]))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", type=str, default=None, help="Qwen3.5 model path or HF id")
    ap.add_argument("--dinov2", type=str, default=None, help="DINOv2 model path or HF id")
    ap.add_argument("--image-root", type=Path, required=True,
                    help="Directory of <class>/<image> subfolders.")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pairs, classes = list_labeled_images(args.image_root)
    if len(classes) < 2:
        raise SystemExit(f"Need >= 2 class folders under {args.image_root}, found {classes}")
    label_to_idx = {c: i for i, c in enumerate(classes)}
    labels = np.array([label_to_idx[c] for _, c in pairs])
    print(f"[data] {len(pairs)} images across {len(classes)} classes: {classes}")

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
