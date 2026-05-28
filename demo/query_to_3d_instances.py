"""
Query → 3D instance localization (multi-instance, structured output).

Pipeline
--------
    query text + image + scene point cloud
        │
        ▼
    Qwen3.5-VL "Detect all <X>" generation
        │  →  parse N bboxes from response
        ▼
    for each bbox:
        Qwen ViT (full merger) patches inside bbox  →  mean
                                                    →  qwen_proj  (H-trained, 2560→512)
                                                    │  q_i (512-d)
                                                    ▼
        Utonia backbone → patch_proj (1332→512) → 3D feats (N_s1, 512)
                                                    │
                                                    ▼ cosine
                                            per-point score for instance i
                                                    │
                                                    ▼ stage-1 → stage-0 → input gather
                                                    │ (KNN smooth optional)
                                                    │ percentile threshold
                                                    │ DBSCAN-style largest connected cluster
                                                    ▼
                                            instance i : {points, AABB, centroid, score}

Output
------
    out/instances.json
        [
          {"id": 0, "query": "chair", "bbox_2d": [x1,y1,x2,y2],
           "num_points": ..., "aabb": [[x,y,z],[x,y,z]], "centroid": [...],
           "max_score": ..., "mean_score": ..., "ply": "instances/inst_0.ply"},
          ...
        ]
    out/scene_rgb.ply                full scene in original colors
    out/heatmap_<id>.ply             per-instance score heatmap on full scene
    out/instances/inst_<id>.ply      points belonging to each instance
    out/instances_all.ply            all instances colored per id
    out/attn_2d.png                  image with all parsed bboxes overlaid

The whole pipeline is single-image / single-scene.  Multi-image fusion
(merge bboxes seen from N viewpoints) is left out — would just be a loop
over images + a per-point max over per-image score maps.

Reuses heavy parts of demo/text_query_to_3d_localization.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
# Reuse battle-tested helpers from the single-instance demo.  These
# already handle Qwen visual forward, H checkpoint loading, KD-tree
# clustering, PLY writing, etc.
from text_query_to_3d_localization import (  # noqa: E402
    build_h_modules,
    build_qwen,
    qwen_vision_full_merger,
    image_to_qwen_tensor,
    utonia_point_features,
    bbox_to_grid_mask,
    knn_smooth_scores,
    largest_connected_cluster,
    write_ply,
    jet_colormap,
)


# ---------------------------------------------------------------------------
# Multi-bbox parsing
# ---------------------------------------------------------------------------
def _parse_all_bboxes(text: str) -> list[tuple[float, float, float, float]]:
    """Pull every (x1,y1),(x2,y2) — or [x1,y1,x2,y2] — pair out of text.

    Qwen3.5-VL trained on grounding will respond with one of:
      - JSON-ish: ```json\n[{"bbox_2d":[..],"label":..}, ...]\n```
      - canonical: <|box_start|>(x1,y1),(x2,y2)<|box_end|> ... (repeated)
      - free text: "(x1,y1),(x2,y2)" lines separated by newlines
    All three reduce to "find every 4-number tuple/group".  We strip
    Qwen's special tokens first and look for paired-tuple matches; if
    that yields nothing, fall back to flat [x1,y1,x2,y2] lists.
    """
    clean = re.sub(r"<\|[^|]+\|>", " ", text)
    paired = re.findall(
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
        r"\s*,\s*"
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
        clean,
    )
    if paired:
        return [tuple(float(x) for x in m) for m in paired]
    flat = re.findall(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
        r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        clean,
    )
    return [tuple(float(x) for x in m) for m in flat]


def _normalize_bboxes(bboxes, proc_H, proc_W):
    """Convert raw bbox coords to normalized [0,1].

    Qwen3.5-VL canonically returns absolute pixel coords in the resized
    image space.  Some variants normalize to 0-1000.  Pick the rule by
    looking at the max coord across all bboxes (per-bbox would be brittle
    against a single tiny detection).
    """
    if not bboxes:
        return []
    max_coord = max(max(b) for b in bboxes)
    if max_coord > max(proc_H, proc_W) * 1.05:
        # Almost certainly 0-1000 scale.
        denom_x = denom_y = 1000.0
    else:
        denom_x = float(proc_W)
        denom_y = float(proc_H)
    out = []
    for x1, y1, x2, y2 in bboxes:
        x1n = max(0.0, min(1.0, x1 / denom_x))
        y1n = max(0.0, min(1.0, y1 / denom_y))
        x2n = max(0.0, min(1.0, x2 / denom_x))
        y2n = max(0.0, min(1.0, y2 / denom_y))
        if x2n > x1n and y2n > y1n:
            out.append((x1n, y1n, x2n, y2n))
    return out


@torch.inference_mode()
def qwen_multi_bbox_grounding(qwen, image_pil, query, device, image_size=512):
    """Run Qwen3.5-VL with a 'detect all' prompt and return ALL parseable bboxes.

    Returns
    -------
    bboxes : list of (x1, y1, x2, y2) in normalized [0, 1].
    raw    : the raw generation string (for logging).
    """
    if "model" not in qwen or qwen["processor"] is None:
        raise RuntimeError(
            "qwen multi-bbox grounding needs the full LLM + processor; "
            "rebuild qwen with keep_llm=True."
        )
    model = qwen["model"]
    processor = qwen["processor"]
    tokenizer = qwen["tokenizer"]

    img = image_pil.convert("RGB").resize((image_size, image_size))
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text",
             "text": (f"Detect every {query} in the image. "
                      "Respond with each bounding box on its own line in "
                      "the format (x1,y1),(x2,y2). "
                      "If there is none, respond with NONE.")},
        ],
    }]
    text_inp = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text_inp], images=[img], return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in inputs.items()}

    out_ids = model.generate(
        **inputs,
        max_new_tokens=256,         # room for multiple bboxes
        do_sample=False,
    )
    in_len = inputs["input_ids"].shape[1]
    raw = tokenizer.decode(out_ids[0, in_len:], skip_special_tokens=False)
    print(f"[qwen-multi-bbox] raw generation: {raw!r}")

    if "NONE" in raw.upper() and not re.search(r"\d", raw):
        return [], raw

    raw_bboxes = _parse_all_bboxes(raw)
    if not raw_bboxes:
        return [], raw

    # Determine processor's resized H/W for denormalization.
    if "image_grid_thw" in inputs:
        thw = inputs["image_grid_thw"][0]
        try:
            patch = model.config.vision_config.patch_size
        except AttributeError:
            patch = getattr(model.config, "patch_size", 14)
        proc_H = int(thw[1].item() * patch)
        proc_W = int(thw[2].item() * patch)
    else:
        proc_H = proc_W = image_size

    return _normalize_bboxes(raw_bboxes, proc_H, proc_W), raw


# ---------------------------------------------------------------------------
# Per-instance scoring
# ---------------------------------------------------------------------------
def score_points_for_query(query_2560, h, point_common_s1, inv_s1_to_s0,
                            inv_grid, knn_smooth, coord):
    """One ROI-mean query (in Qwen 2560-d) → per-original-point cosine score.

    Steps mirror text_query_to_3d_localization.py's main loop, but
    encapsulated for repeated calls (multi-instance).
    """
    with torch.inference_mode():
        q_common = h["qwen_proj"](query_2560.unsqueeze(0).float())[0]  # (512,)
    if not torch.isfinite(q_common).all():
        q_common = torch.nan_to_num(q_common, nan=0.0)

    pcn = F.normalize(point_common_s1.float(), dim=-1, eps=1e-6)
    qcn = F.normalize(q_common.float(), dim=-1, eps=1e-6)
    sim_s1 = (pcn @ qcn).clamp(-1.0, 1.0)                    # (N_s1,)
    sim_s0 = sim_s1[inv_s1_to_s0]                            # (N_s0,)
    sim_orig = sim_s0[inv_grid].cpu().numpy()                # (N_input,)

    if knn_smooth > 1:
        sim_orig = knn_smooth_scores(coord, sim_orig, k=knn_smooth)
    return sim_orig.astype(np.float32)


def extract_instance(sim, coord, top_percentile, cluster_eps,
                     min_points):
    """sim → top-percentile points → largest connected cluster.

    Returns (point_idx, score_stats_dict) or (None, None) if nothing
    survives the thresholds.
    """
    thr = float(np.percentile(sim, 100.0 - top_percentile))
    cand_idx = np.where(sim >= thr)[0]
    if len(cand_idx) < max(min_points, 3):
        return None, None
    local = largest_connected_cluster(coord[cand_idx], eps=cluster_eps)
    if len(local) < min_points:
        return None, None
    pt_idx = cand_idx[local]
    return pt_idx, {
        "threshold": thr,
        "n_candidates": int(len(cand_idx)),
        "n_cluster": int(len(pt_idx)),
        "max_score": float(sim[pt_idx].max()),
        "mean_score": float(sim[pt_idx].mean()),
    }


def aabb_centroid(coord_subset):
    lo = coord_subset.min(axis=0)
    hi = coord_subset.max(axis=0)
    return (lo.astype(float).tolist(),
            hi.astype(float).tolist(),
            ((lo + hi) * 0.5).astype(float).tolist())


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def save_attn_overlay(image_pil, bboxes, query, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_pil)
    W, H = image_pil.size
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        rect = mpatches.Rectangle(
            (x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H,
            linewidth=2.0, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(x1 * W + 4, y1 * H + 14, f"#{i}", color="lime",
                fontsize=10, weight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=1))
    ax.set_title(f"query='{query}'  detected={len(bboxes)} bboxes")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


_PALETTE = np.array([
    [230,  25,  75], [ 60, 180,  75], [255, 225,  25], [  0, 130, 200],
    [245, 130,  48], [145,  30, 180], [ 70, 240, 240], [240,  50, 230],
    [210, 245,  60], [250, 190, 190], [  0, 128, 128], [230, 190, 255],
    [170, 110,  40], [255, 250, 200], [128,   0,   0], [170, 255, 195],
], dtype=np.uint8)


def save_combined_instance_ply(out_path, coord, instances):
    """Color all original points by their assigned instance id; non-instance
    points stay neutral grey for context.
    """
    colors = np.full((len(coord), 3), 180, dtype=np.uint8)
    for idx, inst in enumerate(instances):
        c = _PALETTE[idx % len(_PALETTE)]
        colors[inst["point_idx"]] = c
    write_ply(out_path, coord, colors)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", required=True,
                   help="Preprocessed scene: coord.npy / color.npy / normal.npy")
    p.add_argument("--image-path", required=True,
                   help="Path to RGB image (absolute or under scene-dir).")
    p.add_argument("--query", required=True,
                   help="Open-vocab NL query, e.g. 'chair', 'all the windows'.")
    p.add_argument("--h-ckpt", required=True,
                   help="H/I training checkpoint (Pointcept training format).")
    p.add_argument("--qwen-path", required=True,
                   help="Qwen3.5-VL HF model dir.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image-size", type=int, default=512,
                   help="Square resize for Qwen ViT input; 512 matches H.")
    p.add_argument("--top-percentile", type=float, default=2.0,
                   help="Top-X%% scoring points are clustering candidates.")
    p.add_argument("--cluster-eps", type=float, default=0.15,
                   help="DBSCAN-like link radius (meters for indoor scenes).")
    p.add_argument("--min-points", type=int, default=80,
                   help="Drop clusters smaller than this many points.")
    p.add_argument("--smooth-knn", type=int, default=12,
                   help="KNN spatial score smoothing window. 1 = off.")
    p.add_argument("--save-heatmaps", action="store_true",
                   help="Dump per-instance heatmap PLYs (extra disk).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out = Path(args.out_dir)
    (out / "instances").mkdir(parents=True, exist_ok=True)

    # --- Scene -------------------------------------------------------------
    coord = np.load(os.path.join(args.scene_dir, "coord.npy")).astype(np.float32)
    color = np.load(os.path.join(args.scene_dir, "color.npy")).astype(np.uint8)
    normal = np.load(os.path.join(args.scene_dir, "normal.npy")).astype(np.float32)
    print(f"[scene] {args.scene_dir}  N={len(coord)}")
    write_ply(str(out / "scene_rgb.ply"), coord, color)

    # --- Image -------------------------------------------------------------
    img_path = args.image_path
    if not os.path.isabs(img_path):
        for base in (args.scene_dir,
                     os.path.dirname(os.path.dirname(args.scene_dir.rstrip("/")))):
            cand = os.path.join(base, img_path)
            if os.path.isfile(cand):
                img_path = cand
                break
    if not os.path.isfile(img_path):
        sys.exit(f"[error] image not found: {img_path}")
    print(f"[image] {img_path}")

    # --- Models ------------------------------------------------------------
    qwen = build_qwen(args.qwen_path, device, keep_llm=True)
    h = build_h_modules(args.h_ckpt, device)

    # --- Qwen multi-bbox grounding ----------------------------------------
    from PIL import Image
    image_pil = Image.open(img_path).convert("RGB")
    bboxes, raw = qwen_multi_bbox_grounding(
        qwen, image_pil, args.query, device, image_size=args.image_size,
    )
    print(f"[ground] parsed {len(bboxes)} bbox(es)")
    save_attn_overlay(image_pil, bboxes, args.query, str(out / "attn_2d.png"))

    if not bboxes:
        print("[exit] no bboxes — writing empty instances.json and quitting.")
        (out / "instances.json").write_text(json.dumps([], indent=2))
        return

    # --- Qwen ViT patches + Utonia 3D features -----------------------------
    image_tensor, _ = image_to_qwen_tensor(img_path, args.image_size)
    image_tensor = image_tensor.to(device)
    patch_2d = qwen_vision_full_merger(qwen["visual"], image_tensor, device)
    h_grid, w_grid, D = patch_2d.shape
    patch_2d_flat = patch_2d.reshape(-1, D).float()
    print(f"[2D]  Qwen merged: {h_grid}x{w_grid}x{D}")

    feat_s1, inv_s1_to_s0, inv_grid, _coord_s1 = utonia_point_features(
        h["backbone"], coord, color, normal, device,
    )
    feat_s1 = torch.nan_to_num(feat_s1, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.inference_mode():
        point_common_s1 = h["patch_proj"](feat_s1.float())
    point_common_s1 = torch.nan_to_num(point_common_s1, nan=0.0)

    # --- Per-bbox: score → extract instance --------------------------------
    instances = []
    for i, bbox in enumerate(bboxes):
        mask = bbox_to_grid_mask(bbox, h_grid, w_grid, device).flatten()
        if float(mask.sum()) <= 0:
            print(f"[bbox #{i}] empty grid mask after rasterization; skipping.")
            continue
        weights = mask / mask.sum()
        q_2560 = (weights.unsqueeze(-1) * patch_2d_flat).sum(dim=0)
        sim = score_points_for_query(
            q_2560, h, point_common_s1, inv_s1_to_s0, inv_grid,
            knn_smooth=args.smooth_knn, coord=coord,
        )
        pt_idx, stats = extract_instance(
            sim, coord,
            top_percentile=args.top_percentile,
            cluster_eps=args.cluster_eps,
            min_points=args.min_points,
        )
        if pt_idx is None:
            print(f"[bbox #{i}] no cluster survived thresholds "
                  f"(top {args.top_percentile}% / eps {args.cluster_eps}m / "
                  f"min {args.min_points} pts).")
            continue
        aabb_lo, aabb_hi, cent = aabb_centroid(coord[pt_idx])
        ply_path = out / "instances" / f"inst_{len(instances):02d}.ply"
        write_ply(
            str(ply_path),
            coord[pt_idx],
            np.tile(
                _PALETTE[len(instances) % len(_PALETTE)],
                (len(pt_idx), 1),
            ).astype(np.uint8),
        )
        if args.save_heatmaps:
            jet = jet_colormap(
                np.clip((sim - sim.min()) /
                        max(sim.max() - sim.min(), 1e-9), 0, 1)
            ).astype(np.uint8)
            write_ply(str(out / f"heatmap_{len(instances):02d}.ply"),
                      coord, jet)
        instances.append({
            "id": len(instances),
            "query": args.query,
            "bbox_2d": list(bbox),
            "num_points": int(len(pt_idx)),
            "aabb": [aabb_lo, aabb_hi],
            "centroid": cent,
            "max_score": stats["max_score"],
            "mean_score": stats["mean_score"],
            "threshold": stats["threshold"],
            "point_idx": pt_idx.tolist(),
            "ply": str(ply_path.relative_to(out)),
        })
        print(f"[inst #{len(instances)-1}] bbox={bbox}  pts={len(pt_idx)}  "
              f"score mean={stats['mean_score']:.3f} max={stats['max_score']:.3f}")

    # --- Save combined view + JSON -----------------------------------------
    save_combined_instance_ply(str(out / "instances_all.ply"), coord, instances)

    # Drop the heavy point_idx list from the JSON (keep it inside the PLYs);
    # large queries can produce 1e5+ points per instance, bloating the file.
    json_view = [
        {k: v for k, v in inst.items() if k != "point_idx"}
        for inst in instances
    ]
    (out / "instances.json").write_text(json.dumps(json_view, indent=2))
    # Save point indices separately (NPZ — far smaller than JSON int lists).
    np.savez_compressed(
        str(out / "instances_point_idx.npz"),
        **{f"inst_{inst['id']:02d}": np.asarray(inst["point_idx"], dtype=np.int64)
           for inst in instances},
    )

    print()
    print(f"[done] {len(instances)} instance(s)")
    print(f"       json:  {out / 'instances.json'}")
    print(f"       idx :  {out / 'instances_point_idx.npz'}")
    print(f"       plys:  {out / 'instances'} + {out / 'instances_all.ply'}")
    print(f"       img :  {out / 'attn_2d.png'}")


if __name__ == "__main__":
    main()
