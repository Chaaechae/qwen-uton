"""
Text-query → 3D localization demo on a H-aligned Utonia scene.

Idea
----
H trained Utonia so that the 3D backbone (post patch_proj) and Qwen's
2D vision tokens (post qwen_proj) live in the SAME 512-d common space.
But to route a *text query* into that space we need a separate text→2D
grounding step, because Qwen3.5-VL is end-to-end LM-trained, NOT
contrastively trained à la CLIP — so cosine(text_embed, vision_patch)
is meaningless and produces a near-uniform map ("이미지 전체적으로
빨간 것들이 퍼져있어").

Solution: use SigLIP (which IS contrastive) for the text→2D step, and
use H + Qwen ViT only for the 2D→3D bridge.  Pipeline:

  text query ─► SigLIP text encoder ─┐
                                     ├─► 2D heatmap on SigLIP grid
  scene image ─► SigLIP vision tower─┘    (= clean object localization)
                                     │
                                     ▼ bilinear → Qwen 16x16 grid
                                     │ threshold @ percentile → ROI mask
                                     │
  scene image ─► Qwen ViT + merger ─► patch_2d (16x16, 2560)
                                     │
                          avg(patch_2d[ROI mask])  (only object patches)
                                     │
                                     ▼
                          query_grounded (2560-d, in Qwen's distribution)
                                     │ qwen_proj  (H's trained 2560→512)
                                     ▼
                          query_common (512-d) ◄── H's common space
                                                          ▲
  scene point cloud ─► Utonia backbone ─► (N_s1, 1332)    │
                                              │           │
                                              ▼ patch_proj│
                                          (N_s1, 512) ────┘
                                              │
                                              ▼ cosine
                                          3D heatmap → broadcast to
                                          original points via two
                                          gather steps (stage-1→stage-0,
                                          stage-0→input via GridSample
                                          inverse).

Inputs
------
  --scene-dir    Pointcept-preprocessed scene with coord.npy/color.npy
                 /normal.npy (the standard Utonia input).
  --image-path   Either an absolute path or a path RELATIVE to a sibling
                 images/<split>/<scene>/ tree following Utonia's data
                 layout (the script tries both).
  --query        NL text query, e.g. "냉장고", "refrigerator", "the chair
                 next to the window".  Pass any number of words; they
                 are tokenised and the embeddings are averaged.
  --h-ckpt       H training checkpoint (Pointcept training format).
                 Provides backbone + patch_proj + qwen_proj.
  --qwen-path    Path to Qwen3.5-4B HF dir (same one H trained against).
  --out-dir      Where to write the 2D overlay PNG + 3D heatmap PLY.

Grounding modes
---------------
--ground-mode siglip    (default) external SigLIP gives a text-image
                        cosine heatmap → percentile-threshold ROI mask.
                        Robust open-vocab, but adds SigLIP to inference.
--ground-mode qwen-bbox Ask Qwen3.5-VL to emit a bbox via generation
                        ("Locate the X. Output (x1,y1),(x2,y2)."), parse
                        the coordinates, rasterize as a soft cell-area
                        mask on the 16x16 grid.  Single-model pipeline;
                        leans on Qwen's grounding training.  Useful for
                        compositional queries SigLIP struggles with.

Run both modes against the same scene/query/image and compare the
scene_heatmap_*.ply / viz_*.html outputs to judge whether the H-aligned
bridge truly carries Qwen's grounding into 3D as well as it carries
SigLIP's.

Pose-based 2D→3D frustum filtering (optional, very effective)
-------------------------------------------------------------
--use-pose            Load camera pose + intrinsic (+ depth if present)
                      from ScanNet-style siblings of --image-path, then
                      project the 2D bbox through the camera into a 3D
                      frustum.  Outside-frustum points are clamped below
                      any inside-frustum score, so all downstream steps
                      (top-percentile, clustering, top-K) automatically
                      restrict to the geometric candidate region.  With
                      depth available, additionally drops 3D points
                      hidden behind a closer surface (occlusion test).

                      This is a HARD geometric constraint, complementary
                      to the SOFT feature-cosine score: feature alignment
                      is fuzzy across visually similar surfaces, but the
                      bbox + pose says only points whose ray actually
                      projects into the bbox can be the object.  Combine
                      the two and feature cosine only has to resolve
                      depth ambiguity within one ray bundle.

--estimate-pose       Recover (K_guess, T_c2w) FROM THE IMAGE ITSELF
                      via feature-PnP, no ScanNet pose required.  For
                      each Qwen 2D patch, find top-K best-matching
                      Utonia 3D points through the same H-aligned 512-d
                      common space, then solvePnPRansac(2D centers, 3D
                      world coords).  Lets you run the demo on
                      arbitrary photos.  Quality of recovered pose =
                      quality of H/I feature alignment — log prints
                      RANSAC inlier ratio so you can judge.  Implies
                      --use-pose.  Use --fov-deg to set the intrinsic
                      FOV guess (default 70°, fits most phones).

Outputs (suffixed by --ground-mode for side-by-side comparison)
---------------------------------------------------------------
  <out>/scene_rgb.ply                 Original-color point cloud (shared).
  <out>/attn_2d_<mode>.png            2D matplotlib overlay; qwen-bbox
                                      mode draws the parsed rectangle.
  <out>/scene_heatmap_<mode>.ply      Per-point similarity heatmap.
  <out>/scene_top_<mode>.ply          Top-K most similar points isolated.
  <out>/viz_<mode>.html               ★ Interactive Plotly page:
                                      LEFT — input image + ROI overlay,
                                      RIGHT — 3D heatmap, orbit/zoom.

Usage
-----
  # baseline (SigLIP-based ROI)
  python demo/text_query_to_3d_localization.py \\
      --scene-dir /group-volume/3Ddataset/data/scannet/val/scene0011_00 \\
      --image-path images/val/scene0011_00/color/12.png \\
      --query "refrigerator" \\
      --h-ckpt exp/utonia_q35_h/model/model_last.pth \\
      --qwen-path /group-volume/chaewon.yun/Qwen3.5-4B \\
      --out-dir exp/demo/scene0011_00_refrigerator

  # comparison run (Qwen-native bbox ROI) — same out-dir
  python demo/text_query_to_3d_localization.py \\
      ...same args... \\
      --ground-mode qwen-bbox

  # Qwen-bbox + pose-based frustum filtering (geometric + feature):
  python demo/text_query_to_3d_localization.py \\
      ...same args... \\
      --ground-mode qwen-bbox \\
      --use-pose

  # Arbitrary image (no ScanNet pose) — pose recovered via feature-PnP:
  python demo/text_query_to_3d_localization.py \\
      ...same args... \\
      --ground-mode qwen-bbox \\
      --estimate-pose --fov-deg 70
"""

import argparse
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PLY writer (no open3d dep — same as compare_seg_predictions.py)
# ---------------------------------------------------------------------------
def write_ply(path, coord, color):
    coord = np.asarray(coord, dtype=np.float32)
    color = np.asarray(color, dtype=np.uint8)
    n = coord.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
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


def jet_colormap(values):
    """values in [0, 1] → (N, 3) uint8 RGB using a jet-like palette."""
    v = np.clip(values, 0.0, 1.0).astype(np.float32)
    # 4-segment jet: blue → cyan → yellow → red
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1) * 255  # (N, 3) float


# ---------------------------------------------------------------------------
# Build the H model just enough to forward the 3D backbone + patch_proj
# + qwen_proj.  We do NOT load the full v1m3b training wrapper because
# inference only needs the backbone, patch_proj, qwen_proj.
# ---------------------------------------------------------------------------
def build_h_modules(h_ckpt, device):
    """Returns dict with backbone (callable), patch_proj, qwen_proj."""
    print(f"[load] H checkpoint: {h_ckpt}")
    ckpt = torch.load(h_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    sd = {(k[len("module."):] if k.startswith("module.") else k): v
          for k, v in sd.items()}

    # Pointcept's PT-v3m3 used by H — instantiate directly via the
    # utonia package's PointTransformerV3 so we don't need Pointcept's
    # full registry.
    sys.path.insert(0, "/home/user/qwen-uton")
    from utonia.model import PointTransformerV3
    backbone = PointTransformerV3(
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4, qkv_bias=True, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
        shuffle_orders=True, pre_norm=True, enable_rpe=False,
        # H training used flash + AMP-bf16.  At inference under plain
        # fp32 we've observed device-side asserts deep inside the
        # flash_attn varlen path (likely a cu_seqlens-shape edge case
        # when the cloud isn't pre-cropped to image-view size).
        # Vanilla SerializedAttention with fp32 is just as correct,
        # only ~2x slower — fine for a one-shot demo.
        enable_flash=False,
        upcast_attention=False, upcast_softmax=False,
        enc_mode=True, traceable=False,
        # mask_token has a learned parameter in H's training save
        # (mask_token=True at training).  Keep it on so its weight
        # slot exists and load_state_dict finds it, but it's not used
        # at inference (no masking).
        mask_token=True,
        rope_base=10, shift_coords=None,
        jitter_coords=None, rescale_coords=None,
    )
    # Pull just student.backbone.* into the bare PT-v3 namespace.
    bb_sd = {k[len("student.backbone."):]: v for k, v in sd.items()
             if k.startswith("student.backbone.")}
    miss, unexp = backbone.load_state_dict(bb_sd, strict=False)
    print(f"[load] backbone: matched={len(bb_sd) - len(unexp)} "
          f"missing={len(miss)} unexpected={len(unexp)}")
    backbone = backbone.to(device).eval()

    # patch_proj: MLP(1332 → 2048 → 512) from H config.
    patch_proj = nn.Sequential(
        nn.Linear(1332, 2048),
        nn.GELU(),
        nn.LayerNorm(2048),
        nn.Linear(2048, 512),
        nn.LayerNorm(512),
    )
    pp_sd = {k[len("patch_proj."):]: v for k, v in sd.items()
             if k.startswith("patch_proj.")}
    miss, unexp = patch_proj.load_state_dict(pp_sd, strict=False)
    print(f"[load] patch_proj: matched={len(pp_sd) - len(unexp)} "
          f"missing={len(miss)} unexpected={len(unexp)}")
    patch_proj = patch_proj.to(device).eval()

    # qwen_proj: Linear(2560 → 512, bias=False) + LayerNorm.  H config
    # used enc2d_head_in_channels=2560 (post-merger LLM hidden dim).
    qwen_proj = nn.Sequential(
        nn.Linear(2560, 512, bias=False),
        nn.LayerNorm(512),
    )
    qp_sd = {k[len("qwen_proj."):]: v for k, v in sd.items()
             if k.startswith("qwen_proj.")}
    miss, unexp = qwen_proj.load_state_dict(qp_sd, strict=False)
    print(f"[load] qwen_proj: matched={len(qp_sd) - len(unexp)} "
          f"missing={len(miss)} unexpected={len(unexp)}")
    qwen_proj = qwen_proj.to(device).eval()

    return dict(backbone=backbone, patch_proj=patch_proj, qwen_proj=qwen_proj)


# ---------------------------------------------------------------------------
# SigLIP — CLIP-style contrastive text/image encoder.
#
# Qwen3.5-VL's vision tokens and text embed_tokens share dimensionality
# but were NEVER trained to be cosine-similar to each other (Qwen is end-
# to-end LM, not contrastive).  Trying cosine(text_embed, vision_patch)
# produces a near-uniform map — the user observed exactly this ("이미지
# 전체적으로 빨간 것들이 퍼져있어").  SigLIP IS trained contrastively,
# so its text-image cosine cleanly localizes the query object.  We use
# SigLIP only for the 2D grounding map; H + Qwen handle the 3D bridge.
# ---------------------------------------------------------------------------
def build_siglip(siglip_path, device):
    print(f"[load] SigLIP: {siglip_path}")
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(siglip_path).to(device).eval()
    processor = AutoProcessor.from_pretrained(siglip_path)
    return dict(model=model, processor=processor)


def _patch_siglip_last_attn_to_value_only(model):
    """MaskCLIP trick: in the LAST vision encoder block, replace
    self-attention with a value-only pass (V @ W_O — no Q·K softmax
    mixing).  Without this, per-patch cosine maps are nearly uniform:
    the last block's full attention spreads each query/key globally,
    so every patch ends up carrying a near-identical "image summary",
    and cosine(text, patch_i) ≈ constant ∀ i.

    Reference: Zhou et al., "Extract Free Dense Labels from CLIP"
    (MaskCLIP, ECCV 2022).  Standard recipe for any CLIP-/SigLIP-style
    contrastive tower when you want per-patch grounding instead of a
    single pooled image vector.

    Idempotent: safe to call multiple times.
    """
    last_layer = model.vision_model.encoder.layers[-1]
    attn = last_layer.self_attn
    if getattr(attn, "_maskclip_patched", False):
        return
    if not (hasattr(attn, "v_proj") and hasattr(attn, "out_proj")):
        raise RuntimeError(
            f"SigLIP attention {type(attn).__name__} lacks v_proj/"
            "out_proj — MaskCLIP patch can't be applied here. "
            "(If this is SigLIP2 with a fused QKV, the projections may "
            "be packed into a single Linear; needs a variant-specific "
            "splitter.)"
        )
    v_proj = attn.v_proj
    out_proj = attn.out_proj

    def value_only_forward(hidden_states, *args, **kwargs):
        # (B, N, D) → identity on spatial dim; only feature transform.
        v = v_proj(hidden_states)
        attn_output = out_proj(v)
        return attn_output, None

    attn.forward = value_only_forward
    attn._maskclip_patched = True
    print("[siglip] applied MaskCLIP value-only patch to last vision block.")


@torch.inference_mode()
def siglip_grounding_map(siglip, image_pil, text_query, device):
    """
    Returns:
        heatmap : (h_sig, w_sig) float tensor on `device`, in [0, 1]
                  after min-max stretch.  h_sig × w_sig = SigLIP's
                  native patch grid (e.g. 16×16 for siglip-base-patch16-256).
        h_sig, w_sig : int

    Uses MaskCLIP-modified per-patch features (last self-attn replaced
    with V-only) cosine'd against the text-encoded query.  Without the
    modification, vanilla SigLIP `last_hidden_state` produces a
    near-uniform map — see `_patch_siglip_last_attn_to_value_only`.
    """
    model = siglip["model"]
    processor = siglip["processor"]

    # Make per-patch features text-alignable.  Idempotent.
    _patch_siglip_last_attn_to_value_only(model)

    # ---- Text side ----
    text_inputs = processor(text=[text_query], return_tensors="pt",
                            padding="max_length", truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_out = model.text_model(**text_inputs)
    # SigLIP's pooled text feature is the input to the contrastive loss;
    # it lives in the same space as image patch features projected by
    # the vision tower's pre-final-projection layer.
    text_feat = (text_out.pooler_output
                 if getattr(text_out, "pooler_output", None) is not None
                 else text_out.last_hidden_state[:, -1])
    text_feat = text_feat[0]  # (D,)

    # ---- Image side: per-patch features (post MaskCLIP-modified tower) ---
    img_inputs = processor(images=image_pil, return_tensors="pt")
    img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
    vision_out = model.vision_model(**img_inputs)
    # last_hidden_state has shape (1, n_patches, D) — no CLS for SigLIP.
    patch_feats = vision_out.last_hidden_state[0]  # (n_patches, D)
    n_patches, D = patch_feats.shape
    h_sig = w_sig = int(round(n_patches ** 0.5))
    assert h_sig * w_sig == n_patches, \
        f"SigLIP grid not square ({n_patches} patches); update if rectangular"

    # ---- Cosine similarity per patch ----
    text_n  = F.normalize(text_feat.float(), dim=-1, eps=1e-6)
    patch_n = F.normalize(patch_feats.float(), dim=-1, eps=1e-6)
    cos = (patch_n @ text_n).clamp(-1.0, 1.0)  # (n_patches,)

    # Diagnostic: if cos still looks near-uniform after the patch, the
    # spread tells you something is off (wrong model variant, wrong text
    # pooling, etc.) before you waste time staring at the heatmap.
    print(f"[siglip] cos per-patch: min={cos.min().item():.4f}  "
          f"max={cos.max().item():.4f}  std={cos.std().item():.4f}")

    # Stretch [min, max] → [0, 1] for visualization-friendly range.
    cos_min, cos_max = cos.min(), cos.max()
    heatmap = (cos - cos_min) / (cos_max - cos_min).clamp_min(1e-9)
    heatmap = heatmap.view(h_sig, w_sig)
    return heatmap, h_sig, w_sig


# ---------------------------------------------------------------------------
# Qwen3.5-VL — token embedding (text side) + vision tower (image side).
# Uses transformers.AutoModelForImageTextToText, same as H training.
# ---------------------------------------------------------------------------
def build_qwen(qwen_path, device, keep_llm=False):
    """Load Qwen3.5-VL. Returns dict with tokenizer/processor/embed_tokens/visual.

    If keep_llm=True, also keeps the full model under key 'model' so it can
    run .generate() — needed for --ground-mode qwen-bbox (Qwen-native
    grounding via autoregressive bbox emission).  Otherwise the LLM body
    is freed and only embed_tokens + visual stay resident (the original
    SigLIP path didn't need generation).
    """
    print(f"[load] Qwen: {qwen_path}"
          f"  ({'full LLM (for grounding gen)' if keep_llm else 'visual+embed only'})")
    from transformers import (
        AutoTokenizer, AutoModelForImageTextToText, AutoProcessor,
    )
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)
    except Exception:
        processor = None
    model = AutoModelForImageTextToText.from_pretrained(
        qwen_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model = model.to(device).eval()
    embed_tokens = model.get_input_embeddings()
    visual = model.model.visual
    result = dict(tokenizer=tokenizer, embed_tokens=embed_tokens,
                  visual=visual, processor=processor)
    if keep_llm:
        result["model"] = model
    else:
        # Drop the LLM body to save VRAM.
        del model
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Qwen-native grounding (alternative to SigLIP).
#
# Qwen3.5-VL was trained on grounding data where the LM is supervised to
# emit bounding-box coordinates as text tokens (Qwen2.5-VL canonical
# format:  <|object_ref_start|>name<|object_ref_end|>
#          <|box_start|>(x1,y1),(x2,y2)<|box_end|> ).  This taps that
# capability directly instead of relying on an external contrastive
# model: text query → LLM generation → parse bbox → rasterize a soft
# mask onto Qwen's 16×16 patch grid → same downstream pipeline as the
# SigLIP path.
#
# Why this is interesting to compare against SigLIP
# ------------------------------------------------
# * Qwen knows much richer object/relation vocabulary than SigLIP.
# * Qwen can ground compositional queries ("the chair NEAR the window")
#   while SigLIP gets diluted on relations.
# * If the H-aligned bridge truly works, the 3D heatmap from a
#   Qwen-derived ROI should be at least as clean as the SigLIP one.
#   When it isn't, the gap is informative — points to either ROI quality
#   issues or to the 2D→3D bridge being imperfect.
# ---------------------------------------------------------------------------
def _parse_first_bbox(text):
    """Parse the first 4-number bbox from a generation string.

    Handles:
      <|box_start|>(x1,y1),(x2,y2)<|box_end|>   ← Qwen 2.5/3.x canonical
      (x1,y1),(x2,y2)
      [x1, y1, x2, y2]
      x1, y1, x2, y2

    Returns a tuple of 4 floats or None.
    """
    # Strip Qwen special tokens to simplify regex.
    clean = re.sub(r"<\|[^|]+\|>", " ", text)
    # (a,b),(c,d) — paired tuples
    m = re.search(
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
        r"\s*,\s*"
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
        clean,
    )
    if m:
        return tuple(float(g) for g in m.groups())
    # [a, b, c, d] — flat list
    m = re.search(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
        r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        clean,
    )
    if m:
        return tuple(float(g) for g in m.groups())
    # Fallback: any 4 separator-delimited numbers
    m = re.search(
        r"(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s,]+"
        r"(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)",
        clean,
    )
    if m:
        return tuple(float(g) for g in m.groups())
    return None


@torch.inference_mode()
def qwen_bbox_grounding(qwen, image_pil, text_query, device, image_size=512):
    """Use Qwen LLM generation to ground the query as a 2D bbox.

    Returns (bbox_norm, raw_text):
        bbox_norm : (x1, y1, x2, y2) in normalized [0, 1] image coords,
                    or None if generation produced no parseable bbox.
        raw_text  : the model's raw output string (for logging / debug).
    """
    if "model" not in qwen:
        raise RuntimeError(
            "qwen-bbox mode needs the full LLM but build_qwen was called "
            "with keep_llm=False.")
    if qwen["processor"] is None:
        raise RuntimeError("Qwen processor not available; cannot run generate.")
    model = qwen["model"]
    processor = qwen["processor"]
    tokenizer = qwen["tokenizer"]

    # Squashed-resize to a known square so the bbox we get back lives in
    # the same image geometry as the 16×16 grid used by qwen_vision_full_
    # merger downstream.  Processor may further smart-resize to a multiple
    # of patch_size; we normalize bbox by image_grid_thw below to absorb
    # that small adjustment exactly.
    img = image_pil.convert("RGB").resize((image_size, image_size))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"Locate the {text_query} in the image. "
                        "Respond with only the bounding box coordinates "
                        "in the format (x1,y1),(x2,y2)."
                    ),
                },
            ],
        }
    ]
    text_inp = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text_inp], images=[img], return_tensors="pt")
    inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    out_ids = model.generate(
        **inputs,
        max_new_tokens=96,
        do_sample=False,
    )
    in_len = inputs["input_ids"].shape[1]
    gen_text = tokenizer.decode(out_ids[0, in_len:], skip_special_tokens=False)
    print(f"[qwen-bbox] raw generation: {gen_text!r}")

    raw_bbox = _parse_first_bbox(gen_text)
    if raw_bbox is None:
        return None, gen_text

    # Determine coord scale.  Qwen2.5/3.x-VL canonically uses absolute
    # pixel coords in the processor's *resized* image space.  Some
    # variants normalize to 0-1000.  Use image_grid_thw + patch_size to
    # recover the processor's actual H/W, then decide based on magnitude.
    if "image_grid_thw" in inputs:
        thw = inputs["image_grid_thw"][0]  # (3,)
        try:
            patch_size = model.config.vision_config.patch_size
        except AttributeError:
            patch_size = getattr(model.config, "patch_size", 14)
        proc_H = int(thw[1].item() * patch_size)
        proc_W = int(thw[2].item() * patch_size)
    else:
        proc_H = proc_W = image_size

    x1, y1, x2, y2 = raw_bbox
    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_coord <= 1.5:
        # Already normalized [0, 1].
        x1n, y1n, x2n, y2n = x1, y1, x2, y2
        scale_note = "normalized [0,1]"
    elif max_coord > max(proc_W, proc_H) and max_coord <= 1001:
        # Exceeds the proc image dims but fits in 0-1000 → must be the
        # 0-1000 normalized convention.
        x1n, x2n = x1 / 1000.0, x2 / 1000.0
        y1n, y2n = y1 / 1000.0, y2 / 1000.0
        scale_note = "normalized 0-1000"
    else:
        # Absolute pixel coords in proc_H × proc_W.
        x1n, x2n = x1 / proc_W, x2 / proc_W
        y1n, y2n = y1 / proc_H, y2 / proc_H
        scale_note = f"pixel in {proc_W}x{proc_H}"

    # Enforce ordering and clamp to [0, 1].
    x1n, x2n = sorted((x1n, x2n))
    y1n, y2n = sorted((y1n, y2n))
    bbox_norm = (
        max(0.0, min(1.0, x1n)),
        max(0.0, min(1.0, y1n)),
        max(0.0, min(1.0, x2n)),
        max(0.0, min(1.0, y2n)),
    )
    print(f"[qwen-bbox] parsed bbox ({scale_note}) → norm={bbox_norm}")
    return bbox_norm, gen_text


def bbox_to_grid_mask(bbox_norm, h_grid, w_grid, device):
    """Rasterize a normalized bbox to a soft (h_grid, w_grid) mask.

    Each cell's value is (bbox ∩ cell area) / (cell area), so cells
    entirely inside the box get 1.0 and cells partially covered get a
    fractional weight — better than a hard binary mask when the box
    happens to align poorly with the 16×16 grid.
    """
    x1, y1, x2, y2 = bbox_norm
    mask = torch.zeros(h_grid, w_grid, device=device)
    cell_w = 1.0 / w_grid
    cell_h = 1.0 / h_grid
    cell_area = cell_w * cell_h
    for r in range(h_grid):
        py1 = r * cell_h
        py2 = (r + 1) * cell_h
        iy1 = max(py1, y1)
        iy2 = min(py2, y2)
        if iy2 <= iy1:
            continue
        for c in range(w_grid):
            px1 = c * cell_w
            px2 = (c + 1) * cell_w
            ix1 = max(px1, x1)
            ix2 = min(px2, x2)
            if ix2 > ix1:
                mask[r, c] = (ix2 - ix1) * (iy2 - iy1) / cell_area
    return mask


# ---------------------------------------------------------------------------
# Run Qwen vision tower full-merger path, identical to the H training
# `ENC2D_forward(use_full_merger=True)` so the output 16×16 tokens
# match what qwen_proj was trained against.
# ---------------------------------------------------------------------------
@torch.inference_mode()
def qwen_vision_full_merger(visual, image_tensor, device):
    """image_tensor: (1, 3, 512, 512), in Qwen ViT normalisation.
    Returns (h_grid, w_grid, 2560) tensor.
    """
    P = visual.config.patch_size
    T = visual.config.temporal_patch_size
    B, C, H, W = image_tensor.shape
    h, w = H // P, W // P
    x_t = image_tensor.unsqueeze(1).repeat(1, T, 1, 1, 1)
    patches = x_t.view(B, T, C, h, P, w, P)
    patches = patches.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
    patches = patches.view(B * h * w, T * C * P * P).to(torch.bfloat16)
    grid_thw = torch.tensor([[1, h, w]] * B, device=device, dtype=torch.long)
    hidden = visual.patch_embed(patches)
    rotary_pos_emb = visual.rot_pos_emb(grid_thw)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    for blk in visual.blocks:
        hidden = blk(hidden, cu_seqlens=cu_seqlens,
                     position_embeddings=position_embeddings)
    # Full merger: (B*h*w, D) → reorder to 2×2 block-major before merger
    D = hidden.shape[-1]
    h_blk = (hidden.view(B, h, w, D)
                   .view(B, h // 2, 2, w // 2, 2, D)
                   .permute(0, 1, 3, 2, 4, 5).contiguous()
                   .view(B * (h // 2) * (w // 2) * 4, D))
    merged = visual.merger(h_blk)  # (B*(h/2)*(w/2), 2560)
    merged = merged.view(B, h // 2, w // 2, -1)
    return merged[0]  # (h/2, w/2, 2560)


def image_to_qwen_tensor(image_path, crop_size=512):
    from PIL import Image
    from torchvision import transforms
    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return tf(img).unsqueeze(0), img


# ---------------------------------------------------------------------------
# Run Utonia backbone on a point cloud, returning per-point 1332-d
# features (after 3 upcasts — same level H trained patch_proj at).
# ---------------------------------------------------------------------------
@torch.inference_mode()
def utonia_point_features(backbone, coord, color, normal, device):
    """coord/color/normal: numpy (N, 3).

    Returns (feat_s1, inv_s1_to_s0, inv_grid):
        feat_s1       : (N_s1, 1332) tensor at PT-v3 stage-1 resolution
                        (= what H trained patch_proj against).
        inv_s1_to_s0  : (N_s0,) long tensor — for each stage-0 (post-
                        GridSample grid) point, the index of its
                        owning stage-1 point.
        inv_grid      : (N_input,) long tensor — for each ORIGINAL
                        input point, the index of its owning stage-0
                        grid point.

    Why three return values
    -----------------------
    H's enc2d_upcast_level = 3, so the loss landed at stage-1 features
    (N/8 points, 1332-d).  The OLD demo tried `grid_feat[inv_grid]`
    to broadcast back to the original point cloud — but that indexes
    a stage-1 tensor with a stage-0 inverse → all entries OOB →
    silent CUDA assert that surfaces later as cublas/sgemv failure.

    Correct path: compute patch_proj + similarity at stage-1 resolution,
    then propagate down two levels via inv_s1_to_s0 then inv_grid:
        sim_s1   (N_s1,)
          ↓ sim_s1[inv_s1_to_s0]
        sim_s0   (N_s0,)
          ↓ sim_s0[inv_grid]
        sim_orig (N_input,)
    """
    sys.path.insert(0, "/home/user/qwen-uton")
    from utonia.structure import Point
    from utonia.transform import Compose

    transform = Compose([
        dict(type="GridSample", grid_size=0.01, hash_type="fnv",
             mode="train", return_grid_coord=True, return_inverse=True),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect",
             keys=("coord", "grid_coord", "color", "inverse"),
             feat_keys=("coord", "color", "normal")),
    ])
    pd = transform({
        "coord": coord.astype(np.float32),
        "color": color.astype(np.uint8),
        "normal": normal.astype(np.float32),
    })
    inv_grid = pd["inverse"].clone()  # (N_input,) — maps original → grid
    for k in list(pd.keys()):
        if isinstance(pd[k], torch.Tensor):
            pd[k] = pd[k].to(device)
    if pd.get("batch") is None:
        pd["batch"] = torch.zeros(pd["coord"].shape[0],
                                  dtype=torch.long, device=device)
    if pd.get("offset") is None:
        pd["offset"] = torch.tensor([pd["coord"].shape[0]],
                                    dtype=torch.long, device=device)
    point = Point(pd)
    point = backbone(point)

    # 3 upcasts from stage 4 (576) → stage 1 (1332-d, same level H
    # trained patch_proj on).  After this loop `point` IS the stage-1
    # Point, and it STILL carries its own pooling_parent (= stage 0)
    # + pooling_inverse (= which stage-1 point each stage-0 point pools
    # into).  We grab those WITHOUT popping (= without doing the 4th
    # upcast that would change the feature dim from 1332 to 1386).
    for _ in range(3):
        if "pooling_parent" not in point.keys():
            break
        parent = point.pop("pooling_parent")
        invp = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[invp]], dim=-1)
        point = parent

    feat_s1 = point.feat  # (N_s1, 1332) at stage 1
    coord_s1 = point.coord  # (N_s1, 3) world-frame stage-1 coords
    # The remaining pooling_inverse on `point` maps stage-0 → stage-1.
    inv_s1_to_s0 = (
        point["pooling_inverse"].clone()
        if "pooling_inverse" in point.keys() else
        torch.arange(feat_s1.shape[0], device=device)
    )
    return (
        feat_s1,
        inv_s1_to_s0.to(device),
        inv_grid.to(device),
        coord_s1.to(device),
    )


# ---------------------------------------------------------------------------
# Camera-pose-based 2D→3D mapping (alternative / complement to feature
# cosine).  Idea: H/I alignment gives a SEMANTIC match between 3D points
# and 2D patches, which is fuzzy across visually-similar surfaces.  But
# the bbox in the image + camera pose is a HARD geometric constraint —
# only points whose ray actually projects into the bbox are candidates.
# Combine the two: frustum mask gives the candidate region, cosine score
# within the frustum disambiguates if multiple objects sit in the same
# camera ray bundle.
#
# Layout expected (ScanNet image-dump — supports .txt OR .npy):
#     .../scene_XXXX_YY/color/<frame>.{png,jpg}
#     .../scene_XXXX_YY/pose/<frame>.{txt,npy}        4x4 cam-to-world
#     .../scene_XXXX_YY/intrinsic/intrinsic_color.{txt,npy}    4x4 K
#     .../scene_XXXX_YY/intrinsic/intrinsic_depth.{txt,npy}    4x4 K (opt)
#     .../scene_XXXX_YY/depth/<frame>.{png,npy}                (opt)
#
# Pose / intrinsic auto-detect tries .npy first, then .txt.  Override any
# single file with --pose-path / --intrinsic-color-path / etc.  If pose
# is unknown entirely (arbitrary image not in ScanNet) use --estimate-pose.
# ---------------------------------------------------------------------------
def _load_matrix_first_of(candidate_paths, name):
    """Try each candidate path in order, returning the first that loads.

    Loader picked by extension:  .npy / .npz → np.load,  .txt → np.loadtxt.
    Returns (array, actual_path) or raises FileNotFoundError listing all
    tried paths so the user can diagnose layout mismatch from the log.
    """
    tried = []
    for p in candidate_paths:
        tried.append(p)
        if not os.path.isfile(p):
            continue
        if p.endswith(".npy"):
            return np.load(p).astype(np.float32), p
        if p.endswith(".npz"):
            with np.load(p) as z:
                return z[list(z.keys())[0]].astype(np.float32), p
        return np.loadtxt(p).astype(np.float32), p
    raise FileNotFoundError(
        f"{name} not found.  Tried:\n  " + "\n  ".join(tried)
    )


def _candidate_paths_pose(scene_dir, frame):
    """Per-frame pose file candidates, .npy first then .txt."""
    return [
        os.path.join(scene_dir, "pose", f"{frame}.npy"),
        os.path.join(scene_dir, "pose", f"{frame}.txt"),
        os.path.join(scene_dir, "poses", f"{frame}.npy"),
        os.path.join(scene_dir, "poses", f"{frame}.txt"),
    ]


def _candidate_paths_intrinsic(scene_dir, kind):
    """Intrinsic file candidates.  `kind` ∈ {'color', 'depth', 'shared'}.
    'shared' covers layouts that store ONE intrinsic.npy used for both
    color & depth (e.g. when color and depth share a rectified camera).
    Order: specific-named → generic 'intrinsic' → at scene root.
    """
    by_dir = os.path.join(scene_dir, "intrinsic")
    if kind == "shared":
        roots = ["intrinsic"]
    else:
        roots = [f"intrinsic_{kind}", "intrinsic"]
    paths = []
    for r in roots:
        for ext in (".npy", ".txt"):
            paths.append(os.path.join(by_dir, r + ext))
            paths.append(os.path.join(scene_dir, r + ext))  # at scene root
    return paths


def _load_depth_first_of(candidate_paths):
    """Depth fallback: .png (16-bit mm) → .npy.  Auto-detects mm vs m by
    magnitude.  Returns (depth_meters, actual_path) or (None, None)."""
    from PIL import Image
    for p in candidate_paths:
        if not os.path.isfile(p):
            continue
        if p.endswith(".npy"):
            arr = np.load(p).astype(np.float32)
        elif p.endswith(".npz"):
            with np.load(p) as z:
                arr = z[list(z.keys())[0]].astype(np.float32)
        else:  # .png
            arr = np.array(Image.open(p)).astype(np.float32)
        # If values look like millimeters (max > 100m would be absurd
        # for indoor; mm range goes to ~10000), rescale.
        if np.isfinite(arr).any() and float(np.nanmax(arr)) > 100.0:
            arr = arr / 1000.0
        return arr, p
    return None, None


def load_camera_for_image(
    image_path, override_pose=None, override_intr_color=None,
    override_intr_depth=None, override_depth=None,
):
    """Returns dict with K_color (3,3), K_depth (3,3), T_c2w (4,4),
    H_color/W_color, and (if available) depth (H_d, W_d, float32 meters).

    Auto-detect probes several common layouts.  Each modality is logged
    so you can see exactly which file was used.
    """
    color_dir = os.path.dirname(image_path)
    auto_ok = (os.path.basename(color_dir) == "color")
    if not auto_ok and (override_pose is None or override_intr_color is None):
        raise RuntimeError(
            "auto-detect needs image to live at .../color/<frame>.png; "
            "got parent dir = "
            f"{os.path.basename(color_dir)}.  Override with --pose-path "
            "/ --intrinsic-color-path to skip auto-detect.")
    scene_dir = os.path.dirname(color_dir)
    frame = os.path.splitext(os.path.basename(image_path))[0]

    # --- pose -----------------------------------------------------------
    if override_pose is not None:
        T_c2w, pose_used = _load_matrix_first_of([override_pose], "pose")
    else:
        T_c2w, pose_used = _load_matrix_first_of(
            _candidate_paths_pose(scene_dir, frame), "pose",
        )
    if T_c2w.shape == (3, 4):
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        T_c2w = np.concatenate([T_c2w, bottom], axis=0)
    elif T_c2w.shape != (4, 4):
        raise RuntimeError(
            f"pose shape unexpected: {T_c2w.shape} from {pose_used}")
    if not np.isfinite(T_c2w).all():
        raise RuntimeError(
            f"pose has non-finite values (lost tracking?): {pose_used}")

    # --- intrinsic (color) ---------------------------------------------
    # Candidate order: explicit override → intrinsic_color → generic
    # 'intrinsic' (some preprocessors emit a single shared file).
    if override_intr_color is not None:
        K_c_full, intr_c_used = _load_matrix_first_of(
            [override_intr_color], "intrinsic_color",
        )
    else:
        K_c_full, intr_c_used = _load_matrix_first_of(
            _candidate_paths_intrinsic(scene_dir, "color")
            + _candidate_paths_intrinsic(scene_dir, "shared"),
            "intrinsic_color",
        )
    K_color = K_c_full[:3, :3].astype(np.float32)

    # --- intrinsic (depth) ---------------------------------------------
    # If the dump has a separate depth intrinsic, use it; otherwise share
    # K_color (typical when color and depth are rectified together).
    if override_intr_depth is not None:
        K_d_full, intr_d_used = _load_matrix_first_of(
            [override_intr_depth], "intrinsic_depth",
        )
        K_depth = K_d_full[:3, :3].astype(np.float32)
    else:
        try:
            K_d_full, intr_d_used = _load_matrix_first_of(
                _candidate_paths_intrinsic(scene_dir, "depth"),
                "intrinsic_depth",
            )
            K_depth = K_d_full[:3, :3].astype(np.float32)
        except FileNotFoundError:
            K_depth = K_color
            intr_d_used = "(none — reusing K_color)"

    # --- color image dims ----------------------------------------------
    from PIL import Image
    with Image.open(image_path) as img:
        W_color, H_color = img.size

    # --- depth (optional) ----------------------------------------------
    if override_depth is not None:
        depth_candidates = [override_depth]
    else:
        # Broad probe — depth maps live under wildly different layouts
        # depending on which extractor produced the dump.
        depth_candidates = [
            # canonical ScanNet extract
            os.path.join(scene_dir, "depth", f"{frame}.png"),
            os.path.join(scene_dir, "depth", f"{frame}.npy"),
            # alt subdir naming
            os.path.join(scene_dir, "depths", f"{frame}.png"),
            os.path.join(scene_dir, "depths", f"{frame}.npy"),
            # flat naming at scene root
            os.path.join(scene_dir, f"depth_{frame}.png"),
            os.path.join(scene_dir, f"depth_{frame}.npy"),
            os.path.join(scene_dir, f"{frame}_depth.png"),
            os.path.join(scene_dir, f"{frame}_depth.npy"),
            # color/<frame>.png  →  color/<frame>_depth.png  (some extracts)
            os.path.join(color_dir, f"{frame}_depth.png"),
            os.path.join(color_dir, f"{frame}_depth.npy"),
        ]
    depth, depth_used = _load_depth_first_of(depth_candidates)
    H_d = W_d = None
    if depth is not None:
        H_d, W_d = depth.shape

    print("[pose] files used:")
    print(f"           pose           : {pose_used}")
    print(f"           intrinsic_color: {intr_c_used}")
    print(f"           intrinsic_depth: {intr_d_used}")
    print(f"           depth          : {depth_used or '(none)'}")
    print(f"           K_color [fx,fy,cx,cy] = "
          f"({K_color[0,0]:.2f}, {K_color[1,1]:.2f}, "
          f"{K_color[0,2]:.2f}, {K_color[1,2]:.2f})  "
          f"image {W_color}x{H_color}")
    if depth is None:
        print("[pose] !!! depth map not found — both per-pixel occlusion "
              "AND --depth-slab are silently DISABLED.")
        print("       Without a depth filter, the bbox frustum is "
              "infinite-depth, so floor / background under the bbox "
              "edges WILL be included.")
        print("       Tried paths:")
        for p in depth_candidates:
            print(f"           {p}")
        print("       Pass --depth-path /abs/path explicitly if the file "
              "is somewhere else.")

    return dict(
        K_color=K_color, K_depth=K_depth, T_c2w=T_c2w,
        H_color=H_color, W_color=W_color,
        depth=depth, H_depth=H_d, W_depth=W_d,
    )


def project_world_to_pixel(coord, K, T_c2w, invert_pose=False):
    """coord (N, 3) world → (u, v, z_cam) in pixel coords + camera-frame
    z (used for occlusion / behind-camera checks).

    invert_pose=True: treat T_c2w as world-to-camera instead.  Use when
    the dump's pose .txt actually stores world→cam (some forks do).
    """
    if invert_pose:
        T_w2c = T_c2w  # interpret matrix as already world-to-camera
    else:
        T_w2c = np.linalg.inv(T_c2w)
    R = T_w2c[:3, :3]
    t = T_w2c[:3, 3]
    p_cam = coord @ R.T + t  # (N, 3)
    z = p_cam[:, 2]
    z_safe = np.where(z > 1e-6, z, 1e-6)
    u = K[0, 0] * p_cam[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * p_cam[:, 1] / z_safe + K[1, 2]
    return u, v, z


def bbox_object_depth_range(cam, bbox_norm,
                              q_lo=15.0, q_hi=85.0,
                              margin_front=0.15, margin_back=0.60):
    """Estimate the object's depth range from the depth map inside the bbox.

    Why this exists
    ---------------
    A 2D bbox + camera pose makes a 3D frustum — a cone that has no
    depth bound.  Points BEHIND or IN FRONT OF the object along the
    same rays end up inside the frustum (e.g. floor visible at the
    bottom of the bbox, or wall visible above the object).  Even the
    per-pixel depth-occlusion test doesn't help here, because the
    floor IS what's visible at those pixels — so its depth matches.

    Fix: sample depth values INSIDE the bbox region of the depth map.
    The robust [q_lo, q_hi] percentile gives the foreground surface's
    depth range.  Extend by `margin_back` to allow the object to have
    thickness, and by `margin_front` to absorb sensor noise.  Points
    with camera-z outside this slab are then dropped.

    Returns (z_min, z_max) in meters, or None if no depth available.
    """
    if cam.get("depth") is None:
        return None
    x1, y1, x2, y2 = bbox_norm
    H_d, W_d = cam["H_depth"], cam["W_depth"]
    u1 = max(0, int(x1 * W_d))
    u2 = min(W_d, int(x2 * W_d))
    v1 = max(0, int(y1 * H_d))
    v2 = min(H_d, int(y2 * H_d))
    if u2 <= u1 or v2 <= v1:
        return None
    depth_patch = cam["depth"][v1:v2, u1:u2]
    valid = depth_patch > 0.1
    if int(valid.sum()) < 16:
        return None
    valid_depths = depth_patch[valid]
    z_lo = float(np.percentile(valid_depths, q_lo))
    z_hi = float(np.percentile(valid_depths, q_hi))
    return (z_lo - margin_front, z_hi + margin_back)


def bbox_to_frustum_mask(coord, bbox_norm, cam, depth_tol=0.15,
                          depth_slab_back=0.60, depth_slab_front=0.15,
                          use_depth_slab=True,
                          invert_pose=False, diag=True):
    """Mask (N,) of 3D points whose projection lands inside the 2D bbox
    (and matches the depth map if depth_tol > 0 and depth is available).

    With diag=True, prints stage-by-stage projection statistics and a
    best-guess diagnosis if the projection looks broken.

    Returns:
        mask           : (N,) bool — final mask after bbox + depth
        n_in_frustum   : int — count after geometric bbox test (pre-depth)
        n_depth_dropped: int — count removed by depth occlusion check
        stages         : dict with bool arrays (in_front, in_image,
                         in_bbox) for debug visualization
    """
    x1, y1, x2, y2 = bbox_norm
    u1, u2 = x1 * cam["W_color"], x2 * cam["W_color"]
    v1, v2 = y1 * cam["H_color"], y2 * cam["H_color"]
    N = len(coord)

    u, v, z = project_world_to_pixel(
        coord, cam["K_color"], cam["T_c2w"], invert_pose=invert_pose,
    )

    # Three nested geometric stages — useful for diagnosing where the
    # projection breaks down.
    in_front = z > 1e-3
    in_image = (
        in_front
        & (u >= 0) & (u < cam["W_color"])
        & (v >= 0) & (v < cam["H_color"])
    )
    in_bbox = (
        in_front
        & (u >= u1) & (u < u2)
        & (v >= v1) & (v < v2)
    )

    if diag:
        n_front = int(in_front.sum())
        n_image = int(in_image.sum())
        n_bbox = int(in_bbox.sum())
        cam_pos = cam["T_c2w"][:3, 3] if not invert_pose \
                  else -cam["T_c2w"][:3, :3].T @ cam["T_c2w"][:3, 3]
        scene_min = coord.min(axis=0)
        scene_max = coord.max(axis=0)
        scene_center = (scene_min + scene_max) / 2.0
        dist = float(np.linalg.norm(cam_pos - scene_center))

        print(f"[pose-diag] projection stages (invert_pose={invert_pose}):")
        print(f"           total:        {N}")
        print(f"           z_cam > 0:    {n_front} ({100*n_front/N:.1f}%)")
        print(f"           inside image: {n_image} ({100*n_image/N:.1f}%)")
        print(f"           inside bbox:  {n_bbox} ({100*n_bbox/N:.1f}%)  "
              f"bbox_uv=[{u1:.0f},{v1:.0f}]–[{u2:.0f},{v2:.0f}]  "
              f"image={cam['W_color']}x{cam['H_color']}")
        print(f"           camera pos:   ({cam_pos[0]:+.2f}, {cam_pos[1]:+.2f}, "
              f"{cam_pos[2]:+.2f})")
        print(f"           scene center: ({scene_center[0]:+.2f}, "
              f"{scene_center[1]:+.2f}, {scene_center[2]:+.2f})  "
              f"extent ({scene_max[0]-scene_min[0]:.2f},"
              f"{scene_max[1]-scene_min[1]:.2f},"
              f"{scene_max[2]-scene_min[2]:.2f})")
        print(f"           cam-to-center distance: {dist:.2f}")

        # --- auto-diagnosis -------------------------------------------
        if n_front < 0.05 * N:
            print("[pose-diag] LIKELY BAD: <5% of points are in front of "
                  "camera.")
            print("            → pose convention may be inverted "
                  "(world-to-camera instead of camera-to-world).  "
                  "Try `--invert-pose`.")
        elif n_image < 0.05 * N:
            print("[pose-diag] LIKELY BAD: most points are behind camera "
                  "OR in front but outside the image.")
            print("            → K may be at the wrong resolution, or "
                  "image dims don't match the pose's view.  Verify "
                  "intrinsic_color.txt matches THIS image's resolution.")
        elif n_bbox == 0:
            print("[pose-diag] WARN: projection works but bbox area has "
                  "no scene coverage — likely an occluding wall, or the "
                  "bbox was drawn on background pixels.")
        elif n_bbox < 50:
            print(f"[pose-diag] note: only {n_bbox} pts in bbox — small "
                  "or distant object?  Should still be enough for "
                  "downstream clustering.")
        if dist > 30.0:
            print(f"[pose-diag] WARN: camera is {dist:.1f}m from scene "
                  "center.  Pose may be in a different coord frame than "
                  "the point cloud.")

    mask = in_bbox.copy()
    n_in_frustum = int(mask.sum())
    n_depth_dropped = 0
    n_slab_dropped = 0
    slab = None

    if depth_tol > 0 and cam.get("depth") is not None:
        # Per-pixel occlusion: project each scene point, sample depth at
        # its (u, v), check |camera_z - depth_at_uv| <= tol.  Removes
        # points hidden behind closer surfaces.
        ud, vd, _ = project_world_to_pixel(
            coord, cam["K_depth"], cam["T_c2w"], invert_pose=invert_pose,
        )
        Hd, Wd = cam["H_depth"], cam["W_depth"]
        ui = np.clip(ud.astype(np.int32), 0, Wd - 1)
        vi = np.clip(vd.astype(np.int32), 0, Hd - 1)
        d_at = cam["depth"][vi, ui]
        depth_valid = d_at > 0.1
        depth_match = np.abs(z - d_at) <= depth_tol
        depth_ok = depth_valid & depth_match
        n_depth_dropped = int((mask & ~depth_ok).sum())
        mask = mask & depth_ok

    if use_depth_slab and cam.get("depth") is not None:
        # Depth-slab: estimate the object's depth band from the bbox
        # region of the depth map, then keep only points whose
        # camera_z falls inside that band.  This eliminates floor /
        # background that the per-pixel occlusion test KEEPS (because
        # those distant surfaces ARE what's visible at the bbox's
        # boundary pixels, so their depth happens to match the map).
        slab = bbox_object_depth_range(
            cam, bbox_norm,
            margin_front=depth_slab_front,
            margin_back=depth_slab_back,
        )
        if slab is not None:
            z_min, z_max = slab
            in_slab = (z >= z_min) & (z <= z_max)
            n_slab_dropped = int((mask & ~in_slab).sum())
            mask = mask & in_slab
            if diag:
                print(f"[pose-diag] depth slab from bbox: "
                      f"[{z_min:.2f}m, {z_max:.2f}m]  "
                      f"({n_slab_dropped} extra dropped)")
        elif diag:
            print("[pose-diag] depth slab: bbox region has no valid "
                  "depth pixels — slab disabled.")

    stages = dict(in_front=in_front, in_image=in_image, in_bbox=in_bbox)
    diag_info = dict(
        n_depth_dropped=n_depth_dropped,
        n_slab_dropped=n_slab_dropped,
        depth_slab=slab,
    )
    return mask, n_in_frustum, n_depth_dropped, stages, diag_info


# ---------------------------------------------------------------------------
# Use the precomputed correspondence/<frame>.npy directly — the EXACT
# 2D↔3D mapping that H/I training used.  Beats pose-based projection when
# you don't have pose / depth / intrinsic, because:
#   - No coordinate-convention mismatches (it's the same ray-cast that
#     produced the InfoNCE supervision)
#   - Occlusion is built in: a point hidden behind a closer surface has
#     correspondence (-1, -1) for that frame, so it can't sneak into the
#     bbox by being "behind another surface that the depth check accepted"
#   - No need for K, T, depth files at all
#
# File layout (Pointcept-preprocessed ScanNet image dump):
#   .../scene/correspondence/<frame>.npy   (N_points_full, 2)
#     each row = (row_in_32x32, col_in_32x32) of that 3D point in this
#     image's 32×32 Qwen patch grid, or (-1, -1) if not visible.
#     Row-by-row aligned with the scene's coord.npy.
# ---------------------------------------------------------------------------
def load_correspondence_for_frame(image_path, override_path=None):
    """Returns (corr_array, path_used) or (None, None) if not found."""
    if override_path:
        return (np.load(override_path), override_path)
    color_dir = os.path.dirname(image_path)
    scene_dir = os.path.dirname(color_dir)
    frame = os.path.splitext(os.path.basename(image_path))[0]
    candidates = [
        os.path.join(scene_dir, "correspondence", f"{frame}.npy"),
        os.path.join(scene_dir, "correspondences", f"{frame}.npy"),
        os.path.join(scene_dir, f"correspondence_{frame}.npy"),
        os.path.join(scene_dir, f"{frame}_correspondence.npy"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return np.load(p), p
    return None, None


def correspondence_bbox_mask(correspondence, bbox_norm, patch_grid=32):
    """3D point mask from precomputed correspondence + 2D bbox.

    For each 3D point, True iff its (row_32, col_32) in the Qwen patch
    grid falls inside the bbox.  Points with (-1, -1) (not visible from
    this frame) are False automatically → occlusion handled for free.

    correspondence : (N, 2) int — (row_32, col_32) or (-1, -1)
    bbox_norm      : (x1, y1, x2, y2) in [0, 1] of the image
    patch_grid     : 32 by default (Qwen3.5-VL pre-merger grid)
    """
    x1, y1, x2, y2 = bbox_norm
    valid = (correspondence[:, 0] >= 0) & (correspondence[:, 1] >= 0)
    row = correspondence[:, 0].astype(np.float32)
    col = correspondence[:, 1].astype(np.float32)
    in_bbox = (
        valid
        & (row >= y1 * patch_grid) & (row < y2 * patch_grid)
        & (col >= x1 * patch_grid) & (col < x2 * patch_grid)
    )
    return in_bbox, int(valid.sum())


# ---------------------------------------------------------------------------
# Pose estimation via feature-PnP — the same H-aligned 512-d space that
# powers the cosine localization is ALSO sufficient to recover the
# camera pose when none is given.  For each Qwen 2D patch we find its
# best-matching Utonia 3D point via the common space, build (3D world ↔
# 2D pixel) correspondences, and solve PnP-RANSAC.  This turns arbitrary
# images (phone photos, web images) into 3D-localizable inputs without
# any external pose source.
#
# Caveats
# -------
# - Needs an intrinsic guess (K).  --fov-deg gives a default for a square
#   image; 60-80° covers most phones / ScanNet's StructureSensor.
# - Quality of recovered pose = quality of the H/I alignment.  If
#   feature matches are noisy, RANSAC's inlier set will be small.
#   Always print and check the inlier ratio.
# - PnP solves the GEOMETRIC inverse problem.  It returns ONE pose;
#   if features are too ambiguous (e.g. a symmetric room), the solver
#   may converge to a mirrored/rotated solution.
# ---------------------------------------------------------------------------
def make_default_intrinsic(image_W, image_H, fov_deg=70.0):
    """Build a centered pinhole K assuming horizontal-FOV = fov_deg.
    f is computed from the LONGER side so wide aspect ratios don't
    under-estimate focal length.
    """
    fov_rad = float(np.deg2rad(fov_deg))
    f = max(image_W, image_H) / (2.0 * np.tan(fov_rad / 2.0))
    K = np.array([
        [f, 0.0, image_W / 2.0],
        [0.0, f, image_H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return K


def patch_grid_pixel_centers(h_grid, w_grid, image_W, image_H):
    """(h_grid * w_grid, 2) — pixel coords of each cell's center when an
    image of size image_W × image_H is divided into a h_grid × w_grid
    regular grid (row-major flatten matching qwen_vision_full_merger)."""
    cell_w = image_W / w_grid
    cell_h = image_H / h_grid
    rs = np.arange(h_grid, dtype=np.float32) + 0.5
    cs = np.arange(w_grid, dtype=np.float32) + 0.5
    rr, cc = np.meshgrid(rs, cs, indexing="ij")
    u = cc.flatten() * cell_w
    v = rr.flatten() * cell_h
    return np.stack([u, v], axis=-1).astype(np.float32)


def estimate_pose_via_feature_pnp(
    patch_features_512, patch_centers_uv, K_guess,
    point_features_512, coord_3d,
    top_k=3, ransac_reproj_err=8.0, max_iter=2000,
):
    """Feature-based PnP-RANSAC pose estimation.

    patch_features_512 : (N_patches, 512) torch tensor (post-qwen_proj)
    patch_centers_uv   : (N_patches, 2) np.float32 pixel coords of cell
                         centers in the image
    K_guess            : (3, 3) np.float32 intrinsic guess
    point_features_512 : (N_pts, 512) torch tensor (post-patch_proj)
    coord_3d           : (N_pts, 3) np.float32 world coords matching
                         point_features_512 row-by-row
    top_k              : number of 3D candidates per 2D patch fed to
                         RANSAC (more = more outliers but more chances
                         of a good correspondence surviving the random
                         minimal-sample picks)

    Returns (T_c2w_4x4, inliers, n_correspondences) or None on failure.
    """
    import cv2

    p2 = F.normalize(patch_features_512.float(), dim=-1)  # (N_p, 512)
    p3 = F.normalize(point_features_512.float(), dim=-1)  # (N, 512)
    sim = p2 @ p3.T  # (N_p, N)
    top_sims, top_idx = sim.topk(top_k, dim=-1)  # (N_p, k)

    N_p = patch_centers_uv.shape[0]
    obj_pts = []
    img_pts = []
    for i in range(N_p):
        for j in range(top_k):
            obj_pts.append(coord_3d[top_idx[i, j].item()])
            img_pts.append(patch_centers_uv[i])
    obj_pts = np.asarray(obj_pts, dtype=np.float32)
    img_pts = np.asarray(img_pts, dtype=np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K_guess, distCoeffs=None,
        iterationsCount=max_iter,
        reprojectionError=ransac_reproj_err,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 6:
        return None

    R, _ = cv2.Rodrigues(rvec)
    T_w2c = np.eye(4, dtype=np.float32)
    T_w2c[:3, :3] = R
    T_w2c[:3, 3] = tvec.flatten()
    T_c2w = np.linalg.inv(T_w2c)
    return T_c2w, int(len(inliers)), int(N_p * top_k)


def bbox_from_grid_mask(mask_2d, h_grid, w_grid):
    """Derive a tight enclosing bbox from a 2D grid mask.  Used when
    --ground-mode siglip produces a soft heatmap rather than a bbox but
    --use-pose still needs a discrete rectangle.
    Returns (x1, y1, x2, y2) in normalized [0, 1] or None if empty.
    """
    m = mask_2d.cpu().numpy() if isinstance(mask_2d, torch.Tensor) else mask_2d
    m = m > 0
    rows = np.any(m, axis=1)
    cols = np.any(m, axis=0)
    if not rows.any() or not cols.any():
        return None
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return (
        float(c_min) / w_grid,
        float(r_min) / h_grid,
        float(c_max + 1) / w_grid,
        float(r_max + 1) / h_grid,
    )


# ---------------------------------------------------------------------------
# 3D score post-processing: KNN smoothing + spatial clustering.
#
# The H/I alignment makes a continuous feature manifold — raw cos-with-
# query scores tend to spread response across visually similar regions
# (e.g. fridge query lights up walls, other white surfaces).  These two
# steps tighten the response WITHOUT retraining:
#
#   knn_smooth_scores         per-point average with k nearest neighbors;
#                             kills isolated outlier hot specks, keeps
#                             coherent blobs.
#
#   largest_connected_cluster among the top-X% scoring points, BFS over
#                             a radius graph to find the largest
#                             spatially connected component.  Isolates a
#                             single object instance and drops scattered
#                             look-alike responses elsewhere in the scene.
# ---------------------------------------------------------------------------
def knn_smooth_scores(coord, scores, k):
    """Average each point's score with its k nearest neighbors.

    coord  : (N, 3) float
    scores : (N,) float
    k      : int; k <= 1 → no-op pass-through.

    scipy cKDTree is already a project dep (see demo/9_sem_seg_video.py),
    so this introduces no new requirement.
    """
    if k <= 1:
        return scores
    from scipy.spatial import cKDTree
    tree = cKDTree(coord)
    _, idx = tree.query(coord, k=k)
    return scores[idx].mean(axis=1)


def largest_connected_cluster(coord, eps):
    """Return indices of the largest eps-radius-connected component.

    coord : (M, 3) float — typically the top-X% candidate subset.
    eps   : float — link radius in coord units (meters for ScanNet
            indoor scans).  Two points within eps are graph-connected.
    """
    if len(coord) == 0:
        return np.zeros(0, dtype=np.int64)
    from scipy.spatial import cKDTree
    tree = cKDTree(coord)
    n = len(coord)
    visited = np.zeros(n, dtype=bool)
    best = []
    for seed in range(n):
        if visited[seed]:
            continue
        cluster = []
        stack = [seed]
        visited[seed] = True
        while stack:
            p = stack.pop()
            cluster.append(p)
            for nb in tree.query_ball_point(coord[p], eps):
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(cluster) > len(best):
            best = cluster
    return np.array(best, dtype=np.int64)


# ---------------------------------------------------------------------------
# Save the 2D attention as an overlay PNG.
# ---------------------------------------------------------------------------
def save_2d_overlay(image_pil, attn_grid, out_path, bbox_norm=None,
                     title_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image_pil)
    ax[0].set_title("input image")
    ax[0].axis("off")
    ax[1].imshow(image_pil, alpha=0.6)
    # Resize attn_grid to image size.
    h, w = image_pil.size[1], image_pil.size[0]
    a = attn_grid.cpu().float().numpy()
    a = (a - a.min()) / max(a.max() - a.min(), 1e-9)
    ax[1].imshow(a, alpha=0.55, cmap="jet",
                 extent=(0, w, h, 0), interpolation="bilinear")
    if bbox_norm is not None:
        x1, y1, x2, y2 = bbox_norm
        rect = mpatches.Rectangle(
            (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
            linewidth=2.5, edgecolor="#00ff00", facecolor="none",
        )
        ax[1].add_patch(rect)
    ax[1].set_title(f"2D query attention{title_suffix}")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Interactive Plotly HTML: 2D image + attention overlay on the left,
# 3D point cloud heatmap on the right (orbit/zoom/rotate).
# ---------------------------------------------------------------------------
def save_plotly_combined(
    image_pil, attn_grid, coord, color_rgb, sim_3d, query_text, out_path,
    max_points=80000, point_size=2, bbox_norm=None, mode_label="",
    top_percentile=10.0,
):
    """
    image_pil  : PIL.Image input image
    attn_grid  : (h, w) torch tensor — 2D attention from Qwen
    coord      : (N, 3) numpy point cloud xyz
    color_rgb  : (N, 3) numpy uint8 original colors (for context view)
    sim_3d     : (N,) numpy per-point similarity score to query
    query_text : the NL query string (for the figure title)
    out_path   : output .html path

    Renders ONE html with two side-by-side subplots:
       LEFT  — 2D image with jet-colored attention overlay
       RIGHT — 3D scatter, marker color = jet(sim_norm)
    Browsers struggle past ~200k 3D points; subsample if N > max_points.
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    # ---------- 2D side: bake the attention overlay into one RGB image ----
    img = np.asarray(image_pil.convert("RGB")).astype(np.float32) / 255.0  # (H, W, 3)
    H, W, _ = img.shape

    a = attn_grid.cpu().float().numpy()
    a = (a - a.min()) / max(a.max() - a.min(), 1e-9)
    # Upsample to image resolution via numpy (bilinear via torch).
    import torch.nn.functional as F
    a_up = F.interpolate(
        torch.from_numpy(a)[None, None],
        size=(H, W), mode="bilinear", align_corners=False,
    )[0, 0].numpy()
    jet = cm.get_cmap("jet")(a_up)[..., :3]  # (H, W, 3) in [0, 1]
    # 60% image + 40% heatmap blend.
    overlay = (0.45 * img + 0.55 * jet)
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    # Draw bbox rectangle onto the overlay if provided (qwen-bbox mode).
    if bbox_norm is not None:
        x1n, y1n, x2n, y2n = bbox_norm
        x1, x2 = int(round(x1n * W)), int(round(x2n * W))
        y1, y2 = int(round(y1n * H)), int(round(y2n * H))
        x1, x2 = max(0, x1), min(W - 1, x2)
        y1, y2 = max(0, y1), min(H - 1, y2)
        line_color = np.array([0, 255, 0], dtype=np.uint8)
        for t in range(3):  # 3-pixel-thick stroke
            if 0 <= y1 - t < H:
                overlay[y1 - t, x1:x2 + 1] = line_color
            if 0 <= y2 + t < H:
                overlay[y2 + t, x1:x2 + 1] = line_color
            if 0 <= x1 - t < W:
                overlay[y1:y2 + 1, x1 - t] = line_color
            if 0 <= x2 + t < W:
                overlay[y1:y2 + 1, x2 + t] = line_color

    # ---------- 3D side: subsample for browser performance ----------------
    n = coord.shape[0]
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        coord_s = coord[idx]
        sim_s   = sim_3d[idx]
    else:
        coord_s = coord
        sim_s   = sim_3d

    # Top-percentile split: above-threshold rendered with jet colors and
    # full opacity, below-threshold as faint grey context.  Stretch jet
    # over the above-threshold range so the highest scores are red and
    # the threshold itself sits near blue.
    if top_percentile < 100.0:
        score_thr = float(np.percentile(sim_s, 100.0 - top_percentile))
        above = sim_s >= score_thr
    else:
        score_thr = float(sim_s.min())
        above = np.ones_like(sim_s, dtype=bool)

    if above.any():
        above_lo = float(sim_s[above].min())
        above_hi = float(sim_s[above].max())
        sim_norm_above = np.clip(
            (sim_s[above] - above_lo) / max(above_hi - above_lo, 1e-9),
            0, 1,
        )
    else:
        sim_norm_above = np.zeros(0, dtype=np.float32)

    # ---------- Build subplots --------------------------------------------
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "image"}, {"type": "scene"}]],
        column_widths=[0.4, 0.6],
        subplot_titles=(
            f"2D attention  (query='{query_text}'"
            f"{', mode=' + mode_label if mode_label else ''})",
            "3D point cloud — heatmap",
        ),
        horizontal_spacing=0.04,
    )

    fig.add_trace(go.Image(z=overlay), row=1, col=1)

    # 3D scatter — two traces for clean object isolation:
    #   (a) below-threshold points: faint grey context.
    #   (b) above-threshold points: jet colored, full opacity.
    coord_below = coord_s[~above]
    if coord_below.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=coord_below[:, 0],
                y=coord_below[:, 1],
                z=coord_below[:, 2],
                mode="markers",
                marker=dict(
                    size=max(1, point_size - 1),
                    color="rgb(170,170,170)",
                    opacity=0.18,
                ),
                hoverinfo="skip",
                name="context",
            ),
            row=1, col=2,
        )

    coord_above = coord_s[above]
    sim_above_raw = sim_s[above]
    if coord_above.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=coord_above[:, 0],
                y=coord_above[:, 1],
                z=coord_above[:, 2],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=sim_norm_above,
                    colorscale="Jet",
                    cmin=0.0, cmax=1.0,
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text=f"top {top_percentile:.0f}% score",
                            side="right",
                        ),
                        thickness=14, len=0.7, x=1.02,
                    ),
                    opacity=0.97,
                ),
                customdata=sim_above_raw,
                hovertemplate=(
                    "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}"
                    "<br>score: %{customdata:.3f}<extra></extra>"
                ),
                name=f"top {top_percentile:.0f}%",
            ),
            row=1, col=2,
        )

    fig.update_layout(
        title=(
            f"Text-query 3D localization — '{query_text}'"
            f"{'  [' + mode_label + ']' if mode_label else ''}"
        ),
        height=720, width=1500,
        scene=dict(
            aspectmode="data",
            xaxis=dict(showbackground=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showticklabels=False, title=""),
            zaxis=dict(showbackground=False, showticklabels=False, title=""),
            camera=dict(eye=dict(x=1.3, y=-1.3, z=0.8)),
        ),
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)

    fig.write_html(out_path, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir",  required=True)
    p.add_argument("--image-path", required=True)
    p.add_argument("--query",      required=True,
                   help="Text query, e.g. 'refrigerator' or '냉장고'")
    p.add_argument("--h-ckpt",     required=True)
    p.add_argument("--qwen-path",  required=True)
    p.add_argument(
        "--ground-mode", choices=["siglip", "qwen-bbox"], default="siglip",
        help="How to convert the text query into a 2D ROI on Qwen's 16x16 "
             "patch grid.  'siglip' (default) uses an external "
             "contrastively-trained model — robust, but adds a SigLIP "
             "dependency at inference.  'qwen-bbox' uses Qwen3.5-VL's "
             "own grounding ability (LLM emits bbox coordinates as text "
             "tokens) — keeps the pipeline single-model but quality "
             "depends on Qwen recognizing the query.  Run both, compare "
             "the 3D heatmap PLYs.")
    p.add_argument(
        "--siglip-path",
        default=os.environ.get(
            "SIGLIP_PATH", "google/siglip2-base-patch16-256",
        ),
        help="SigLIP model path (HF id or local dir) for text→2D "
             "grounding when --ground-mode siglip.  Defaults to "
             "$SIGLIP_PATH if set, else the HF id "
             "'google/siglip2-base-patch16-256'.  Qwen3.5-VL is NOT a "
             "CLIP-style contrastive model — its text embed_tokens and "
             "vision patches share a dim but are NOT cosine-comparable, "
             "so the cosine-based ROI step needs an external contrastive "
             "model here.")
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--siglip-thr-percentile", type=float, default=85.0,
                   help="Percentile threshold (0..100) on the SigLIP 2D map "
                        "to define which patches count as 'inside the object' "
                        "for the grounded query.  Higher = stricter ROI.")
    p.add_argument("--temperature", type=float, default=0.07,
                   help="Softmax temperature for legacy 2D attention map "
                        "(unused when --siglip-path is set).")
    p.add_argument("--top-k", type=int, default=1024,
                   help="# of top-scored points to isolate in scene_top.ply")
    p.add_argument(
        "--bg-subtract", dest="bg_subtract",
        action="store_true", default=True,
        help="Score points as cos(point, ROI-mean) - cos(point, "
             "outside-ROI-mean) instead of plain cos against the ROI "
             "mean.  Rewards 'looks like the object AND unlike the "
             "scene background', which sharpens localization a lot. "
             "On by default; pass --no-bg-subtract for a vanilla "
             "ROI-cos baseline.")
    p.add_argument(
        "--no-bg-subtract", dest="bg_subtract", action="store_false",
        help="Disable background subtraction (baseline ROI-cos only).")
    p.add_argument(
        "--top-percentile", type=float, default=10.0,
        help="Show only the top X%% of (post-bg-subtract) scored points "
             "with jet colors in scene_heatmap/viz; the rest are "
             "rendered as flat grey context.  Counters the 'everything "
             "is red' effect that percentile [5, 99] stretch produces "
             "when the cos distribution is narrow.  Use 100 to disable "
             "masking and color the full spectrum (old behavior).")
    p.add_argument(
        "--roi-erode", type=float, default=0.5,
        help="Soft erosion threshold on the 2D ROI mask before building "
             "q_pos.  Cells whose mask weight is below this are dropped "
             "from the q_pos average — keeps the query closer to a "
             "pure-object signal (less bbox-boundary contamination). "
             "Higher = stricter (smaller effective ROI).  0.0 disables.")
    p.add_argument(
        "--smooth-knn", type=int, default=16,
        help="K-nearest-neighbor smoothing of the per-point 3D score "
             "(each point's score becomes the mean of its K NNs in 3D "
             "space).  Removes isolated outlier responses while keeping "
             "coherent object-shaped blobs.  Set to 0 or 1 to disable. "
             "Cheap (one scipy cKDTree query).")
    p.add_argument(
        "--cluster-eps", type=float, default=0.05,
        help="Spatial clustering radius in coord units (meters for "
             "ScanNet).  Among the top-(--top-percentile)%% points, "
             "find the largest eps-radius-connected component and save "
             "it as scene_object_<mode>.ply — single-instance extraction "
             "that drops scattered look-alike responses elsewhere in "
             "the scene.  Set to 0 to disable clustering.")
    p.add_argument(
        "--use-correspondence", action="store_true", default=False,
        help="Use the precomputed correspondence/<frame>.npy file (the "
             "ray-cast mapping that H/I training used) to filter 3D "
             "points by the 2D bbox.  Preferred over --use-pose when "
             "you have correspondence files but no pose/intrinsic/depth, "
             "because it's the EXACT mapping training saw, with built-in "
             "occlusion handling: hidden points have (-1, -1) and are "
             "auto-excluded.  No coordinate-convention or K-scale "
             "concerns.  Implies the same downstream pipeline as "
             "--use-pose (frustum mask → cosine score within frustum).")
    p.add_argument(
        "--correspondence-path", default=None,
        help="Override path to <frame>.npy correspondence file (default: "
             "auto-detect from --image-path siblings).")
    p.add_argument(
        "--correspondence-patch-grid", type=int, default=32,
        help="Patch grid resolution of the correspondence file.  H/I "
             "training stores (row, col) in 32x32 units even when the "
             "effective post-merger grid is 16x16 (the loss path halves "
             "via correspondence_stride=2).  Keep at 32 unless your "
             "preprocessing differs.")
    p.add_argument(
        "--use-pose", action="store_true", default=False,
        help="Enable camera-pose-based 2D→3D frustum filtering.  Loads "
             "pose / intrinsic / depth from sibling dirs of --image-path "
             "(ScanNet extract layout).  After the cosine score is "
             "computed, points OUTSIDE the bbox frustum are clamped to "
             "the bottom of the score distribution — so top-X%% / "
             "clustering only ever consider the geometric candidate "
             "region.  With depth available, also drops 3D points that "
             "project into the bbox but lie behind a closer surface "
             "(occlusion test).  Hard geometric constraint that "
             "complements the fuzzy feature alignment.")
    p.add_argument(
        "--depth-tol", type=float, default=0.15,
        help="Depth match tolerance in meters for the occlusion test. "
             "A 3D point is kept if abs(projected camera-z - depth_map[u,v]) "
             "<= this. Set to 0 to skip depth checking (frustum only).")
    p.add_argument("--pose-path", default=None,
                   help="Override path to <frame>.txt pose file "
                        "(default: auto-detect ScanNet layout).")
    p.add_argument("--intrinsic-color-path", default=None,
                   help="Override path to intrinsic_color.txt.")
    p.add_argument("--intrinsic-depth-path", default=None,
                   help="Override path to intrinsic_depth.txt.")
    p.add_argument("--depth-path", default=None,
                   help="Override path to <frame>.png depth map.")
    p.add_argument(
        "--depth-slab", dest="depth_slab",
        action="store_true", default=True,
        help="In addition to per-pixel occlusion, estimate the object's "
             "depth band from the depth-map region inside the bbox and "
             "keep only points whose camera-z falls within that band. "
             "Fixes the 'frustum extends to the floor / wall' issue: "
             "those distant surfaces ARE visible at bbox-edge pixels, so "
             "the per-pixel occlusion test accepts them, but they have "
             "very different depth from the actual object.  Default on; "
             "needs depth map.  Disable with --no-depth-slab.")
    p.add_argument(
        "--no-depth-slab", dest="depth_slab", action="store_false",
        help="Disable bbox-region depth-slab filtering (per-pixel "
             "occlusion still runs if --depth-tol > 0).")
    p.add_argument(
        "--depth-slab-front", type=float, default=0.15,
        help="Meters of tolerance IN FRONT of the bbox-estimated "
             "foreground depth (absorbs depth sensor noise).")
    p.add_argument(
        "--depth-slab-back", type=float, default=0.60,
        help="Meters of object thickness allowed BEHIND the bbox-"
             "estimated foreground depth.  Larger objects (couch) "
             "need bigger value; thin objects (poster) need smaller.")
    p.add_argument(
        "--invert-pose", action="store_true", default=False,
        help="Interpret the pose .txt matrix as world-to-camera instead "
             "of camera-to-world.  Standard ScanNet is cam-to-world; "
             "some forks / preprocessors swap it.  Turn on if the "
             "[pose-diag] log shows <5%% of points in front of camera, "
             "or the scene_proj_diag PLY is mostly grey/blue.")
    p.add_argument(
        "--estimate-pose", action="store_true", default=False,
        help="Estimate camera pose from the image using feature-PnP "
             "instead of loading from ScanNet sibling files.  For each "
             "Qwen 2D patch, find its top-K best-matching Utonia 3D "
             "points (via the same H-aligned 512-d common space we use "
             "for cosine localization), build 2D↔3D correspondences, "
             "solve PnP-RANSAC for (R, t).  Lets you run the demo on "
             "ARBITRARY images (phone photos, web shots) — no ScanNet "
             "pose required.  Implies --use-pose.  Needs cv2 (already "
             "a project dep).")
    p.add_argument(
        "--fov-deg", type=float, default=70.0,
        help="Approximate horizontal FOV in degrees for the intrinsic "
             "guess used by --estimate-pose.  ScanNet's StructureSensor "
             "≈58, most phone wide-angle ≈70-80, ultra-wide ≈100.")
    p.add_argument(
        "--pnp-top-k", type=int, default=3,
        help="Per Qwen 2D patch, build correspondences with the top-K "
             "3D points by feature cosine.  Higher = more outliers but "
             "more chances of finding a good RANSAC inlier set.")
    p.add_argument(
        "--pnp-reproj-err", type=float, default=8.0,
        help="RANSAC reprojection error threshold in pixels.  "
             "Correspondences whose projected 3D point lands within "
             "this distance of the 2D patch center count as inliers.")
    p.add_argument("--plot-max-points", type=int, default=80000,
                   help="Subsample cap for the Plotly 3D scatter "
                        "(browser perf gets bad past ~150k).")
    p.add_argument("--plot-point-size", type=int, default=2,
                   help="Marker size for Plotly 3D scatter.")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load scene point cloud --------------------------------------------
    coord = np.load(os.path.join(args.scene_dir, "coord.npy")).astype(np.float32)
    color = np.load(os.path.join(args.scene_dir, "color.npy")).astype(np.uint8)
    normal = np.load(os.path.join(args.scene_dir, "normal.npy")).astype(np.float32)
    print(f"[scene] {args.scene_dir}  N_points={len(coord)}")

    # --- Resolve image path (absolute or relative to scene root) -----------
    if not os.path.isabs(args.image_path):
        for prefix in [
            args.scene_dir,
            os.path.dirname(os.path.dirname(args.scene_dir.rstrip("/"))),
        ]:
            cand = os.path.join(prefix, args.image_path)
            if os.path.isfile(cand):
                args.image_path = cand
                break
    if not os.path.isfile(args.image_path):
        sys.exit(f"[error] image not found: {args.image_path}")
    print(f"[image] {args.image_path}")

    # --- Build models ------------------------------------------------------
    # Qwen: need the full LLM iff we're going to call .generate() for the
    # bbox-grounding path; otherwise we drop the LM body to save VRAM.
    qwen = build_qwen(
        args.qwen_path, device,
        keep_llm=(args.ground_mode == "qwen-bbox"),
    )
    h = build_h_modules(args.h_ckpt, device)
    siglip = (build_siglip(args.siglip_path, device)
              if args.ground_mode == "siglip" else None)

    # --- Image → Qwen 2D patches (post-merger, 2560-d) ---------------------
    image_tensor, image_pil = image_to_qwen_tensor(args.image_path, 512)
    image_tensor = image_tensor.to(device)
    patch_2d = qwen_vision_full_merger(qwen["visual"], image_tensor, device)
    h_grid, w_grid, D2 = patch_2d.shape
    print(f"[2D]  Qwen merged patches: {h_grid}x{w_grid} x {D2}")
    patch_2d_flat = patch_2d.reshape(-1, D2).float()  # (256, 2560)

    # --- Utonia 3D features (run NOW so pose estimation has them) --------
    # Moved up from later in the flow: --estimate-pose needs both
    # patch_common_all (Qwen patches → 512-d) and point_common_s1 +
    # coord_s1 (3D points → 512-d, with coords) to run feature-PnP
    # BEFORE frustum filtering.  Downstream cosine scoring just reuses
    # the already-computed point_common_s1 — no extra work.
    print("[3D]  running Utonia backbone ...")
    feat_s1, inv_s1_to_s0, inv_grid, coord_s1 = utonia_point_features(
        h["backbone"], coord, color, normal, device,
    )
    print(f"[3D]  stage-1 feat: {tuple(feat_s1.shape)}   "
          f"(stage-0 N={inv_s1_to_s0.shape[0]}, orig N={inv_grid.shape[0]})")

    if not torch.isfinite(feat_s1).all():
        n_bad = (~torch.isfinite(feat_s1)).any(dim=-1).sum().item()
        print(f"[warn] feat_s1 has {n_bad}/{feat_s1.shape[0]} non-finite rows; "
              "replacing with zeros.")
        feat_s1 = torch.nan_to_num(feat_s1, nan=0.0, posinf=0.0, neginf=0.0)

    with torch.inference_mode():
        point_common_s1 = h["patch_proj"](feat_s1.float())  # (N_s1, 512)
        patch_common_all = h["qwen_proj"](patch_2d_flat)    # (256, 512)
    if not torch.isfinite(point_common_s1).all():
        n_bad = (~torch.isfinite(point_common_s1)).any(dim=-1).sum().item()
        print(f"[warn] point_common_s1 has {n_bad} non-finite rows; "
              "replacing with zeros.")
        point_common_s1 = torch.nan_to_num(
            point_common_s1, nan=0.0, posinf=0.0, neginf=0.0,
        )

    # --- Build the 2D ROI mask from the chosen grounding source -----------
    # Two paths, same downstream:
    #   siglip   : external contrastive model gives a soft heatmap →
    #              percentile threshold → binary mask.
    #   qwen-bbox: Qwen LLM emits a bbox as text tokens → rasterize to
    #              a soft cell-area mask on the 16x16 grid.
    bbox_norm = None
    if args.ground_mode == "siglip":
        print(f"[siglip] computing text-image map for query='{args.query}' ...")
        sig_heat, h_sig, w_sig = siglip_grounding_map(
            siglip, image_pil, args.query, device,
        )
        print(f"[siglip] grid={h_sig}x{w_sig}  "
              f"heat range=[{sig_heat.min().item():.3f}, "
              f"{sig_heat.max().item():.3f}]")
        # Resample SigLIP heatmap onto Qwen's post-merger 16x16 grid so we
        # can use it to weight Qwen patches.
        qwen_heat = F.interpolate(
            sig_heat[None, None],
            size=(h_grid, w_grid), mode="bilinear", align_corners=False,
        )[0, 0]  # (h_grid, w_grid)
        thr = torch.quantile(qwen_heat.flatten(),
                             args.siglip_thr_percentile / 100)
        mask = (qwen_heat >= thr).float()
        print(f"[siglip] threshold @ p{args.siglip_thr_percentile:.0f} = "
              f"{thr.item():.3f}")
    else:  # qwen-bbox
        print(f"[qwen-bbox] grounding query='{args.query}' "
              "via Qwen LLM generation ...")
        bbox_norm, _gen = qwen_bbox_grounding(
            qwen, image_pil, args.query, device,
        )
        if bbox_norm is None:
            print("[qwen-bbox] failed to parse a bbox from generation. "
                  "Falling back to uniform mask over all patches "
                  "(comparison against siglip will not be meaningful).")
            mask = torch.ones(h_grid, w_grid, device=device)
        else:
            mask = bbox_to_grid_mask(bbox_norm, h_grid, w_grid, device)
        # Use the mask itself as the 'heatmap' for 2D visualization.
        qwen_heat = mask

    mask_flat = mask.flatten()
    n_kept = int((mask_flat > 0).sum().item())
    print(f"[mask] mode={args.ground_mode}  nonzero={n_kept}/{h_grid*w_grid}"
          f"  sum={mask_flat.sum().item():.2f}")

    # Output filename suffix — defined here (before the pose block) so the
    # scene_proj_diag PLY written from inside that block can use it too.
    # Both ground modes coexist in one out-dir for side-by-side compare.
    sfx = f"_{args.ground_mode}"

    # --- Optional: 2D bbox → 3D point mask, three possible sources ------
    #   --use-correspondence : use the precomputed correspondence/<frame>.npy
    #                          (the EXACT mapping H/I training used; built-in
    #                          occlusion handling).  Best when you have it.
    #   --use-pose           : load (K, T) and depth from ScanNet sibling
    #                          files, project bbox into 3D frustum, occlusion
    #                          via depth map.
    #   --estimate-pose      : recover (K_guess, T) via feature-PnP using
    #                          patch_common_all ↔ point_common_s1 + coord_s1.
    #                          Sets depth=None (occlusion check skipped).
    # The mask is applied to sim_3d *after* KNN smoothing so the smoothing
    # step isn't contaminated by clamped values.
    frustum_mask = None
    cam = None

    if args.use_correspondence:
        bbox_for_frustum = bbox_norm
        if bbox_for_frustum is None:
            bbox_for_frustum = bbox_from_grid_mask(mask, h_grid, w_grid)
            if bbox_for_frustum is not None:
                print(f"[corr] derived bbox from ROI mask: "
                      f"({bbox_for_frustum[0]:.3f}, {bbox_for_frustum[1]:.3f}, "
                      f"{bbox_for_frustum[2]:.3f}, {bbox_for_frustum[3]:.3f})")
        if bbox_for_frustum is None:
            print("[corr] no bbox available; skipping correspondence filter.")
        else:
            corr, corr_path = load_correspondence_for_frame(
                args.image_path, override_path=args.correspondence_path,
            )
            if corr is None:
                print("[corr] correspondence file not found; tried:")
                print(f"           {os.path.dirname(os.path.dirname(args.image_path))}"
                      "/correspondence/<frame>.npy and variants.")
                print("       Use --correspondence-path /abs/path or "
                      "fall back to --use-pose / --estimate-pose.")
            else:
                print(f"[corr] loaded: {corr_path}")
                print(f"[corr] correspondence shape: {corr.shape}  "
                      f"vs coord.npy: {coord.shape[0]} points")
                if corr.shape[0] != coord.shape[0]:
                    print(f"[corr] WARN: correspondence row count "
                          f"{corr.shape[0]} ≠ coord row count "
                          f"{coord.shape[0]} — mapping is likely wrong, "
                          "skipping correspondence filter.")
                else:
                    frustum_mask, n_valid = correspondence_bbox_mask(
                        corr, bbox_for_frustum,
                        patch_grid=args.correspondence_patch_grid,
                    )
                    n_kept = int(frustum_mask.sum())
                    n_total = len(coord)
                    print(f"[corr] visible in this frame: {n_valid}/"
                          f"{n_total} ({100*n_valid/n_total:.1f}%)")
                    print(f"[corr] inside bbox:           {n_kept}/"
                          f"{n_total} ({100*n_kept/n_total:.2f}%)")
                    # Debug PLY: red = inside bbox, yellow = visible but
                    # outside bbox, grey = not visible in this frame.
                    diag_colors = np.full(
                        (n_total, 3), 80, dtype=np.uint8,
                    )  # grey = (-1, -1)
                    valid_mask = (corr[:, 0] >= 0) & (corr[:, 1] >= 0)
                    diag_colors[valid_mask & ~frustum_mask] = \
                        np.array([220, 200, 60], dtype=np.uint8)  # yellow
                    diag_colors[frustum_mask] = \
                        np.array([230, 30, 30], dtype=np.uint8)   # red
                    write_ply(
                        os.path.join(args.out_dir,
                                     f"scene_proj_diag{sfx}.ply"),
                        coord, diag_colors,
                    )
                    print(f"[save] {args.out_dir}/scene_proj_diag{sfx}.ply  "
                          "(red=in bbox, yellow=visible elsewhere, "
                          "grey=hidden from this frame)")
                    if n_kept == 0:
                        print("[corr] EMPTY frustum — bbox doesn't overlap "
                              "any visible point in this frame.  Ignoring "
                              "filter.")
                        frustum_mask = None

    elif args.use_pose or args.estimate_pose:
        bbox_for_frustum = bbox_norm
        if bbox_for_frustum is None:
            bbox_for_frustum = bbox_from_grid_mask(mask, h_grid, w_grid)
            if bbox_for_frustum is not None:
                print(f"[pose] derived bbox from ROI mask: "
                      f"({bbox_for_frustum[0]:.3f}, {bbox_for_frustum[1]:.3f}, "
                      f"{bbox_for_frustum[2]:.3f}, {bbox_for_frustum[3]:.3f})")

        if bbox_for_frustum is None:
            print("[pose] no bbox available; skipping frustum filter.")
        elif args.estimate_pose:
            # --- feature-PnP pose recovery -----------------------------
            img_W, img_H = image_pil.size
            K_guess = make_default_intrinsic(img_W, img_H, args.fov_deg)
            # Patch centers in image-pixel coords: the 16×16 grid spans
            # the 512×512 resize that fed qwen_vision_full_merger.  Use
            # 512×512 as the centers' image size (NOT img_W × img_H) so
            # they match the geometry that produced patch_2d, then
            # rescale K to that resolution.
            K_for_pnp = make_default_intrinsic(512, 512, args.fov_deg)
            patch_centers = patch_grid_pixel_centers(
                h_grid, w_grid, image_W=512, image_H=512,
            )
            print(f"[pose-est] feature-PnP with fov={args.fov_deg:.1f}° "
                  f"top-k={args.pnp_top_k} reproj_err={args.pnp_reproj_err}px")
            try:
                pnp_out = estimate_pose_via_feature_pnp(
                    patch_common_all,
                    patch_centers,
                    K_for_pnp,
                    point_common_s1,
                    coord_s1.detach().cpu().numpy().astype(np.float32),
                    top_k=args.pnp_top_k,
                    ransac_reproj_err=args.pnp_reproj_err,
                )
            except Exception as e:
                print(f"[pose-est] PnP raised: {e}")
                pnp_out = None
            if pnp_out is None:
                print("[pose-est] PnP failed (too few inliers / bad "
                      "features); continuing WITHOUT frustum filter.")
            else:
                T_c2w, n_inliers, n_pairs = pnp_out
                print(f"[pose-est] success  "
                      f"inliers={n_inliers}/{n_pairs} "
                      f"({100*n_inliers/n_pairs:.1f}%)  "
                      f"t=({T_c2w[0,3]:+.2f},{T_c2w[1,3]:+.2f},"
                      f"{T_c2w[2,3]:+.2f})")
                # Treat the 512-px K as our color intrinsic since the
                # 2D bbox lives in normalized [0,1] anyway — the
                # bbox_to_frustum_mask multiplies by H_color/W_color.
                cam = dict(
                    K_color=K_for_pnp, K_depth=K_for_pnp,
                    T_c2w=T_c2w,
                    H_color=512, W_color=512,
                    depth=None, H_depth=None, W_depth=None,
                )
        else:
            # --- load from ScanNet siblings ----------------------------
            try:
                cam = load_camera_for_image(
                    args.image_path,
                    override_pose=args.pose_path,
                    override_intr_color=args.intrinsic_color_path,
                    override_intr_depth=args.intrinsic_depth_path,
                    override_depth=args.depth_path,
                )
                depth_note = (
                    f"depth={cam['W_depth']}x{cam['H_depth']}"
                    if cam.get("depth") is not None else "depth=N/A"
                )
                print(f"[pose] camera loaded  "
                      f"image={cam['W_color']}x{cam['H_color']}  "
                      f"K_color[fx,fy]=({cam['K_color'][0,0]:.1f},"
                      f"{cam['K_color'][1,1]:.1f})  {depth_note}")
            except (FileNotFoundError, RuntimeError) as e:
                print(f"[pose] failed: {e}")
                print("[pose] continuing WITHOUT frustum filter.")

        if cam is not None:
            frustum_mask, n_geo, n_dropped, stages, diag_info = \
                bbox_to_frustum_mask(
                    coord, bbox_for_frustum, cam,
                    depth_tol=args.depth_tol,
                    depth_slab_front=args.depth_slab_front,
                    depth_slab_back=args.depth_slab_back,
                    use_depth_slab=args.depth_slab,
                    invert_pose=args.invert_pose,
                    diag=True,
                )
            n_total = len(coord)
            n_kept_pose = int(frustum_mask.sum())
            if args.depth_tol > 0 and cam.get("depth") is not None:
                slab_note = ""
                if diag_info.get("depth_slab") is not None:
                    z_lo, z_hi = diag_info["depth_slab"]
                    slab_note = (f", {diag_info['n_slab_dropped']} dropped by "
                                 f"depth-slab[{z_lo:.2f}m,{z_hi:.2f}m]")
                print(f"[pose] frustum: {n_geo}/{n_total} inside bbox, "
                      f"{n_dropped} dropped by per-pixel occlusion"
                      f"{slab_note}, {n_kept_pose} kept "
                      f"({100*n_kept_pose/n_total:.2f}% of scene)")
            else:
                print(f"[pose] frustum: {n_kept_pose}/{n_total} inside "
                      f"bbox ({100*n_kept_pose/n_total:.2f}% of scene)")

            # Always dump the 4-color projection diagnostic PLY so the
            # user can visually verify pose alignment.
            #   red    = in bbox (final frustum mask candidates)
            #   yellow = in image but outside bbox
            #   blue   = in front of camera but outside image
            #   grey   = behind camera (z_cam <= 0)
            diag_colors = np.full((n_total, 3), 80, dtype=np.uint8)  # grey
            diag_colors[stages["in_front"] & ~stages["in_image"]] = \
                np.array([60, 110, 210], dtype=np.uint8)             # blue
            diag_colors[stages["in_image"] & ~stages["in_bbox"]] = \
                np.array([220, 200, 60], dtype=np.uint8)             # yellow
            diag_colors[stages["in_bbox"]] = \
                np.array([230, 30, 30], dtype=np.uint8)              # red
            write_ply(
                os.path.join(args.out_dir, f"scene_proj_diag{sfx}.ply"),
                coord, diag_colors,
            )
            print(f"[save] {args.out_dir}/scene_proj_diag{sfx}.ply  "
                  "(red=bbox, yellow=in image, blue=in front, grey=behind cam)")

            if n_kept_pose == 0:
                print("[pose] EMPTY frustum after depth check — "
                      "ignoring frustum filter for the rest of the run.")
                frustum_mask = None
                if not args.invert_pose:
                    print("       Hint: if scene_proj_diag is mostly grey "
                          "(behind-camera) or blue (off-image), try "
                          "`--invert-pose` — your pose .txt may store "
                          "world→cam instead of cam→world.")

    save_2d_overlay(
        image_pil, qwen_heat,
        os.path.join(args.out_dir, f"attn_2d{sfx}.png"),
        bbox_norm=bbox_norm,
        title_suffix=f" [{args.ground_mode}]",
    )
    print(f"[save] {args.out_dir}/attn_2d{sfx}.png")

    # --- Grounded query feature: pos = mean(ROI patches), neg = mean(rest)
    # The Qwen patches `patch_2d_flat` live in H's qwen_proj training
    # distribution.  The ROI mask says which patches are on the object.
    #   - q_pos : weighted mean inside ROI ("looks like the object")
    #   - q_neg : weighted mean outside ROI ("looks like the scene
    #             background of this image"), if --bg-subtract.
    # Both are projected via H's qwen_proj into the 512-d common space;
    # we score points later as cos(point, pos) - cos(point, neg), which
    # rewards features that are distinctly object-like rather than
    # generally "indoor-scene-like".
    if mask_flat.sum() <= 0:
        print("[warn] ROI mask empty — falling back to uniform pos query.")
        pos_weights = torch.ones_like(mask_flat) / mask_flat.numel()
    else:
        # --roi-erode: drop cells with low partial coverage to build a
        # purer q_pos.  Bbox-boundary cells often mix object + background
        # pixels at the Qwen 32x32-px resolution, so the boundary patches
        # carry a "half-object half-scene" feature that pulls q_pos toward
        # generic-indoor and weakens the 3D contrast.  Higher threshold =
        # tighter (smaller effective ROI but purer signal).
        if args.roi_erode > 0:
            eroded = torch.where(
                mask_flat >= args.roi_erode,
                mask_flat, torch.zeros_like(mask_flat),
            )
            kept = int((eroded > 0).sum().item())
            raw = int((mask_flat > 0).sum().item())
            if eroded.sum() > 0:
                pos_weights = eroded / eroded.sum()
                print(f"[roi] erode>={args.roi_erode}: "
                      f"{kept}/{raw} cells kept for q_pos")
            else:
                pos_weights = mask_flat / mask_flat.sum()
                print(f"[roi] erode>={args.roi_erode} removed all cells; "
                      "falling back to raw mask.")
        else:
            pos_weights = mask_flat / mask_flat.sum()
    q_pos_2d = (pos_weights.unsqueeze(-1) * patch_2d_flat).sum(dim=0)  # (2560,)

    q_neg_2d = None
    if args.bg_subtract:
        neg_mask = 1.0 - mask_flat
        if float(neg_mask.sum().item()) > 0:
            neg_weights = neg_mask / neg_mask.sum()
            q_neg_2d = (neg_weights.unsqueeze(-1) * patch_2d_flat).sum(dim=0)
        else:
            print("[bg-subtract] mask covers entire image; "
                  "no patches available for background — disabling.")

    with torch.inference_mode():
        query_common = h["qwen_proj"](q_pos_2d.unsqueeze(0).float())[0]  # (512,)
        if q_neg_2d is not None:
            query_neg_common = h["qwen_proj"](q_neg_2d.unsqueeze(0).float())[0]
        else:
            query_neg_common = None
    print(f"[query] pos.shape={tuple(query_common.shape)}, "
          f"pos.norm={query_common.norm().item():.3f}"
          + (f"  neg.norm={query_neg_common.norm().item():.3f}"
             if query_neg_common is not None else "  (no bg-subtract)"))

    # (feat_s1, coord_s1, point_common_s1 already computed earlier so
    # pose estimation could use them.)
    if not torch.isfinite(query_common).all():
        print("[warn] query_common has non-finite values. Replacing.")
        query_common = torch.nan_to_num(query_common, nan=0.0,
                                         posinf=0.0, neginf=0.0)

    # --- Cosine sim at stage-1 resolution --------------------------------
    # With --bg-subtract:  score = cos(point, q_pos) - cos(point, q_neg)
    # Without:             score = cos(point, q_pos)   (legacy)
    # Subtracting cosines (rather than the raw 2560-d query vectors before
    # qwen_proj) is the right place to do this: each query goes through
    # the LayerNorm-containing qwen_proj naturally, and the difference is
    # taken in the scoring space where it semantically means "shift the
    # ranking toward points that out-cosine the background".
    pcn = F.normalize(point_common_s1.float(), dim=-1, eps=1e-6)
    qcn = F.normalize(query_common.float(), dim=-1, eps=1e-6)
    sim_pos = (pcn @ qcn)  # (N_s1,)
    if query_neg_common is not None:
        qnc = F.normalize(query_neg_common.float(), dim=-1, eps=1e-6)
        sim_neg = (pcn @ qnc)
        sim_s1 = sim_pos - sim_neg
        print(f"[sim] bg-subtracted  "
              f"pos=[{sim_pos.min().item():.3f},{sim_pos.max().item():.3f}]  "
              f"neg=[{sim_neg.min().item():.3f},{sim_neg.max().item():.3f}]  "
              f"diff=[{sim_s1.min().item():.3f},{sim_s1.max().item():.3f}] "
              f"mean={sim_s1.mean().item():.3f} std={sim_s1.std().item():.3f}")
    else:
        sim_s1 = sim_pos
        print(f"[sim] raw cos  "
              f"range=[{sim_s1.min().item():.3f},{sim_s1.max().item():.3f}] "
              f"mean={sim_s1.mean().item():.3f} std={sim_s1.std().item():.3f}")
    sim_s1 = torch.nan_to_num(sim_s1, nan=0.0, posinf=0.0, neginf=0.0)

    # Broadcast stage-1 sim → stage-0 grid via the stage-0→stage-1
    # pooling_inverse → original input points via the GridSample
    # inverse.  At each step we just gather, so values stay in [-1, 1].
    sim_s0 = sim_s1[inv_s1_to_s0]            # (N_s0,)
    sim_orig = sim_s0[inv_grid].cpu().numpy() # (N_input,)
    sim_3d = sim_orig
    print(f"[3D]  sim range: [{sim_3d.min():.3f}, {sim_3d.max():.3f}], "
          f"mean={sim_3d.mean():.3f}, std={sim_3d.std():.3f}")

    # --- KNN spatial smoothing of per-point scores -----------------------
    # Reduces "noisy hot specks" that come from individual points sitting
    # on feature-manifold ambiguities (e.g. a chair leg pixel that happens
    # to project cosine-close to a fridge edge).  Coherent object-shaped
    # responses survive because every neighbor also scores high.
    if args.smooth_knn > 1:
        before_std = float(sim_3d.std())
        sim_3d = knn_smooth_scores(coord, sim_3d, k=args.smooth_knn)
        after_std = float(sim_3d.std())
        print(f"[smooth] knn={args.smooth_knn}  "
              f"std {before_std:.4f} → {after_std:.4f}  "
              f"range=[{sim_3d.min():.3f}, {sim_3d.max():.3f}]")

    # --- Pose-based frustum gating ---------------------------------------
    # Hard geometric constraint: clamp outside-frustum points to a value
    # below ANY inside-frustum score, so top-percentile / clustering /
    # top-K downstream all naturally restrict to the bbox cone.  Coupled
    # with depth occlusion (if available), this gives single-instance
    # extraction directly — the cosine feature score then only resolves
    # depth-ambiguity within the same ray bundle.
    if frustum_mask is not None and frustum_mask.any():
        inside_sim = sim_3d[frustum_mask]
        pin_value = float(inside_sim.min()) - 1.0
        sim_3d = np.where(frustum_mask, sim_3d, pin_value)
        print(f"[pose] gated  inside-frustum sim "
              f"range=[{inside_sim.min():.3f}, {inside_sim.max():.3f}] "
              f"mean={inside_sim.mean():.3f} std={inside_sim.std():.3f}  "
              f"outside pinned to {pin_value:.3f}")
        # Debug PLY: red inside frustum, grey outside.  Lets you sanity-
        # check whether the pose/intrinsics correctly hit the object.
        fr_colors = np.where(
            frustum_mask[:, None],
            np.array([[230, 30, 30]], dtype=np.uint8),
            np.array([[160, 160, 160]], dtype=np.uint8),
        )
        write_ply(os.path.join(args.out_dir, f"scene_frustum{sfx}.ply"),
                  coord, fr_colors)
        print(f"[save] {args.out_dir}/scene_frustum{sfx}.ply  "
              "(red=inside frustum, grey=outside)")

    # --- Render PLYs ------------------------------------------------------
    # (a) rgb for orientation — shared across modes (overwrite OK).
    write_ply(os.path.join(args.out_dir, "scene_rgb.ply"), coord, color)

    # (b) heatmap with top-percentile masking.
    #
    # The old "percentile [5, 99] stretch on the whole cloud" approach
    # mapped 94% of points into jet's warm zone whenever the score
    # distribution was narrow — visually indistinguishable from "the
    # whole scene matches".  Instead: color only the top X% with a jet
    # scale stretched OVER THAT TOP SUBSET, and grey out everything
    # below — so the object pops, the rest provides context.
    if args.top_percentile < 100.0:
        score_thr = float(np.percentile(sim_3d, 100.0 - args.top_percentile))
        above = sim_3d >= score_thr
        n_above = int(above.sum())
        print(f"[viz] top-{args.top_percentile:.1f}% threshold={score_thr:.4f}"
              f"  ({n_above}/{len(sim_3d)} colored, rest grey)")
    else:
        above = np.ones_like(sim_3d, dtype=bool)
        n_above = len(sim_3d)
        score_thr = float(sim_3d.min())

    if n_above > 0:
        above_lo = float(sim_3d[above].min())
        above_hi = float(sim_3d[above].max())
        sim_norm = np.clip(
            (sim_3d - above_lo) / max(above_hi - above_lo, 1e-9), 0, 1,
        )
    else:
        sim_norm = np.zeros_like(sim_3d)

    colors_above = jet_colormap(sim_norm).astype(np.uint8)
    colors_grey = np.full_like(colors_above, 110)  # mid-grey context
    hot = np.where(above[:, None], colors_above, colors_grey)
    write_ply(os.path.join(args.out_dir, f"scene_heatmap{sfx}.ply"),
              coord, hot)
    print(f"[save] {args.out_dir}/scene_heatmap{sfx}.ply")

    # (c) top-K most similar points only
    k = min(args.top_k, len(sim_3d))
    idx_top = np.argpartition(-sim_3d, k - 1)[:k]
    write_ply(os.path.join(args.out_dir, f"scene_top{sfx}.ply"),
              coord[idx_top], np.array([[230, 30, 30]] * k, dtype=np.uint8))
    print(f"[save] {args.out_dir}/scene_top{sfx}.ply  (top {k} pts)")

    # (c2) single-instance extraction via spatial clustering.
    # Among the top-(top_percentile)% scoring points, find the largest
    # connected component — drops scattered look-alike responses across
    # the scene and isolates one object instance.
    if args.cluster_eps > 0:
        cluster_thr = float(np.percentile(sim_3d, 100.0 - args.top_percentile))
        cand_idx = np.where(sim_3d >= cluster_thr)[0]
        if len(cand_idx) >= 3:
            local_cluster = largest_connected_cluster(
                coord[cand_idx], eps=args.cluster_eps,
            )
            obj_idx = cand_idx[local_cluster]
            obj_color = np.array(
                [[30, 200, 30]] * len(obj_idx), dtype=np.uint8,
            )
            write_ply(
                os.path.join(args.out_dir, f"scene_object{sfx}.ply"),
                coord[obj_idx], obj_color,
            )
            print(f"[cluster] eps={args.cluster_eps}m  "
                  f"top-{args.top_percentile:.0f}% candidates={len(cand_idx)}  "
                  f"largest cluster={len(obj_idx)} pts "
                  f"({100.0 * len(obj_idx) / max(len(cand_idx), 1):.1f}% of "
                  "candidates)")
            print(f"[save] {args.out_dir}/scene_object{sfx}.ply  "
                  "(largest spatially connected cluster)")
        else:
            print(f"[cluster] only {len(cand_idx)} candidates above threshold; "
                  "skipping.")

    # (d) interactive Plotly HTML — 2D overlay + 3D heatmap together.
    # The 2D heat is whichever ROI source we used; qwen-bbox additionally
    # draws the parsed rectangle in green on the overlay.
    html_path = os.path.join(args.out_dir, f"viz{sfx}.html")
    save_plotly_combined(
        image_pil=image_pil,
        attn_grid=qwen_heat,
        coord=coord,
        color_rgb=color,
        sim_3d=sim_3d,
        query_text=args.query,
        out_path=html_path,
        max_points=args.plot_max_points,
        point_size=args.plot_point_size,
        bbox_norm=bbox_norm,
        mode_label=args.ground_mode,
        top_percentile=args.top_percentile,
    )
    print(f"[save] {html_path}  "
          "(open in browser — left: 2D attention, right: 3D heatmap, "
          "drag to rotate)")

    print("\n[done]")


if __name__ == "__main__":
    main()
