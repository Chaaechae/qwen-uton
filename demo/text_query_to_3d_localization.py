"""
Text-query → 3D localization demo on a H-aligned Utonia scene.

Idea
----
H trained Utonia so that the 3D backbone (post patch_proj) and Qwen's
2D vision tokens (post qwen_proj) live in the SAME 512-d common space.
That lets us route a NL query like "냉장고" / "refrigerator" through
Qwen's vision-language path and land it in 3D space with no per-scene
fine-tuning:

  text query  ─────► Qwen tokenizer + embed_tokens ─► query_text  (2560-d)
                                                          │
  scene image ─────► Qwen ViT (+ merger)             ─► patch_2d   (256, 2560)
                              │
                              ▼
            softmax(cos(query_text, patch_2d) / τ)  ─► attn_2d   (256,)
                              │
                              ▼
              Σ attn_i · patch_2d_i                 ─► query_grounded (2560-d)
                              │
                              ▼              ┌─────  qwen_proj  ─► query_common (512)
                                              │                            │
  scene point cloud ─► Utonia backbone     ─► point_3d (N, 1332)            │
                                                          │                │
                                                          ▼                │
                                                  patch_proj          ─► point_common (N, 512)
                                                                           │
                                                                           ▼
                                          cosine(point_common, query_common)
                                                          │
                                                          ▼
                                          3D heatmap (PLY) + top-K points

Why the "grounded" step (image-attention weighted average)
-----------------------------------------------------------
The raw text embedding from Qwen's LLM (`embed_tokens(token_id)`) lives
in the LLM token space — its statistical distribution differs from the
post-merger image patch distribution that H trained `qwen_proj` on.
By weighting the IMAGE-side patches with text-image cosine and
averaging, we get a query vector that's in the SAME distribution as
H's training-time 2D side — `qwen_proj` lands it cleanly in common.

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

Outputs
-------
  <out>/attn_2d.png       Static 2D matplotlib overlay (input image
                          side-by-side with attention heatmap).
  <out>/scene_rgb.ply     Original-color point cloud (for orientation).
  <out>/scene_heatmap.ply Per-point similarity heatmap (jet colormap).
  <out>/scene_top.ply     Top-K most similar points isolated (K = 1024
                          default; --top-k to change).
  <out>/viz.html          ★ Single interactive Plotly page:
                          LEFT  — input image with attention overlay.
                          RIGHT — 3D point cloud, orbit/zoom/rotate,
                                  hover shows (x, y, z, similarity).
                          Browser-renderable, no extra deps; subsamples
                          to --plot-max-points (default 80000) when the
                          cloud is large.

Usage
-----
  python demo/text_query_to_3d_localization.py \\
      --scene-dir /group-volume/3Ddataset/data/scannet/val/scene0011_00 \\
      --image-path images/val/scene0011_00/color/12.png \\
      --query "refrigerator" \\
      --h-ckpt exp/utonia_q35_h/model/model_last.pth \\
      --qwen-path /group-volume/chaewon.yun/Qwen3.5-4B \\
      --out-dir exp/demo/scene0011_00_refrigerator
"""

import argparse
import os
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
        enable_flash=True, upcast_attention=False, upcast_softmax=False,
        enc_mode=True, traceable=False, mask_token=False,
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
# Qwen3.5-VL — token embedding (text side) + vision tower (image side).
# Uses transformers.AutoModelForImageTextToText, same as H training.
# ---------------------------------------------------------------------------
def build_qwen(qwen_path, device):
    print(f"[load] Qwen: {qwen_path}")
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
    # Grab the embed_tokens layer + visual tower; free the rest.
    embed_tokens = model.get_input_embeddings()
    visual = model.model.visual
    embed_tokens = embed_tokens.to(device).eval()
    visual = visual.to(device).eval()
    # Drop the LLM body to save VRAM — we only need embed_tokens + visual.
    del model
    torch.cuda.empty_cache()
    return dict(tokenizer=tokenizer, embed_tokens=embed_tokens,
                visual=visual, processor=processor)


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
    """coord/color/normal: numpy (N, 3).  Returns (N_valid, 1332) tensor."""
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
    inv = pd["inverse"].clone()
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
    # H's enc2d_upcast_level = 3 — pop pooling_parent 3 times so the
    # output dim matches what patch_proj expects (1332-d).
    for _ in range(3):
        if "pooling_parent" not in point.keys():
            break
        parent = point.pop("pooling_parent")
        invp = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[invp]], dim=-1)
        point = parent
    grid_feat = point.feat  # (N_grid, 1332)
    return grid_feat[inv]  # (N_orig, 1332)


# ---------------------------------------------------------------------------
# Save the 2D attention as an overlay PNG.
# ---------------------------------------------------------------------------
def save_2d_overlay(image_pil, attn_grid, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    ax[1].set_title(f"2D query attention")
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
    max_points=80000, point_size=2,
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

    # Percentile-stretch for visibility (otherwise tail outliers compress
    # the useful range into a thin band of color).
    lo, hi = np.percentile(sim_s, [5, 99])
    sim_norm = np.clip((sim_s - lo) / max(hi - lo, 1e-9), 0, 1)

    # ---------- Build subplots --------------------------------------------
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "image"}, {"type": "scene"}]],
        column_widths=[0.4, 0.6],
        subplot_titles=(f"2D attention  (query='{query_text}')",
                        "3D point cloud — heatmap"),
        horizontal_spacing=0.04,
    )

    fig.add_trace(go.Image(z=overlay), row=1, col=1)

    # 3D scatter with jet colorbar.
    fig.add_trace(
        go.Scatter3d(
            x=coord_s[:, 0], y=coord_s[:, 1], z=coord_s[:, 2],
            mode="markers",
            marker=dict(
                size=point_size,
                color=sim_norm,
                colorscale="Jet",
                cmin=0.0, cmax=1.0,
                showscale=True,
                colorbar=dict(
                    title=dict(text="sim(query, 3D)",
                               side="right"),
                    thickness=14, len=0.7, x=1.02,
                ),
                opacity=0.95,
            ),
            hovertemplate=(
                "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}"
                "<br>sim: %{marker.color:.3f}<extra></extra>"
            ),
            name="points",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=f"Text-query 3D localization — '{query_text}'",
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
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--temperature", type=float, default=0.07,
                   help="Softmax temperature for 2D attention map.")
    p.add_argument("--top-k", type=int, default=1024,
                   help="# of top-scored points to isolate in scene_top.ply")
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
    qwen = build_qwen(args.qwen_path, device)
    h = build_h_modules(args.h_ckpt, device)

    # --- Image → Qwen 2D patches (post-merger, 2560-d) ---------------------
    image_tensor, image_pil = image_to_qwen_tensor(args.image_path, 512)
    image_tensor = image_tensor.to(device)
    patch_2d = qwen_vision_full_merger(qwen["visual"], image_tensor, device)
    h_grid, w_grid, D2 = patch_2d.shape
    print(f"[2D]  Qwen merged patches: {h_grid}x{w_grid} x {D2}")
    patch_2d_flat = patch_2d.reshape(-1, D2).float()  # (256, 2560)

    # --- Text query → token embeddings → averaged 2560-d vector -----------
    token_ids = qwen["tokenizer"](args.query, return_tensors="pt",
                                   add_special_tokens=False).input_ids[0].to(device)
    with torch.inference_mode():
        text_embs = qwen["embed_tokens"](token_ids).float()  # (T, 2560)
    text_query = text_embs.mean(dim=0)  # (2560,)
    print(f"[text] query='{args.query}'  tokens={len(token_ids)}  "
          f"emb.shape={tuple(text_embs.shape)}")

    # --- 2D attention: cosine(text_query, patch_2d) → softmax -------------
    text_n = F.normalize(text_query, dim=-1)
    patch_n = F.normalize(patch_2d_flat, dim=-1)
    cos_2d = patch_n @ text_n  # (256,)
    attn_2d = F.softmax(cos_2d / args.temperature, dim=0)
    print(f"[2D]  attn max={attn_2d.max().item():.4f}, "
          f"min={attn_2d.min().item():.4e}, "
          f"entropy={(-attn_2d * (attn_2d.clamp_min(1e-9)).log()).sum().item():.2f}")
    save_2d_overlay(image_pil,
                    attn_2d.view(h_grid, w_grid),
                    os.path.join(args.out_dir, "attn_2d.png"))
    print(f"[save] {args.out_dir}/attn_2d.png")

    # --- Grounded query feature: attention-weighted avg of 2D patches -----
    # patch_2d_flat is already in the SAME distribution H's qwen_proj
    # was trained against (post-merger 2560-d).  Averaging by attention
    # picks the image-relevant subset → cleaner qwen_proj input than
    # raw token embedding.
    query_grounded = (attn_2d.unsqueeze(-1) * patch_2d_flat).sum(dim=0)  # (2560,)
    with torch.inference_mode():
        query_common = h["qwen_proj"](query_grounded.unsqueeze(0).float())[0]  # (512,)
    print(f"[query] common.shape={tuple(query_common.shape)}, "
          f"norm={query_common.norm().item():.3f}")

    # --- Scene → Utonia → patch_proj → 512-d common -----------------------
    print("[3D]  running Utonia backbone ...")
    feat_3d = utonia_point_features(h["backbone"], coord, color, normal, device)
    print(f"[3D]  per-point feat: {tuple(feat_3d.shape)}")

    # NaN/Inf sanity right after the backbone — catch a broken (collapsed)
    # checkpoint before it crashes the downstream cublas matmul. cublas
    # CUBLAS_STATUS_EXECUTION_FAILED in the sgemv at sim_3d=pcn@qcn is
    # almost always upstream NaN propagating here.
    if not torch.isfinite(feat_3d).all():
        n_bad = (~torch.isfinite(feat_3d)).any(dim=-1).sum().item()
        print(f"[warn] feat_3d has {n_bad}/{feat_3d.shape[0]} non-finite rows "
              "(checkpoint may be collapsed). Replacing with zeros.")
        feat_3d = torch.nan_to_num(feat_3d, nan=0.0, posinf=0.0, neginf=0.0)

    with torch.inference_mode():
        point_common = h["patch_proj"](feat_3d.float())  # (N, 512)

    if not torch.isfinite(point_common).all():
        n_bad = (~torch.isfinite(point_common)).any(dim=-1).sum().item()
        print(f"[warn] point_common (post patch_proj) has {n_bad} non-finite "
              "rows. Replacing with zeros.")
        point_common = torch.nan_to_num(point_common, nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(query_common).all():
        print("[warn] query_common has non-finite values. Replacing.")
        query_common = torch.nan_to_num(query_common, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Cosine sim 3D points vs query in common space -------------------
    # Safer eps so all-zero rows produce zero vectors (cos=0) rather than
    # NaN that would tank the cublasSgemv at sim = pcn @ qcn.
    pcn = F.normalize(point_common.float(), dim=-1, eps=1e-6)
    qcn = F.normalize(query_common.float(), dim=-1, eps=1e-6)
    sim_3d = (pcn @ qcn).cpu().numpy()  # (N,)
    sim_3d = np.nan_to_num(sim_3d, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[3D]  sim range: [{sim_3d.min():.3f}, {sim_3d.max():.3f}], "
          f"mean={sim_3d.mean():.3f}")

    # --- Render PLYs ------------------------------------------------------
    # (a) rgb for orientation
    write_ply(os.path.join(args.out_dir, "scene_rgb.ply"), coord, color)

    # (b) heatmap colored by sim (after percentile-stretch for visibility)
    sim_lo, sim_hi = np.percentile(sim_3d, [5, 99])
    sim_norm = np.clip((sim_3d - sim_lo) / max(sim_hi - sim_lo, 1e-9), 0, 1)
    hot = jet_colormap(sim_norm).astype(np.uint8)
    write_ply(os.path.join(args.out_dir, "scene_heatmap.ply"), coord, hot)
    print(f"[save] {args.out_dir}/scene_heatmap.ply")

    # (c) top-K most similar points only
    k = min(args.top_k, len(sim_3d))
    idx_top = np.argpartition(-sim_3d, k - 1)[:k]
    write_ply(os.path.join(args.out_dir, "scene_top.ply"),
              coord[idx_top], np.array([[230, 30, 30]] * k, dtype=np.uint8))
    print(f"[save] {args.out_dir}/scene_top.ply  (top {k} pts)")

    # (d) interactive Plotly HTML — 2D overlay + 3D heatmap together
    html_path = os.path.join(args.out_dir, "viz.html")
    save_plotly_combined(
        image_pil=image_pil,
        attn_grid=attn_2d.view(h_grid, w_grid),
        coord=coord,
        color_rgb=color,
        sim_3d=sim_3d,
        query_text=args.query,
        out_path=html_path,
        max_points=args.plot_max_points,
        point_size=args.plot_point_size,
    )
    print(f"[save] {html_path}  "
          "(open in browser — left: 2D attention, right: 3D heatmap, "
          "drag to rotate)")

    print("\n[done]")


if __name__ == "__main__":
    main()
