"""
Stage-1 -> Stage-2 demo: localize an object in the 3D map from a single 2D image,
WITHOUT GT correspondence and WITHOUT pose/depth for the lift.

This is the real pipeline the feasibility probe (probe_2d3d_alignment.py) validated.
The probe selected the object's DINO patches using ground-truth correspondence to
isolate *alignment* quality; here Stage-1 is image-only:

    Stage-1 (2D, no GT): get the object's DINO patches from the image
        --mode box    : an explicit pixel box (from any 2D detector, incl. Qwen-VL)
        --mode json   : boxes from a detector dump (qwen-uton); pick by --text label
        --mode point  : a clicked pixel, grown by DINO self-similarity
        --mode text   : open-vocab heatmap via open_clip (optional dep; experimental)
    Stage-2 (2D->3D)  : cosine-match the selected DINO feature(s) to the
        DINO-aligned 3D map features (patch_proj space) -> per-point score.

Outputs: a heatmap point cloud (<out>.ply), top-K point indices (<out>_topk.npy),
and -- if --eval_instance is given -- AP/IoU vs that GT instance, so you can read
the degradation from the probe's GT-surrogate numbers to an image-only Stage-1.

Run (no conda activate; repo root on sys.path automatically):
    python tools/viewpoint_probe/localize_from_2d.py \
        --scene_dir /group-volume/3Ddataset/data/scannet/val/scene0011_00 \
        --frame 300 --mode point --point 640 360 --out /tmp/loc

Reuses model/feature code from probe_2d3d_alignment.py.
"""

import argparse
import os
import sys
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn.functional as F

import probe_2d3d_alignment as P   # backbone/patch_proj/dino/F3D/dino-patch helpers


def select_patches_box(box, W, H):
    """box=(x0,y0,x1,y1) in image pixels -> DINO patch indices whose center is inside."""
    x0, y0, x1, y1 = box
    sx, sy = P.CROP / W, P.CROP / H
    idx = []
    for pv in range(P.PATCH_HW):
        for pu in range(P.PATCH_HW):
            cx = (pu + 0.5) * P.PATCH_SIZE / sx       # patch center back in image px
            cy = (pv + 0.5) * P.PATCH_SIZE / sy
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                idx.append(pv * P.PATCH_HW + pu)
    return np.asarray(idx, dtype=np.int64)


def select_patches_point(F2D, point, W, H, tau):
    """Clicked pixel -> its patch, grown by DINO self-similarity (cosine > tau)."""
    sx, sy = P.CROP / W, P.CROP / H
    pu = int(np.clip(point[0] * sx / P.PATCH_SIZE, 0, P.PATCH_HW - 1))
    pv = int(np.clip(point[1] * sy / P.PATCH_SIZE, 0, P.PATCH_HW - 1))
    p0 = pv * P.PATCH_HW + pu
    sims = (F2D @ F2D[p0]).detach().cpu().numpy()
    idx = np.where(sims > tau)[0]
    return idx if len(idx) else np.array([p0], dtype=np.int64)


def select_patches_text(image_chw_unused, text, rgb, device, tau):
    """Open-vocab via open_clip dense features (EXPERIMENTAL; needs `pip install open_clip_torch`).
    Returns DINO patch indices over the same 37x37 grid by thresholding a CLIP
    text-patch cosine heatmap resampled to 37x37."""
    import open_clip
    import torchvision.transforms.functional as TF
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    tok = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(device).eval()
    H, W = rgb.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    x = TF.resize(x, [224, 224], antialias=True)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    x = ((x - mean) / std).unsqueeze(0).to(device)
    with torch.inference_mode():
        v = model.visual
        tokens = v.trunk.forward_features(x) if hasattr(v, "trunk") else None
        if tokens is None:                       # open_clip native ViT
            feats = _openclip_dense(v, x)        # [1, n_patch, C]
        else:
            feats = tokens
        feats = feats[:, -((224 // 16) ** 2):, :]
        feats = F.normalize(feats[0], dim=-1)    # [196, C]
        t = F.normalize(model.encode_text(tok([text]).to(device)).float(), dim=-1)[0]
        heat = (feats @ t).reshape(14, 14).detach().cpu().numpy()
    heat = (heat - heat.min()) / (heat.ptp() + 1e-6)
    # resample 14x14 -> 37x37 grid, threshold
    grid = np.kron(heat, np.ones((3, 3)))[:P.PATCH_HW, :P.PATCH_HW]
    return np.where(grid.reshape(-1) > tau)[0]


def _openclip_dense(visual, x):
    """Best-effort dense token extraction from an open_clip ViT (no attention pool)."""
    v = visual
    z = v.conv1(x)
    z = z.reshape(z.shape[0], z.shape[1], -1).permute(0, 2, 1)
    cls = v.class_embedding.to(z.dtype) + torch.zeros(z.shape[0], 1, z.shape[-1], device=z.device, dtype=z.dtype)
    z = torch.cat([cls, z], dim=1) + v.positional_embedding.to(z.dtype)
    z = v.ln_pre(z)
    z = z.permute(1, 0, 2)
    z = v.transformer(z)
    z = z.permute(1, 0, 2)
    z = v.ln_post(z)
    if v.proj is not None:
        z = z @ v.proj
    return z


def write_ply(path, coord, color01):
    color = (np.clip(color01, 0, 1) * 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(coord)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), (r, g, b) in zip(coord, color):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone_ckpt",
                    default="/group-volume/Utonia/pretrain-utonia-v1m1-0-base-stagev2.pth")
    ap.add_argument("--patch_proj_ckpt", default=None)
    ap.add_argument("--scene_dir",
                    default="/group-volume/3Ddataset/data/scannet/val/scene0011_00")
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--frame", default="0", help="scene frame id used as the 2D query image")
    ap.add_argument("--image", default=None, help="external query image (overrides --frame)")
    ap.add_argument("--mode", choices=["box", "json", "point", "text"], default="point")
    ap.add_argument("--box", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--point", type=float, nargs=2, default=None, metavar=("X", "Y"))
    ap.add_argument("--text", default=None, help="label for --mode text/json")
    ap.add_argument("--boxes_json", default=None,
                    help="detector dump: {frame_id: [{label, box:[x0,y0,x1,y1]}, ...]}")
    ap.add_argument("--tau", type=float, default=0.6, help="self-sim / heatmap threshold")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=2000)
    ap.add_argument("--eval_instance", type=int, default=None,
                    help="GT instance id to score AP/IoU against (sanity vs the probe)")
    ap.add_argument("--out", default="/tmp/loc")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = P.load_backbone(args.backbone_ckpt, device)
    patch_proj = P.load_patch_proj(args.patch_proj_ckpt or args.backbone_ckpt, device)
    dino = P.load_dino(device)
    F3D, scene = P.compute_scene_F3D(args.scene_dir, model, patch_proj, device, args.scale)
    coord = scene["coord"]
    print(f"[F3D] {tuple(F3D.shape)}")

    # load the query image
    import imageio.v2 as imageio
    if args.image:
        rgb = imageio.imread(args.image)
    else:
        image_dir = args.image_dir or P.default_image_dir(args.scene_dir)
        rgb = imageio.imread(os.path.join(image_dir, "color", f"{args.frame}.png"))
    H, W = rgb.shape[:2]
    img_t, sx, sy = P.dino_preprocess(rgb)
    F2D = F.normalize(P.extract_dino_patches(dino, img_t, device), dim=-1)

    # Stage-1: image-only patch selection
    if args.mode == "box":
        assert args.box, "--box required"
        kp = select_patches_box(args.box, W, H)
    elif args.mode == "json":
        assert args.boxes_json and args.text, "--boxes_json and --text required"
        dets = json.load(open(args.boxes_json)).get(str(args.frame), [])
        cand = [d for d in dets if args.text.lower() in str(d.get("label", "")).lower()]
        assert cand, f"no '{args.text}' box for frame {args.frame} in {args.boxes_json}"
        kp = select_patches_box(cand[0]["box"], W, H)
    elif args.mode == "point":
        assert args.point, "--point required"
        kp = select_patches_point(F2D, args.point, W, H, args.tau)
    else:  # text
        kp = select_patches_text(img_t, args.text, rgb, device, args.tau)
    assert len(kp) > 0, "Stage-1 selected no patches"
    print(f"[stage-1] mode={args.mode} -> {len(kp)} DINO patches")

    # Stage-2: cosine match to the DINO-aligned 3D map
    q = F.normalize(F2D[torch.as_tensor(kp, device=device)].mean(0), dim=0)
    scores = (F3D @ q).detach().cpu().numpy()
    topk = np.argsort(-scores)[: args.topk]
    np.save(args.out + "_topk.npy", topk)

    import matplotlib.cm as cm
    s01 = (scores - scores.min()) / (scores.ptp() + 1e-6)
    write_ply(args.out + ".ply", coord, cm.get_cmap("Spectral_r")(s01)[:, :3])
    print(f"[out] {args.out}.ply  {args.out}_topk.npy")

    if args.eval_instance is not None:
        gt = (scene["instance"] == args.eval_instance)
        m = P.retrieval_metrics(scores, gt)
        rand = (F3D @ F.normalize(torch.randn(P.DINO_DIM, device=device), dim=0)
                ).detach().cpu().numpy()
        mr = P.retrieval_metrics(rand, gt)
        print(f"[eval inst {args.eval_instance}] AP={m['AP']:.3f} IoU={m['best_IoU']:.3f} "
              f"prec@100={m['prec@100']:.3f} | chance AP={mr['AP']:.3f} "
              f"(lift {m['AP'] / max(mr['AP'], 1e-6):.1f}x)")


if __name__ == "__main__":
    main()
