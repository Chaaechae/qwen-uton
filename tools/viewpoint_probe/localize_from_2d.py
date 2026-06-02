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

Outputs: a self-contained plotly heatmap (<out>.html, open in a browser), top-K
point indices (<out>_topk.npy), and -- if --eval_instance is given -- AP/IoU vs that
GT instance, so you can read the degradation from the probe's GT-surrogate numbers
to an image-only Stage-1.

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


def patches_with_instance_pixels(corr, instance, inst_id, sx, sy, N):
    """Pixel-accurate (mask-level) patch set: patches that actually contain inst_id
    pixels (via correspondence) -- same selection the probe used. Isolates the
    bounding-box penalty from the alignment quality."""
    px, py, pidx = corr[:, 0], corr[:, 1], corr[:, -1].astype(np.int64)
    valid = (pidx >= 0) & (pidx < N)
    px, py, pidx = px[valid], py[valid], pidx[valid]
    sel = instance[pidx] == inst_id
    if sel.sum() == 0:
        return np.array([], dtype=np.int64)
    return np.unique(P.px_to_patch(px[sel], py[sel], sx, sy))


def foreground_filter(F2D, kp, box, W, H, device):
    """Within a (loose) box, keep the patch cluster nearest the box center via a
    tiny 2-means on DINO features -- a training-free, SAM-free foreground proxy."""
    if len(kp) < 6:
        return kp
    feats = F2D[torch.as_tensor(kp, device=device)]            # [n,C] normalized
    # patch grid centers in image px
    sx, sy = P.CROP / W, P.CROP / H
    pu = (kp % P.PATCH_HW); pv = (kp // P.PATCH_HW)
    cx = (pu + 0.5) * P.PATCH_SIZE / sx; cy = (pv + 0.5) * P.PATCH_SIZE / sy
    bx, by = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    # 2-means (5 iters) seeded by the two most dissimilar patches
    g = (feats @ feats.T)
    i = int(g.sum(1).argmin()); j = int(g[i].argmin())
    c = torch.stack([feats[i], feats[j]])
    for _ in range(5):
        a = (feats @ c.T).argmax(1)
        for t in (0, 1):
            if (a == t).any():
                c[t] = F.normalize(feats[a == t].mean(0), dim=0)
    a = a.cpu().numpy()
    # pick the cluster whose patches are closer to the box center
    d = (cx - bx) ** 2 + (cy - by) ** 2
    keep = 0 if d[a == 0].mean() <= d[a == 1].mean() else 1
    return kp[a == keep]


def instance_box_from_corr(corr, instance, inst_id, N, pad=8):
    """Bounding box (x0,y0,x1,y1) of inst_id's projected pixels -- detector stand-in."""
    px, py, pidx = corr[:, 0], corr[:, 1], corr[:, -1].astype(np.int64)
    valid = (pidx >= 0) & (pidx < N)
    px, py, pidx = px[valid], py[valid], pidx[valid]
    sel = instance[pidx] == inst_id
    if sel.sum() == 0:
        return None
    return (float(px[sel].min() - pad), float(py[sel].min() - pad),
            float(px[sel].max() + pad), float(py[sel].max() + pad))


def visible_instances(corr, instance, N, top=12):
    pidx = corr[:, -1].astype(np.int64)
    pidx = pidx[(pidx >= 0) & (pidx < N)]
    insts = instance[pidx]
    insts = insts[insts >= 0]
    if len(insts) == 0:
        return []
    vals, counts = np.unique(insts, return_counts=True)
    order = np.argsort(-counts)
    return [(int(vals[i]), int(counts[i])) for i in order[:top]]


def best_frame_for_instance(image_dir, instance, inst_id, N):
    """Scan correspondence/*.npy; return (frame_id, pixel_count) where inst_id shows most."""
    best_f, best_c = None, 0
    for f in P.list_frames(image_dir):
        try:
            corr = np.load(os.path.join(image_dir, "correspondence", f"{f}.npy"))
        except (FileNotFoundError, OSError):
            continue
        if corr.ndim != 2 or corr.shape[1] < 3:
            continue
        pidx = corr[:, -1].astype(np.int64)
        pidx = pidx[(pidx >= 0) & (pidx < N)]
        c = int((instance[pidx] == inst_id).sum())
        if c > best_c:
            best_f, best_c = f, c
    return best_f, best_c


def dominant_instance_under_patches(corr, instance, kp, sx, sy, N):
    """What GT instances actually sit under the Stage-1-selected patches?
    Lets you see if a click/box hit the object you meant. Returns [(inst, count), ...]."""
    px, py, pidx = corr[:, 0], corr[:, 1], corr[:, -1].astype(np.int64)
    valid = (pidx >= 0) & (pidx < N)
    px, py, pidx = px[valid], py[valid], pidx[valid]
    patch = P.px_to_patch(px, py, sx, sy)
    under = np.isin(patch, kp)
    insts = instance[pidx[under]]
    insts = insts[insts >= 0]
    if len(insts) == 0:
        return None
    vals, counts = np.unique(insts, return_counts=True)
    order = np.argsort(-counts)
    return [(int(vals[i]), int(counts[i])) for i in order[:5]]


def _subsample(n, max_points, seed=0):
    if n <= max_points:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=max_points, replace=False)


def write_scatter_html(path, coord, value=None, point_rgb=None, colorscale="Spectral_r",
                       title="", overlay=None, image=None, max_points=120000):
    """Self-contained plotly HTML: the input 2D image (left) next to the 3D point-cloud
    heatmap (right), so you can eyeball that the highlighted region is the right place.
    value: per-point scalar -> colorscale. point_rgb: per-point [0,1]^3 colors.
    overlay: optional (mask, color, name) extra 3D trace toggled via the legend.
    image: HxWx3 uint8 input image to show beside the scene."""
    import plotly.graph_objects as go
    idx = _subsample(len(coord), max_points)
    c = coord[idx]
    if value is not None:
        marker = dict(size=1.5, color=np.asarray(value)[idx], colorscale=colorscale,
                      colorbar=dict(title="cosine", x=1.0), opacity=0.8)
    elif point_rgb is not None:
        cols = (np.clip(np.asarray(point_rgb)[idx], 0, 1) * 255).astype(int)
        marker = dict(size=1.5, color=[f"rgb({r},{g},{b})" for r, g, b in cols], opacity=0.8)
    else:
        marker = dict(size=1.5, opacity=0.8)
    cloud = go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers",
                         marker=marker, name="points")
    over = None
    if overlay is not None:
        mask, ocolor, oname = overlay
        oc = c[np.asarray(mask)[idx]]
        over = go.Scatter3d(x=oc[:, 0], y=oc[:, 1], z=oc[:, 2], mode="markers",
                            marker=dict(size=2.0, color=ocolor), name=oname,
                            visible="legendonly")

    if image is not None:
        from plotly.subplots import make_subplots
        img = np.asarray(image)
        step = max(1, int(np.ceil(img.shape[1] / 640)))   # downscale wide images
        fig = make_subplots(rows=1, cols=2, column_widths=[0.38, 0.62],
                            specs=[[{"type": "image"}, {"type": "scene"}]],
                            subplot_titles=("input image", "3D scene (bright = match)"))
        fig.add_trace(go.Image(z=img[::step, ::step]), row=1, col=1)
        fig.add_trace(cloud, row=1, col=2)
        if over is not None:
            fig.add_trace(over, row=1, col=2)
    else:
        fig = go.Figure([cloud] + ([over] if over is not None else []))
    fig.update_layout(title=title, scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(path, include_plotlyjs=True, full_html=True)


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
    ap.add_argument("--mode",
                    choices=["box", "json", "point", "text", "auto", "auto_mask",
                             "image"],
                    default="point",
                    help="auto: box from --eval_instance's correspondence (loose "
                         "detector stand-in). auto_mask: pixel-accurate patches "
                         "(== probe selection; isolates the bounding-box penalty).")
    ap.add_argument("--agg", choices=["mean", "max"], default="mean",
                    help="Stage-2 aggregation over selected patches (max is more "
                         "robust to background patches in a loose box).")
    ap.add_argument("--fg", action="store_true",
                    help="foreground-filter a (loose) box via 2-means before matching")
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
    ap.add_argument("--max_points", type=int, default=120000,
                    help="subsample cap for the HTML scatter (keeps it responsive)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = P.load_backbone(args.backbone_ckpt, device)
    patch_proj = P.load_patch_proj(args.patch_proj_ckpt or args.backbone_ckpt, device)
    dino = P.load_dino(device)
    F3D, scene = P.compute_scene_F3D(args.scene_dir, model, patch_proj, device, args.scale)
    coord = scene["coord"]
    print(f"[F3D] {tuple(F3D.shape)}")

    import imageio.v2 as imageio
    N = len(coord)
    image_dir = args.image_dir or P.default_image_dir(args.scene_dir)
    frame = args.frame

    # auto/auto_mask: if the requested object isn't visible in --frame, jump to the
    # frame that shows it best (so the eval prompt provably targets the object).
    if args.mode in ("auto", "auto_mask") and not args.image:
        assert args.eval_instance is not None, "--mode auto* needs --eval_instance"
        _, corr0 = P.load_frame_corr(image_dir, frame)
        if instance_box_from_corr(corr0, scene["instance"], args.eval_instance, N) is None:
            bf, bc = best_frame_for_instance(image_dir, scene["instance"],
                                             args.eval_instance, N)
            if bf is None:
                vis = visible_instances(corr0, scene["instance"], N)
                sys.exit(f"instance {args.eval_instance} not visible in ANY frame.\n"
                         f"instances visible in frame {frame} (inst,count): {vis}")
            print(f"[auto] inst {args.eval_instance} not in frame {frame}; "
                  f"switching to frame {bf} ({bc} px).")
            frame = bf

    # load the query image (+ correspondence for auto box / diagnostics)
    corr = None
    if args.image:
        rgb = imageio.imread(args.image)
    else:
        rgb = imageio.imread(os.path.join(image_dir, "color", f"{frame}.png"))
        try:
            _, corr = P.load_frame_corr(image_dir, frame)
        except FileNotFoundError:
            pass
    H, W = rgb.shape[:2]
    img_t, sx, sy = P.dino_preprocess(rgb)
    F2D = F.normalize(P.extract_dino_patches(dino, img_t, device), dim=-1)
    if corr is not None:
        print(f"[frame {frame}] visible instances (inst,count): "
              f"{visible_instances(corr, scene['instance'], N)}")

    # Stage-1: object -> DINO patches.  box=loose detector box (mixes background),
    # auto_mask=pixel-accurate patches (== probe selection; isolates the box penalty),
    # image=ALL patches (no text/box: "where in the scene is this whole image looking?").
    box = None
    if args.mode == "image":
        kp = np.arange(P.PATCH_HW * P.PATCH_HW, dtype=np.int64)
    elif args.mode == "auto":
        box = instance_box_from_corr(corr, scene["instance"], args.eval_instance, N)
        assert box, f"instance {args.eval_instance} not visible in frame {frame}"
        print(f"[stage-1] auto box from inst {args.eval_instance}: "
              f"({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})")
        kp = select_patches_box(box, W, H)
    elif args.mode == "auto_mask":
        kp = patches_with_instance_pixels(corr, scene["instance"],
                                          args.eval_instance, sx, sy, N)
        assert len(kp), f"instance {args.eval_instance} not visible in frame {frame}"
    elif args.mode == "box":
        assert args.box, "--box required"
        box = tuple(args.box)
        kp = select_patches_box(box, W, H)
    elif args.mode == "json":
        assert args.boxes_json and args.text, "--boxes_json and --text required"
        dets = json.load(open(args.boxes_json)).get(str(frame), [])
        cand = [d for d in dets if args.text.lower() in str(d.get("label", "")).lower()]
        assert cand, f"no '{args.text}' box for frame {frame} in {args.boxes_json}"
        box = tuple(cand[0]["box"])
        kp = select_patches_box(box, W, H)
    elif args.mode == "point":
        assert args.point, "--point required"
        kp = select_patches_point(F2D, args.point, W, H, args.tau)
    else:  # text
        kp = select_patches_text(img_t, args.text, rgb, device, args.tau)
    assert len(kp) > 0, "Stage-1 selected no patches"
    if args.fg and box is not None:                       # box foreground filter
        kp2 = foreground_filter(F2D, kp, box, W, H, device)
        print(f"[stage-1] fg filter: {len(kp)} -> {len(kp2)} patches")
        kp = kp2 if len(kp2) else kp
    print(f"[stage-1] mode={args.mode} agg={args.agg} -> {len(kp)} DINO patches")

    # diagnostic: which GT instances actually sit under the selected patches?
    if corr is not None:
        dom = dominant_instance_under_patches(corr, scene["instance"], kp, sx, sy, N)
        print(f"[stage-1 diag] GT instances under selected patches (inst,count): {dom}"
              + ("" if args.eval_instance is None else
                 f"  <- you are evaluating inst {args.eval_instance}"))

    # Stage-2: cosine match to the DINO-aligned 3D map.
    #   mean = single averaged query (sensitive to off-object patches in the set)
    #   max  = per-3D-point max cosine over the selected patches (chunked, memory-safe)
    # For --mode image, mean over the whole image = generic scene appearance (useless),
    # so we force max: each 3D point lights up by its single best-matching image patch.
    agg = args.agg
    if args.mode == "image" and agg == "mean":
        print("[stage-2] --mode image: forcing --agg max (mean over all patches is "
              "uninformative)")
        agg = "max"
    Fk = F2D[torch.as_tensor(kp, device=device)]
    if agg == "max":
        scores = P._max_sim_scores(F3D, Fk)
    else:
        q = F.normalize(Fk.mean(0), dim=0)
        scores = (F3D @ q).detach().cpu().numpy()
    topk = np.argsort(-scores)[: args.topk]
    np.save(args.out + "_topk.npy", topk)

    # --mode image: GT "footprint" = points actually visible in the frame.
    overlay = None
    if args.mode == "image" and corr is not None:
        pidx = corr[:, -1].astype(np.int64)
        pidx = pidx[(pidx >= 0) & (pidx < N)]
        gt_vis = np.zeros(N, dtype=bool)
        gt_vis[np.unique(pidx)] = True
        overlay = (gt_vis, "red", "GT visible (toggle)")
        m = P.retrieval_metrics(scores, gt_vis)
        print(f"[eval image footprint] visible={int(gt_vis.sum())} | "
              f"AP={m['AP']:.3f} IoU={m['best_IoU']:.3f} prec@500={m['prec@500']:.3f}")

    write_scatter_html(args.out + ".html", coord, value=scores, image=rgb,
                       title=f"{args.mode} | bright=match | red trace=GT (toggle)",
                       overlay=overlay, max_points=args.max_points)
    print(f"[out] {args.out}.html  {args.out}_topk.npy  (open the .html in a browser)")

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
