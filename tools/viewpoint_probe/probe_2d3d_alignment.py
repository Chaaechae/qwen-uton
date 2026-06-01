"""
Feasibility probe: 2D -> 3D localization via Utonia's learned 2D-3D alignment.

Question this script answers (quantitatively, BEFORE building a full pipeline):
    Given Utonia's frozen 3D features (projected into the DINO space by the trained
    `patch_proj`), can we localize an object in the 3D map by matching it to the
    DINO feature(s) of that object in a single 2D image -- WITHOUT using camera
    pose/depth for the lift?  And how does it compare to what the frame geometrically
    covers (the raycast correspondence already stored by the preprocessor)?

It reports, per (frame, instance):
    - retrieval AP / best-IoU / prec@K of the learned-alignment matching (the test)
    - single-frame geometric coverage (raycast-visible part of the instance)
    - instance-ambiguity diagnostic (same-class duplicates polluting the top-K)
    - chance floor (random query vector)

Data layout (Pointcept concerto ScanNet preprocessing):
    scene_dir = <root>/scannet/val/scene0011_00/
        coord.npy color.npy normal.npy segment20.npy instance.npy   (point order = "gt")
    image_dir = <root>/scannet/images/val/scene0011_00/
        color/<f>.png                         raycast RGB
        correspondence/<f>.npy   shape (M,4)  [pixel_x, pixel_y, 1, gt_point_index]
        (depth/ pose/ intrinsic/ also present but NOT needed -- correspondence is
         already occlusion-aware via mesh raycasting.)

Prerequisite checkpoint:
    A FULL Utonia-v1m1 *pretrain* checkpoint containing BOTH
    `module.student.backbone.*` AND `module.patch_proj.*`.  The released standalone
    weight is backbone-only and is NOT sufficient: the DINO-aligned space only exists
    after `patch_proj`.

Nothing is trained here; matching is pure cosine.
"""

import argparse
import os
import sys
import glob
import numpy as np

# --- make the standalone `utonia` package importable without `pip install -e .` ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F
import utonia


# Backbone config (PT-v3m3) copied verbatim from
# configs/utonia/pretrain-utonia-v1m1-0-base_stagev1.py so the standalone
# PointTransformerV3 matches pretraining exactly.
BACKBONE_CONFIG = dict(
    in_channels=9,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(3, 3, 3, 12, 3),
    enc_channels=(54, 108, 216, 432, 576),
    enc_num_head=(3, 6, 12, 24, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    mlp_ratio=4,
    qkv_bias=True,
    drop_path=0.0,           # eval: no drop path
    shuffle_orders=False,    # eval: deterministic
    pre_norm=True,
    enable_rpe=False,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
    enc_mode=True,
    traceable=True,          # REQUIRED so pooling_parent chain exists for up_cast
    mask_token=True,
    rope_base=10,
)

ENC2D_UPCAST_LEVEL = 3            # enc2d_upcast_level=3 -> 576+432+216+108 = 1332
BACKBONE_OUT_CHANNELS = 1332     # backbone_out_channels (patch_proj input dim)
DINO_DIM = 1536                  # DINOv2-giant feature dim (enc2d_head_in_channels)
DINO_MODEL_ID = "facebook/dinov2-with-registers-giant"
PATCH_SIZE = 14
CROP = 518
PATCH_HW = CROP // PATCH_SIZE     # 37 -> 37*37 = 1369 tokens
IGNORE = -100                     # preprocessor IGNORE_INDEX for instance/segment


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _strip_prefixes(sd):
    backbone_sd, patch_proj_sd = {}, {}
    for k, v in sd.items():
        kk = k[len("module."):] if k.startswith("module.") else k
        if kk.startswith("student.backbone."):
            backbone_sd[kk[len("student.backbone."):]] = v
        elif kk.startswith("backbone."):
            backbone_sd[kk[len("backbone."):]] = v
        elif kk.startswith("patch_proj."):
            patch_proj_sd[kk[len("patch_proj."):]] = v
    return backbone_sd, patch_proj_sd


def load_backbone_and_proj(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    backbone_sd, patch_proj_sd = _strip_prefixes(sd)
    if not backbone_sd:
        raise RuntimeError(
            f"No backbone weights in {ckpt_path} (expected 'module.student.backbone.*'). "
            f"Available top-level prefixes: "
            f"{sorted({k.split('.')[0] for k in sd})[:10]}"
        )
    model = utonia.model.PointTransformerV3(**BACKBONE_CONFIG)
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    print(f"[backbone] loaded; missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()

    if not patch_proj_sd:
        print("ERROR: checkpoint has no `patch_proj` -> the DINO-aligned space does "
              "not exist. Supply a FULL Utonia-v1m1 pretrain checkpoint.")
        sys.exit(2)
    patch_proj = torch.nn.Linear(BACKBONE_OUT_CHANNELS, DINO_DIM)
    patch_proj.load_state_dict(patch_proj_sd)
    patch_proj = patch_proj.to(device).eval()
    print("[patch_proj] loaded (faithful aligned space).")
    return model, patch_proj


def load_dino(device):
    from transformers import AutoModel
    dino = AutoModel.from_pretrained(DINO_MODEL_ID, trust_remote_code=True)
    return dino.to(device).eval()


# --------------------------------------------------------------------------- #
# 3D feature extraction (reproduces the enc2d alignment space per input point)
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def extract_3d_features(model, patch_proj, data_dict, inverse, device, cos_shift=True):
    """Per-ORIGINAL-point features in the DINO-aligned space, [N_orig, 1536].
    Mirrors Utonia.up_cast (3 concat levels -> 1332) + patch_proj + cos_shift."""
    for k in data_dict:
        if torch.is_tensor(data_dict[k]):
            data_dict[k] = data_dict[k].to(device)
    point = model(data_dict)

    for _ in range(ENC2D_UPCAST_LEVEL):              # concat up-cast -> 1332 ch
        assert "pooling_parent" in point.keys(), "need traceable=True backbone"
        parent = point.pop("pooling_parent")
        inv = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inv]], dim=-1)
        point = parent
    assert point.feat.shape[-1] == BACKBONE_OUT_CHANNELS, (
        f"upcast dim {point.feat.shape[-1]} != {BACKBONE_OUT_CHANNELS}")

    feat = patch_proj(point.feat)                    # -> DINO space
    if cos_shift:
        feat = feat - feat.mean(dim=-1, keepdim=True)

    while "pooling_parent" in point.keys():          # broadcast to grid resolution
        parent = point.pop("pooling_parent")
        inv = point.pop("pooling_inverse")
        parent.feat = feat[inv]
        point = parent
        feat = point.feat

    return feat[inverse]                             # grid -> original points


@torch.inference_mode()
def extract_dino_patches(dino, image_chw, device, cos_shift=True):
    out = dino(image_chw.unsqueeze(0).to(device))
    feats = out.last_hidden_state[:, -PATCH_HW * PATCH_HW:, :]  # drop cls/register
    feats = feats.reshape(-1, feats.shape[-1])                 # [1369, 1536]
    if cos_shift:
        feats = feats - feats.mean(dim=-1, keepdim=True)
    return feats


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_scene(scene_dir):
    def _load(*names):
        for n in names:
            p = os.path.join(scene_dir, n + ".npy")
            if os.path.isfile(p):
                return np.load(p)
        raise FileNotFoundError(f"{names} not in {scene_dir}")
    return dict(
        coord=_load("coord").astype(np.float64),
        color=_load("color").astype(np.float32),
        normal=_load("normal").astype(np.float32),
        instance=_load("instance").astype(np.int64).reshape(-1),
        semantic=_load("segment20", "segment", "semantic").astype(np.int64).reshape(-1),
    )


def default_image_dir(scene_dir):
    """.../scannet/val/scene0011_00 -> .../scannet/images/val/scene0011_00"""
    scene = os.path.normpath(scene_dir)
    split = os.path.basename(os.path.dirname(scene))   # val/train/test
    root = os.path.dirname(os.path.dirname(scene))     # .../scannet
    return os.path.join(root, "images", split, os.path.basename(scene))


def list_frames(image_dir):
    fs = glob.glob(os.path.join(image_dir, "correspondence", "*.npy"))
    return sorted([os.path.splitext(os.path.basename(f))[0] for f in fs],
                  key=lambda x: int(x) if x.isdigit() else x)


def load_frame_corr(image_dir, frame):
    """Returns (rgb[H,W,3] uint8, corr[M,4] = [px, py, 1, point_idx])."""
    import imageio.v2 as imageio
    png = os.path.join(image_dir, "color", f"{frame}.png")
    rgb = imageio.imread(png)
    corr = np.load(os.path.join(image_dir, "correspondence", f"{frame}.npy"))
    return rgb, corr


def dino_preprocess(rgb):
    """rgb[H,W,3] uint8 -> (tensor[3,518,518], sx, sy) where s maps pixels to 518 grid."""
    import torchvision.transforms.functional as TF
    H, W = rgb.shape[:2]
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    t = TF.resize(t, [CROP, CROP], antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std, CROP / W, CROP / H


def px_to_patch(px, py, sx, sy):
    pu = np.clip((px * sx / PATCH_SIZE).astype(int), 0, PATCH_HW - 1)
    pv = np.clip((py * sy / PATCH_SIZE).astype(int), 0, PATCH_HW - 1)
    return pv * PATCH_HW + pu


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def retrieval_metrics(scores, gt_mask, ks=(50, 100, 200, 500)):
    order = np.argsort(-scores)
    gt = gt_mask[order]
    n_gt = int(gt_mask.sum())
    out = {}
    for k in ks:
        out[f"prec@{k}"] = float(gt[:k].mean()) if k <= len(gt) else float("nan")
        out[f"rec@{k}"] = float(gt[:k].sum() / max(n_gt, 1))
    tp = np.cumsum(gt)
    prec = tp / (np.arange(len(gt)) + 1)
    out["AP"] = float((prec * gt).sum() / max(n_gt, 1))
    best_iou = 0.0
    for k in np.unique(np.linspace(1, len(gt), 50).astype(int)):
        pred = np.zeros(len(gt), dtype=bool)
        pred[order[:k]] = True
        inter = (pred & gt_mask).sum()
        union = (pred | gt_mask).sum()
        best_iou = max(best_iou, inter / max(union, 1))
    out["best_IoU"] = float(best_iou)
    return out


def _seed_score(seed_mask, coord, radius):
    from scipy.spatial import cKDTree
    if seed_mask.sum() == 0:
        return np.zeros(len(coord))
    d, _ = cKDTree(coord[seed_mask]).query(coord, k=1)
    return np.exp(-d / max(radius, 1e-6))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_probe(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_dir = args.image_dir or default_image_dir(args.scene_dir)
    print(f"[paths] scene_dir={args.scene_dir}\n        image_dir={image_dir}")

    model, patch_proj = load_backbone_and_proj(args.pretrain_ckpt, device)
    dino = load_dino(device)

    scene = load_scene(args.scene_dir)
    coord_w, instance, semantic = scene["coord"], scene["instance"], scene["semantic"]
    N = len(coord_w)
    print(f"[scene] {N} points, {len(np.unique(instance[instance >= 0]))} instances")

    # 3D features in DINO-aligned space (original point order == correspondence idx)
    transform = utonia.transform.default(args.scale)
    data = transform(dict(coord=coord_w.astype(np.float32).copy(),
                          color=scene["color"].copy(), normal=scene["normal"].copy()))
    inverse = data.pop("inverse")
    if torch.is_tensor(inverse):
        inverse = inverse.to(device)
    F3D = F.normalize(extract_3d_features(model, patch_proj, data, inverse, device), dim=-1)
    print(f"[F3D] {tuple(F3D.shape)}")

    frames = args.frames.split(",") if args.frames else list_frames(image_dir)
    print(f"[frames] {len(frames)} -> {frames[:8]}{'...' if len(frames) > 8 else ''}")

    rows = []
    for fid in frames:
        try:
            rgb, corr = load_frame_corr(image_dir, fid)
        except FileNotFoundError as e:
            print(f"[frame {fid}] skip ({e})")
            continue
        if corr.ndim != 2 or corr.shape[0] < 50 or (corr < 0).all():
            continue
        px, py, pidx = corr[:, 0], corr[:, 1], corr[:, 3].astype(np.int64)
        valid = (pidx >= 0) & (pidx < N)
        px, py, pidx = px[valid], py[valid], pidx[valid]
        if len(pidx) < 50:
            continue

        img_t, sx, sy = dino_preprocess(rgb)
        F2D = F.normalize(extract_dino_patches(dino, img_t, device), dim=-1)
        patch = px_to_patch(px, py, sx, sy)
        inst_of_corr = instance[pidx]

        vis_inst, counts = np.unique(inst_of_corr[inst_of_corr >= 0], return_counts=True)
        targets = [int(i) for i, c in zip(vis_inst, counts) if c >= args.min_inst_pts]
        targets = targets[: args.max_targets]

        for k in targets:
            gt_mask = (instance == k)
            sem_k = int(np.bincount(semantic[gt_mask & (semantic >= 0)]).argmax())

            sel_rows = inst_of_corr == k                       # corr rows seeing inst k
            kp = np.unique(patch[sel_rows])
            if len(kp) < args.min_patches:
                continue
            q = F.normalize(F2D[torch.as_tensor(kp, device=device)].mean(0), dim=0)

            s_align = (F3D @ q).detach().cpu().numpy()          # (A) learned alignment
            m_align = retrieval_metrics(s_align, gt_mask)

            vis_k = np.zeros(N, dtype=bool)                     # (B) frame coverage
            vis_k[np.unique(pidx[sel_rows])] = True
            m_geo = retrieval_metrics(_seed_score(vis_k, coord_w, args.bp_radius), gt_mask)

            s_rand = (F3D @ F.normalize(torch.randn(DINO_DIM, device=device), dim=0)
                      ).detach().cpu().numpy()
            m_rand = retrieval_metrics(s_rand, gt_mask)

            topk = np.argsort(-s_align)[: args.amb_k]
            amb = ((semantic[topk] == sem_k) & (instance[topk] != k)).mean()
            ndist = len(np.unique(instance[topk][instance[topk] >= 0]))

            rows.append(dict(
                frame=fid, inst=k, sem=sem_k, n_pts=int(gt_mask.sum()),
                vis_pts=int(vis_k.sum()),
                align_AP=m_align["AP"], align_IoU=m_align["best_IoU"],
                align_rec200=m_align["rec@200"],
                geo_AP=m_geo["AP"], geo_rec200=m_geo["rec@200"],
                rand_AP=m_rand["AP"],
                amb_sameclass_otherinst=float(amb), amb_distinct_inst=int(ndist)))
            print(f"[{fid} inst{k} sem{sem_k} n={rows[-1]['n_pts']} vis={rows[-1]['vis_pts']}] "
                  f"align AP={m_align['AP']:.3f} IoU={m_align['best_IoU']:.3f} "
                  f"rec200={m_align['rec@200']:.3f} | geo AP={m_geo['AP']:.3f} "
                  f"rec200={m_geo['rec@200']:.3f} | rand={m_rand['AP']:.3f} | "
                  f"amb={amb:.2f} ndist={ndist}")

    _summarize(rows, args.out_csv)


def _summarize(rows, out_csv):
    if not rows:
        print("\nNo (frame,instance) pairs evaluated. Check paths/thresholds.")
        return
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    arr = lambda k: np.array([r[k] for r in rows], float)
    print(f"\n============= SUMMARY ({len(rows)} pairs) =============")
    for k in ("align_AP", "align_IoU", "align_rec200", "geo_AP", "geo_rec200",
              "rand_AP", "amb_sameclass_otherinst", "amb_distinct_inst"):
        v = arr(k)
        print(f"  {k:26s} mean={np.nanmean(v):.3f} median={np.nanmedian(v):.3f}")
    a, g, r = arr("align_AP"), arr("geo_AP"), arr("rand_AP")
    ar, gr = arr("align_rec200"), arr("geo_rec200")
    amb = arr("amb_sameclass_otherinst")
    print("\n  Verdict:")
    print(f"    alignment vs chance:   {np.nanmean(a):.3f} vs {np.nanmean(r):.3f}  "
          f"-> {'PASS' if np.nanmean(a) > 3 * np.nanmean(r) else 'WEAK'}")
    print(f"    alignment vs frame-coverage (recall@200): {np.nanmean(ar):.3f} vs "
          f"{np.nanmean(gr):.3f}  "
          f"-> {'lights up unseen parts' if np.nanmean(ar) > np.nanmean(gr) else 'within frame only'}")
    print(f"    instance ambiguity: {np.nanmean(amb):.2f}  "
          f"-> {'LOW (good)' if np.nanmean(amb) < 0.2 else 'HIGH (duplicates pollute)'}")
    print(f"\n  CSV -> {out_csv}")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrain_ckpt", default="/group-volume/Utonia/utonia.pth")
    p.add_argument("--scene_dir",
                   default="/group-volume/3Ddataset/data/scannet/val/scene0011_00")
    p.add_argument("--image_dir", default=None,
                   help="defaults to <root>/images/<split>/<scene> derived from scene_dir")
    p.add_argument("--frames", default="",
                   help="comma-separated frame ids; empty = all frames in the scene")
    p.add_argument("--scale", type=float, default=1.0,
                   help="utonia.transform.default coord scale; MATCH your extraction")
    p.add_argument("--min_inst_pts", type=int, default=200)
    p.add_argument("--min_patches", type=int, default=3)
    p.add_argument("--max_targets", type=int, default=20)
    p.add_argument("--bp_radius", type=float, default=0.10)
    p.add_argument("--amb_k", type=int, default=200)
    p.add_argument("--out_csv", default="viewpoint_probe_results.csv")
    return p


if __name__ == "__main__":
    run_probe(build_argparser().parse_args())
