"""
Feasibility probe: text/2D -> 3D localization via Utonia's learned 2D-3D alignment.

Question this script answers (quantitatively, BEFORE building a full pipeline):
    Given Utonia's frozen 3D features (projected into the DINO space by the trained
    `patch_proj`), can we localize an object in the 3D map by matching it to the
    DINO feature(s) of that object in a single 2D image -- WITHOUT using camera
    pose / depth for the lift?  And how does this learned-alignment matching compare
    to the geometric upper bound (pose+depth back-projection)?

It reports, per (frame, instance):
    - retrieval@K / AP / best-IoU of the learned-alignment matching (the thing we test)
    - the same metrics for the back-projection upper bound (geometry ceiling)
    - an instance-ambiguity diagnostic (how much same-class duplicates pollute the top-K)
    - chance floor (random query vector)

Decision rule (see README.md):
    learned-alignment >> chance and reasonably close to the back-projection ceiling
        -> training-free 2-stage (2D detect -> 3D match) is realistic.
    learned-alignment ~ chance, or instance-ambiguity dominates
        -> alignment is too weak / not instance-discriminative; you need geometry
           (back-projection) or a trained head.

IMPORTANT prerequisites:
    * A FULL Utonia *pretrain* checkpoint (Utonia-v1m1), which contains BOTH the
      student backbone (`module.student.backbone.*`) AND `module.patch_proj.*`.
      The released standalone weight is backbone-only and is NOT sufficient, because
      the DINO-aligned space only exists after `patch_proj`.
    * DINOv2-giant-with-registers (the enc2d teacher) -- `facebook/dinov2-with-registers-giant`.
    * One scene with: point cloud (coord/color/normal) + GT instance/semantic labels
      + at least one posed RGB-D frame (rgb, depth[m], depth-intrinsic, cam2world).
      ScanNet / ScanNet++ / Structured3D all qualify; a ScanNet reference loader is
      provided below -- adapt `load_scene` / `load_frame` to your export layout.

This script is intentionally read-only w.r.t. the model: nothing is trained, the
matching is pure cosine.  If your checkpoint lacks `patch_proj` the aligned space
does not exist and the script aborts -- supply a full pretrain checkpoint.
"""

import argparse
import os
import sys
import numpy as np

import torch
import torch.nn.functional as F

# Utonia standalone package (backbone + transforms).  Must be importable
# (`pip install -e .` in the repo root, or run from the repo root).
import utonia


# Config of the pretrain backbone (PT-v3m3) -- copied from
# configs/utonia/pretrain-utonia-v1m1-0-base_stagev1.py so the standalone
# PointTransformerV3 is built with the exact same architecture as pretraining.
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
    shuffle_orders=False,    # eval: deterministic ordering
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

ENC2D_UPCAST_LEVEL = 3           # config: enc2d_upcast_level=3 -> 576+432+216+108=1332
BACKBONE_OUT_CHANNELS = 1332     # config: backbone_out_channels
DINO_DIM = 1536                  # DINOv2-giant feature dim (enc2d_head_in_channels)
DINO_MODEL_ID = "facebook/dinov2-with-registers-giant"
PATCH_SIZE = 14
CROP = 518
PATCH_HW = CROP // PATCH_SIZE     # 37 -> 37*37 = 1369 patch tokens


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _strip_prefixes(sd):
    """Pretrain checkpoints store weights under module.student.backbone.* etc.
    Return (backbone_sd, patch_proj_sd) with clean keys for the standalone model."""
    backbone_sd, patch_proj_sd = {}, {}
    for k, v in sd.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]
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
            f"No backbone weights found in {ckpt_path}. Expected keys like "
            f"'module.student.backbone.*'. Is this a full Utonia-v1m1 pretrain ckpt?"
        )

    model = utonia.model.PointTransformerV3(**BACKBONE_CONFIG)
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    print(f"[backbone] missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()

    patch_proj = None
    if patch_proj_sd:
        patch_proj = torch.nn.Linear(BACKBONE_OUT_CHANNELS, DINO_DIM)
        patch_proj.load_state_dict(patch_proj_sd)
        patch_proj = patch_proj.to(device).eval()
        print("[patch_proj] loaded from checkpoint (faithful aligned space).")
    else:
        print(
            "[patch_proj] NOT FOUND in checkpoint. The DINO-aligned space is "
            "unavailable; supply a full Utonia-v1m1 pretrain checkpoint."
        )
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
    """Return per-ORIGINAL-point features in the DINO-aligned space, [N_orig, 1536].

    Steps mirror Utonia.up_cast + patch_proj + enc2d_cos_shift:
      1. backbone forward (enc_mode, traceable) -> deepest point (576-ch) w/ pooling chain
      2. up_cast ENC2D_UPCAST_LEVEL levels by CONCAT -> 1332-ch at that resolution
      3. broadcast (no concat) up the remaining pooling levels to grid resolution
      4. patch_proj: 1332 -> 1536  (the trained 3D->DINO map)
      5. cos_shift: subtract per-vector channel mean (matches enc2d_cos_shift)
      6. scatter to original points via GridSample `inverse`
    """
    for k in data_dict:
        if torch.is_tensor(data_dict[k]):
            data_dict[k] = data_dict[k].to(device)

    point = model(data_dict)

    # (2) concat up-cast to reach the 1332-ch enc2d feature
    for _ in range(ENC2D_UPCAST_LEVEL):
        assert "pooling_parent" in point.keys(), "need traceable=True backbone"
        parent = point.pop("pooling_parent")
        inv = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inv]], dim=-1)
        point = parent
    assert point.feat.shape[-1] == BACKBONE_OUT_CHANNELS, (
        f"upcast dim {point.feat.shape[-1]} != {BACKBONE_OUT_CHANNELS}; "
        f"check ENC2D_UPCAST_LEVEL / enc_channels"
    )

    feat = point.feat
    if patch_proj is not None:
        feat = patch_proj(feat)            # (4) -> DINO space
    # (5) cos_shift mean-subtract (per-point over channels)
    if cos_shift and patch_proj is not None:
        feat = feat - feat.mean(dim=-1, keepdim=True)

    # (3) broadcast remaining pooling levels to grid resolution (no concat)
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inv = point.pop("pooling_inverse")
        parent.feat = feat[inv]
        point = parent
        feat = point.feat

    # (6) grid-resolution feat -> original points
    feat_orig = feat[inverse]              # inverse: [N_orig] -> grid index
    return feat_orig


@torch.inference_mode()
def extract_dino_patches(dino, image_chw, device, cos_shift=True):
    """image_chw: float tensor [3, 518, 518], DINO-normalized. Returns [1369, 1536]."""
    x = image_chw.unsqueeze(0).to(device)
    out = dino(x)
    feats = out.last_hidden_state[:, -PATCH_HW * PATCH_HW:, :]  # drop cls/register tokens
    feats = feats.reshape(-1, feats.shape[-1])                  # [1369, 1536]
    if cos_shift:
        feats = feats - feats.mean(dim=-1, keepdim=True)
    return feats


# --------------------------------------------------------------------------- #
# Geometry: project map points into a posed frame, build patch<->3D correspondence
# --------------------------------------------------------------------------- #
def project_points(coord_world, cam2world, K, H, W):
    """Pinhole-project world points into the frame.
    Returns u, v (float pixel coords), z (camera-space depth), and in_front mask."""
    world2cam = np.linalg.inv(cam2world)
    Xc = (world2cam[:3, :3] @ coord_world.T + world2cam[:3, 3:4]).T  # [N,3]
    z = Xc[:, 2]
    in_front = z > 1e-6
    u = np.full(len(coord_world), -1.0)
    v = np.full(len(coord_world), -1.0)
    zz = np.where(in_front, z, 1.0)
    u_ = (K[0, 0] * Xc[:, 0] / zz) + K[0, 2]
    v_ = (K[1, 1] * Xc[:, 1] / zz) + K[1, 2]
    u[in_front] = u_[in_front]
    v[in_front] = v_[in_front]
    in_img = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, in_img


def visible_mask(coord_world, cam2world, K, depth, H, W, depth_tol=0.05):
    """z-buffer visibility against the frame's measured depth map.
    A point is 'visible' if it projects in-image and its camera z matches depth(u,v)."""
    u, v, z, in_img = project_points(coord_world, cam2world, K, H, W)
    vis = np.zeros(len(coord_world), dtype=bool)
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    d_meas = depth[vi, ui]
    ok = in_img & (d_meas > 1e-3) & (np.abs(z - d_meas) < depth_tol)
    vis[ok] = True
    return vis, u, v, z, in_img


def point_to_patch(u, v, scale_x, scale_y):
    """Map full-res pixel (u,v) -> DINO patch index in the 518x518 resized image."""
    pu = np.clip((u * scale_x / PATCH_SIZE).astype(int), 0, PATCH_HW - 1)
    pv = np.clip((v * scale_y / PATCH_SIZE).astype(int), 0, PATCH_HW - 1)
    return pv * PATCH_HW + pu


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def retrieval_metrics(scores, gt_mask, ks=(50, 100, 200, 500)):
    """scores: [N] cosine sims, gt_mask: [N] bool target membership."""
    order = np.argsort(-scores)
    gt = gt_mask[order]
    out = {}
    n_gt = int(gt_mask.sum())
    for k in ks:
        out[f"prec@{k}"] = float(gt[:k].mean()) if k <= len(gt) else float("nan")
        out[f"rec@{k}"] = float(gt[:k].sum() / max(n_gt, 1))
    # average precision
    tp = np.cumsum(gt)
    prec = tp / (np.arange(len(gt)) + 1)
    out["AP"] = float((prec * gt).sum() / max(n_gt, 1))
    # best IoU over thresholds (sweep on rank)
    best_iou = 0.0
    for k in np.unique(np.linspace(1, len(gt), 50).astype(int)):
        pred = np.zeros(len(gt), dtype=bool)
        pred[order[:k]] = True
        inter = (pred & gt_mask).sum()
        union = (pred | gt_mask).sum()
        best_iou = max(best_iou, inter / max(union, 1))
    out["best_IoU"] = float(best_iou)
    return out


# --------------------------------------------------------------------------- #
# Data loaders (ScanNet reference -- ADAPT to your export layout)
# --------------------------------------------------------------------------- #
def load_scene(scene_dir):
    """Return dict(coord[N,3] world meters, color[N,3] 0-255, normal[N,3],
                   instance[N] int, semantic[N] int).
    Reference: a Pointcept-preprocessed ScanNet scene saved as .npy files.
    Adapt freely; only the returned keys matter downstream."""
    def _load(name, *alts):
        for n in (name, *alts):
            p = os.path.join(scene_dir, n)
            if os.path.isfile(p):
                return np.load(p)
        raise FileNotFoundError(f"{name} not found in {scene_dir}")
    coord = _load("coord.npy").astype(np.float64)
    color = _load("color.npy").astype(np.float32)
    normal = _load("normal.npy").astype(np.float32)
    instance = _load("instance.npy", "segment_instance.npy").astype(np.int64).reshape(-1)
    semantic = _load("segment.npy", "segment20.npy", "semantic.npy").astype(np.int64).reshape(-1)
    return dict(coord=coord, color=color, normal=normal,
                instance=instance, semantic=semantic)


def load_frame(scene_dir, frame_id):
    """Return dict(rgb[H,W,3] uint8, depth[H,W] float meters, K[3,3] depth-intrinsic,
                   cam2world[4,4]).  Reference: ScanNet `color/ depth/ pose/ intrinsic/`."""
    import imageio.v2 as imageio
    rgb = imageio.imread(os.path.join(scene_dir, "color", f"{frame_id}.jpg"))
    depth = imageio.imread(os.path.join(scene_dir, "depth", f"{frame_id}.png")).astype(np.float32)
    depth /= 1000.0  # ScanNet depth is millimetres
    K = np.loadtxt(os.path.join(scene_dir, "intrinsic", "intrinsic_depth.txt"))[:3, :3]
    cam2world = np.loadtxt(os.path.join(scene_dir, "pose", f"{frame_id}.txt"))
    return dict(rgb=rgb, depth=depth, K=K.astype(np.float64),
                cam2world=cam2world.astype(np.float64))


def dino_preprocess(rgb):
    """rgb[H,W,3] uint8 -> tensor[3,518,518] with DINO normalization. Returns
    (tensor, scale_x, scale_y) where scale maps full-res pixels to the 518 grid."""
    import torchvision.transforms.functional as TF
    H, W = rgb.shape[:2]
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = TF.resize(t, [CROP, CROP], antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = (t - mean) / std
    return t, CROP / W, CROP / H


# --------------------------------------------------------------------------- #
# Main probe over (frame, instance) pairs
# --------------------------------------------------------------------------- #
def run_probe(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, patch_proj = load_backbone_and_proj(args.pretrain_ckpt, device)
    if patch_proj is None:
        print("ERROR: checkpoint has no patch_proj -> the DINO-aligned space does "
              "not exist. Supply a full Utonia-v1m1 pretrain checkpoint. Aborting.")
        sys.exit(2)
    dino = load_dino(device)

    scene = load_scene(args.scene_dir)
    coord_w = scene["coord"]
    instance = scene["instance"]
    semantic = scene["semantic"]
    N = len(coord_w)
    print(f"[scene] {N} points, {len(np.unique(instance))} instances")

    # ---- 3D features in DINO-aligned space (per original point) ----
    transform = utonia.transform.default(args.scale)
    data = dict(coord=coord_w.copy().astype(np.float32),
                color=scene["color"].copy(),
                normal=scene["normal"].copy())
    data = transform(data)
    inverse = data.pop("inverse")
    if torch.is_tensor(inverse):
        inverse = inverse.to(device)
    F3D = extract_3d_features(model, patch_proj, data, inverse, device)  # [N,1536]
    F3D = F.normalize(F3D, dim=-1)
    print(f"[F3D] {tuple(F3D.shape)}")

    frame_ids = args.frames.split(",")
    rows = []
    for fid in frame_ids:
        frame = load_frame(args.scene_dir, fid)
        H, W = frame["depth"].shape
        vis, u, v, z, in_img = visible_mask(
            coord_w, frame["cam2world"], frame["K"], frame["depth"], H, W,
            depth_tol=args.depth_tol)
        if vis.sum() < 100:
            print(f"[frame {fid}] too few visible points ({vis.sum()}), skip")
            continue

        img_t, sx, sy = dino_preprocess(frame["rgb"])
        F2D = extract_dino_patches(dino, img_t, device)  # [1369,1536]
        F2D = F.normalize(F2D, dim=-1)

        patch_idx = point_to_patch(u, v, sx, sy)  # per-point patch (only valid where vis)

        # candidate target instances: visible AND of a "thing" class, enough points
        vis_inst, counts = np.unique(instance[vis], return_counts=True)
        targets = [int(i) for i, c in zip(vis_inst, counts)
                   if c >= args.min_inst_pts and i >= 0]
        targets = targets[: args.max_targets]

        for k in targets:
            gt_mask = (instance == k)
            sem_k = int(np.bincount(semantic[gt_mask]).argmax())

            # --- Stage 1 surrogate: pick the 2D patches that SEE instance k ---
            sel = vis & (instance == k)
            kp = np.unique(patch_idx[sel])
            if len(kp) < args.min_patches:
                continue
            q = F2D[torch.as_tensor(kp, device=device)].mean(0)
            q = F.normalize(q, dim=0)

            # --- (A) learned-alignment retrieval: cosine(q, F3D) ---
            s_align = (F3D @ q).detach().cpu().numpy()
            m_align = retrieval_metrics(s_align, gt_mask)

            # --- (B) single-frame geometric coverage (NOT an absolute ceiling) ---
            #   Back-projecting the selected patches recovers the part of instance k
            #   that is VISIBLE in this frame; it cannot recover occluded/out-of-frame
            #   parts.  Measured against the FULL instance mask, so:
            #     geo recall  = fraction of the instance this single frame sees.
            #     align beating geo recall = the alignment lit up parts NOT visible
            #                                here (the interesting, useful signal).
            geo_pred = np.zeros(N, dtype=bool)
            geo_pred[sel] = True  # visible instance-k map points (from depth+pose)
            m_geo = retrieval_metrics(_seed_score(geo_pred, coord_w, args.bp_radius), gt_mask)

            # --- diagnostics ---
            s_rand = (F3D @ F.normalize(torch.randn(DINO_DIM, device=device), dim=0)
                      ).detach().cpu().numpy()
            m_rand = retrieval_metrics(s_rand, gt_mask)

            # instance ambiguity: in top-K, fraction that are same-class but other-instance
            topk = np.argsort(-s_align)[: args.amb_k]
            same_class_other_inst = ((semantic[topk] == sem_k) & (instance[topk] != k)).mean()
            n_distinct_inst = len(np.unique(instance[topk][instance[topk] >= 0]))

            row = dict(frame=fid, inst=k, sem=sem_k, n_pts=int(gt_mask.sum()),
                       align_AP=m_align["AP"], align_IoU=m_align["best_IoU"],
                       align_prec100=m_align["prec@100"],
                       geo_AP=m_geo["AP"], geo_IoU=m_geo["best_IoU"],
                       rand_AP=m_rand["AP"],
                       amb_sameclass_otherinst=float(same_class_other_inst),
                       amb_distinct_inst=int(n_distinct_inst))
            rows.append(row)
            print(f"[{fid} inst{k} sem{sem_k}] "
                  f"align AP={row['align_AP']:.3f} IoU={row['align_IoU']:.3f} | "
                  f"geo AP={row['geo_AP']:.3f} | rand AP={row['rand_AP']:.3f} | "
                  f"amb(sameclass/other)={row['amb_sameclass_otherinst']:.2f} "
                  f"distinct={row['amb_distinct_inst']}")

    _summarize(rows, args.out_csv)


def _seed_score(seed_mask, coord, radius):
    """Turn a boolean seed set into a soft score: 1 inside, decaying by distance to
    the nearest seed point (cheap kNN-free proxy for back-projected region growth)."""
    from scipy.spatial import cKDTree
    if seed_mask.sum() == 0:
        return np.zeros(len(coord))
    tree = cKDTree(coord[seed_mask])
    d, _ = tree.query(coord, k=1)
    return np.exp(-d / max(radius, 1e-6))


def _summarize(rows, out_csv):
    if not rows:
        print("No (frame,instance) pairs evaluated. Check data paths / thresholds.")
        return
    import csv
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    arr = lambda k: np.array([r[k] for r in rows], float)
    print("\n==================== SUMMARY ({} pairs) ====================".format(len(rows)))
    for k in ("align_AP", "align_IoU", "geo_AP", "geo_IoU", "rand_AP",
              "amb_sameclass_otherinst", "amb_distinct_inst"):
        v = arr(k)
        print(f"  {k:24s} mean={np.nanmean(v):.3f}  median={np.nanmedian(v):.3f}")
    a, g, r = arr("align_AP"), arr("geo_AP"), arr("rand_AP")
    print("\n  Interpretation hints:")
    print(f"    learned-alignment vs chance:   {np.nanmean(a):.3f} vs {np.nanmean(r):.3f} "
          f"({'PASS' if np.nanmean(a) > 3 * np.nanmean(r) else 'WEAK'})")
    print(f"    learned-alignment vs geometry: {np.nanmean(a):.3f} vs {np.nanmean(g):.3f} "
          f"(ratio {np.nanmean(a) / max(np.nanmean(g), 1e-6):.2f})")
    print(f"    instance ambiguity (same-class/other-inst in top-K): "
          f"{np.nanmean(arr('amb_sameclass_otherinst')):.2f} "
          f"({'LOW (good)' if np.nanmean(arr('amb_sameclass_otherinst')) < 0.2 else 'HIGH (duplicates pollute)'})")
    print(f"\n  CSV written to {out_csv}")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrain_ckpt", required=True,
                   help="Full Utonia-v1m1 pretrain .pth (backbone + patch_proj).")
    p.add_argument("--scene_dir", required=True,
                   help="Scene dir (see load_scene/load_frame for layout).")
    p.add_argument("--frames", default="0",
                   help="Comma-separated frame ids to use as 2D queries.")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Coord scale for utonia.transform.default; MATCH your extraction.")
    p.add_argument("--depth_tol", type=float, default=0.05, help="z-buffer tol (m).")
    p.add_argument("--min_inst_pts", type=int, default=200)
    p.add_argument("--min_patches", type=int, default=3)
    p.add_argument("--max_targets", type=int, default=20)
    p.add_argument("--bp_radius", type=float, default=0.10,
                   help="Back-projection region-growth radius (m) for the geo ceiling.")
    p.add_argument("--amb_k", type=int, default=200, help="top-K for ambiguity diagnostic.")
    p.add_argument("--out_csv", default="viewpoint_probe_results.csv")
    return p


if __name__ == "__main__":
    run_probe(build_argparser().parse_args())
