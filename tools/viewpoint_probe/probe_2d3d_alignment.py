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

# --- import the standalone `utonia` package the same way the repo's demos do ---
# The README convention is `export PYTHONPATH=./` from the repo root (NO conda
# activate, NO pip install).  We replicate that here by putting the repo root on
# sys.path automatically, so the script runs from any cwd with any interpreter
# that has the deps -- you do not need to remember PYTHONPATH or to activate.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F

try:
    import utonia
except ModuleNotFoundError as e:
    if e.name in ("utonia",):
        sys.exit(
            f"Cannot import `utonia` (looked under repo root: {_REPO_ROOT}).\n"
            f"Run from the repo root, or check that {_REPO_ROOT}/utonia/ exists."
        )
    # utonia was found but one of ITS dependencies is missing -> env problem,
    # not a path problem. Tell the user exactly which, no conda activate needed.
    sys.exit(
        f"`utonia` is on the path but its dependency `{e.name}` is missing in this\n"
        f"interpreter ({sys.executable}).\n"
        f"Use the interpreter that has the utonia env's packages "
        f"(e.g. /path/to/envs/utonia/bin/python {__file__}) -- no `conda activate` "
        f"required -- or `pip install {e.name}` into it."
    )


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
#
# Two checkpoint formats exist:
#   * STANDALONE  (the released `utonia.pth`): {"config": {...}, "state_dict": {...}}
#       with BARE backbone keys ("embedding.*", "enc.*", ...). NO `patch_proj`.
#       -> load via the embedded config (same as utonia.load).
#   * FULL PRETRAIN (e.g. stagev2 `*.pth`): state_dict keys prefixed
#       "module.student.backbone.*" and "module.patch_proj.*". Contains the
#       trained `patch_proj` (1332->1536) -- the ONLY place the DINO-aligned space
#       lives. Saved with optimizer/EMA state, so it needs weights_only=False.
# --------------------------------------------------------------------------- #
def _torch_load(path):
    # weights_only=False: full pretrain ckpts carry non-tensor objects (addict
    # Dict / numpy / EMA bookkeeping) that the torch>=2.6 default rejects.
    return torch.load(path, map_location="cpu", weights_only=False)


def _state_dict(ckpt):
    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt


def _is_standalone(sd):
    return (any(k.startswith(("embedding.", "enc.", "dec.")) for k in sd)
            and not any(k.startswith(("module.", "student.", "teacher.")) for k in sd))


def load_backbone(backbone_ckpt, device):
    ckpt = _torch_load(backbone_ckpt)
    sd = _state_dict(ckpt)
    if _is_standalone(sd):
        cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
        if cfg is None:
            raise RuntimeError(f"{backbone_ckpt}: standalone format but no 'config'.")
        model = utonia.model.PointTransformerV3(**cfg)
        model.load_state_dict(sd)
        print(f"[backbone] standalone format from embedded config "
              f"(enc_mode={cfg.get('enc_mode')}, traceable={cfg.get('traceable')}, "
              f"in_channels={cfg.get('in_channels')})")
        if not cfg.get("traceable", False):
            print("  WARNING: traceable=False -> no pooling chain for up_cast. "
                  "Use a full pretrain ckpt as --backbone_ckpt instead.")
    else:
        bsd = {}
        for k, v in sd.items():
            kk = k[len("module."):] if k.startswith("module.") else k
            if kk.startswith("student.backbone."):
                bsd[kk[len("student.backbone."):]] = v
            elif kk.startswith("backbone.") and "teacher" not in kk:
                bsd[kk[len("backbone."):]] = v
        if not bsd:
            raise RuntimeError(
                f"{backbone_ckpt}: no backbone weights. prefixes seen: "
                f"{sorted({k.split('.')[0] for k in sd})[:8]}")
        model = utonia.model.PointTransformerV3(**BACKBONE_CONFIG)
        missing, unexpected = model.load_state_dict(bsd, strict=False)
        print(f"[backbone] pretrain student.backbone loaded; "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


def load_patch_proj(pp_ckpt, device):
    """patch_proj lives ONLY in a full pretrain ckpt. Returns the Linear or exits."""
    sd = _state_dict(_torch_load(pp_ckpt))
    pp = {}
    for k, v in sd.items():
        kk = k[len("module."):] if k.startswith("module.") else k
        if kk.startswith("patch_proj."):
            pp[kk[len("patch_proj."):]] = v
    if not pp:
        print(f"ERROR: no `patch_proj` in {pp_ckpt}. The DINO-aligned space does not\n"
              f"exist without it. The released utonia.pth is backbone-only -- point\n"
              f"--patch_proj_ckpt at a FULL pretrain checkpoint (e.g. stagev2).")
        sys.exit(2)
    out_c, in_c = pp["weight"].shape           # [DINO_DIM, BACKBONE_OUT_CHANNELS]
    lin = torch.nn.Linear(in_c, out_c)
    lin.load_state_dict(pp)
    if (in_c, out_c) != (BACKBONE_OUT_CHANNELS, DINO_DIM):
        print(f"  WARNING: patch_proj is {in_c}->{out_c}, expected "
              f"{BACKBONE_OUT_CHANNELS}->{DINO_DIM}. Check enc2d_upcast_level / dims.")
    print(f"[patch_proj] loaded {in_c}->{out_c} (faithful aligned space).")
    return lin.to(device).eval()


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


def _ap_only(scores, gt_mask):
    g = gt_mask[np.argsort(-scores)]
    n = int(gt_mask.sum())
    if n == 0:
        return float("nan")
    tp = np.cumsum(g)
    prec = tp / (np.arange(len(g)) + 1)
    return float((prec * g).sum() / n)


def _max_sim_scores(F3D, Fk, chunk=256):
    """Per-3D-point max cosine over patch features Fk[n,C], memory-safe."""
    best = torch.full((F3D.shape[0],), -1e9, device=F3D.device)
    for i in range(0, Fk.shape[0], chunk):
        best = torch.maximum(best, (F3D @ Fk[i:i + chunk].T).max(dim=1).values)
    return best.detach().cpu().numpy()


def _box_patches(pu0, pu1, pv0, pv1):
    gu, gv = np.meshgrid(np.arange(min(pu0, pu1), max(pu0, pu1) + 1),
                         np.arange(min(pv0, pv1), max(pv0, pv1) + 1))
    return np.unique((gv * PATCH_HW + gu).reshape(-1))


def jittered_box_patches(pu0, pu1, pv0, pv1, frac, rng):
    """Detector-noise box: expand each side by frac*size and random-shift by ~frac*size."""
    bw, bh = pu1 - pu0 + 1, pv1 - pv0 + 1
    padu, padv = int(round(frac * bw)), int(round(frac * bh))
    su = int(round(rng.uniform(-frac, frac) * bw))
    sv = int(round(rng.uniform(-frac, frac) * bh))
    cl = lambda x: int(np.clip(x, 0, PATCH_HW - 1))
    return _box_patches(cl(pu0 - padu + su), cl(pu1 + padu + su),
                        cl(pv0 - padv + sv), cl(pv1 + padv + sv))


def _fg_patches(F2D, kpb, device):
    """Foreground filter inside a box: 2-means on DINO features, keep the cluster
    nearest the box centroid (object usually occupies the box interior)."""
    if len(kpb) < 6:
        return kpb
    feats = F2D[torch.as_tensor(kpb, device=device)]            # [n,C] normalized
    pu, pv = kpb % PATCH_HW, kpb // PATCH_HW
    cu, cv = pu.mean(), pv.mean()
    g = feats @ feats.T
    i = int(g.sum(1).argmin()); j = int(g[i].argmin())
    c = torch.stack([feats[i], feats[j]])
    for _ in range(5):
        a = (feats @ c.T).argmax(1)
        for t in (0, 1):
            if (a == t).any():
                c[t] = F.normalize(feats[a == t].mean(0), dim=0)
    a = a.cpu().numpy()
    d = (pu - cu) ** 2 + (pv - cv) ** 2
    keep = 0 if d[a == 0].mean() <= d[a == 1].mean() else 1
    return kpb[a == keep]


# --------------------------------------------------------------------------- #
# Reusable: 3D features in the DINO-aligned space for one scene
# --------------------------------------------------------------------------- #
def compute_scene_F3D(scene_dir, model, patch_proj, device, scale):
    """Returns (F3D[N,1536] L2-normed, scene dict). Original-point order == the
    point index used by correspondence files. Shared by the probe and the demo."""
    scene = load_scene(scene_dir)
    data = utonia.transform.default(scale)(dict(
        coord=scene["coord"].astype(np.float32).copy(),
        color=scene["color"].copy(), normal=scene["normal"].copy()))
    inverse = data.pop("inverse")
    if torch.is_tensor(inverse):
        inverse = inverse.to(device)
    F3D = F.normalize(extract_3d_features(model, patch_proj, data, inverse, device), dim=-1)
    return F3D, scene


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _resolve_scenes(args):
    if args.scenes:
        root = os.path.dirname(os.path.normpath(args.scene_dir))
        return [os.path.join(root, n) for n in args.scenes.split(",")]
    if args.scene_glob:
        return sorted(glob.glob(args.scene_glob))
    return [args.scene_dir]


def run_probe(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_backbone(args.backbone_ckpt, device)
    # patch_proj defaults to the backbone ckpt (works when that is a full pretrain
    # ckpt); for the released utonia.pth you MUST pass --patch_proj_ckpt <stagev2>.
    patch_proj = load_patch_proj(args.patch_proj_ckpt or args.backbone_ckpt, device)
    dino = load_dino(device)

    scene_dirs = _resolve_scenes(args)
    print(f"[scenes] {len(scene_dirs)}")
    rows = []
    for scene_dir in scene_dirs:
        try:
            probe_one_scene(scene_dir, model, patch_proj, dino, device, args, rows)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[scene {os.path.basename(scene_dir)}] skip ({e})")
    _summarize(rows, args.out_csv)


def probe_one_scene(scene_dir, model, patch_proj, dino, device, args, rows):
    image_dir = args.image_dir or default_image_dir(scene_dir)
    name = os.path.basename(os.path.normpath(scene_dir))
    F3D, scene = compute_scene_F3D(scene_dir, model, patch_proj, device, args.scale)
    coord_w, instance, semantic = scene["coord"], scene["instance"], scene["semantic"]
    N = len(coord_w)
    print(f"[{name}] {N} pts, {len(np.unique(instance[instance >= 0]))} inst, "
          f"F3D={tuple(F3D.shape)}")

    frames = args.frames.split(",") if args.frames else list_frames(image_dir)
    if args.max_frames > 0 and len(frames) > args.max_frames:        # even subsample
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, args.max_frames).astype(int)]

    for fid in frames:
        try:
            rgb, corr = load_frame_corr(image_dir, fid)
        except FileNotFoundError as e:
            print(f"[frame {fid}] skip ({e})")
            continue
        if corr.ndim != 2 or corr.shape[1] < 3 or corr.shape[0] < 50 or (corr < 0).all():
            continue
        # correspondence columns: [px, py, (1,) , point_idx]. The homogeneous "1"
        # column is present in some preprocessor versions (M,4) and absent in others
        # (M,3). point index is ALWAYS the last column; pixel x,y the first two.
        px, py, pidx = corr[:, 0], corr[:, 1], corr[:, -1].astype(np.int64)
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
            sem_pts = semantic[gt_mask & (semantic >= 0)]
            sem_k = int(np.bincount(sem_pts).argmax()) if len(sem_pts) else -1

            sel_rows = inst_of_corr == k                       # corr rows seeing inst k
            kp = np.unique(patch[sel_rows])
            if len(kp) < args.min_patches:
                continue
            q = F.normalize(F2D[torch.as_tensor(kp, device=device)].mean(0), dim=0)

            s_align = (F3D @ q).detach().cpu().numpy()          # (A) learned alignment
            m_align = retrieval_metrics(s_align, gt_mask)

            # (A') BOX variant -- detector stand-in: a bbox over the instance's pixels.
            #   box_AP (tight, mean query) quantifies the bbox Stage-1 penalty, and the
            #   JITTER sweep (expand+shift, mean & max agg) tests robustness to the
            #   loose / mis-aligned boxes a real detector (e.g. Qwen-VL) produces.
            pu0 = int(np.clip(px[sel_rows].min() * sx / PATCH_SIZE, 0, PATCH_HW - 1))
            pu1 = int(np.clip(px[sel_rows].max() * sx / PATCH_SIZE, 0, PATCH_HW - 1))
            pv0 = int(np.clip(py[sel_rows].min() * sy / PATCH_SIZE, 0, PATCH_HW - 1))
            pv1 = int(np.clip(py[sel_rows].max() * sy / PATCH_SIZE, 0, PATCH_HW - 1))

            def _box_ap(kpb, agg):
                Fk = F2D[torch.as_tensor(kpb, device=device)]
                if agg == "max":
                    return _ap_only(_max_sim_scores(F3D, Fk), gt_mask)
                q_ = F.normalize(Fk.mean(0), dim=0)
                return _ap_only((F3D @ q_).detach().cpu().numpy(), gt_mask)

            box_AP = _box_ap(_box_patches(pu0, pu1, pv0, pv1), "mean")  # tight, mean
            rng = np.random.default_rng(abs(hash((name, fid, k))) % (2**32))
            jit = {}  # mean-agg AP at each jitter level (+ fg-filtered at the largest)
            for L in args.jitter_levels:
                kpj = jittered_box_patches(pu0, pu1, pv0, pv1, L, rng)
                jit[f"boxj{L:g}_AP"] = _box_ap(kpj, "mean")
            Lmax = max(args.jitter_levels) if args.jitter_levels else 0.0
            kpj = jittered_box_patches(pu0, pu1, pv0, pv1, Lmax, rng)
            # foreground filter recovery (the real lever; max-agg makes loose boxes worse)
            jit[f"boxj{Lmax:g}_fg_AP"] = _box_ap(_fg_patches(F2D, kpj, device), "mean")

            s_rand = (F3D @ F.normalize(torch.randn(DINO_DIM, device=device), dim=0)
                      ).detach().cpu().numpy()
            m_rand = retrieval_metrics(s_rand, gt_mask)

            vis_k = np.zeros(N, dtype=bool)                     # (B) frame coverage
            vis_k[np.unique(pidx[sel_rows])] = True
            m_geo = retrieval_metrics(_seed_score(vis_k, coord_w, args.bp_radius), gt_mask)

            # (C) OCCLUDED-SUBSET AP -- the decisive test for "finds the part of the
            #     object NOT visible in this frame". Rank only {occluded_k U background}
            #     (visible_k excluded); positives = occluded_k. occ_AP >> occ_chance
            #     => alignment lifts unseen object parts above background.
            occ_mask = gt_mask & ~vis_k
            restrict = occ_mask | (~gt_mask)
            occ_n = int(occ_mask.sum())
            if occ_n >= args.min_occ_pts and restrict.sum() > occ_n:
                occ_AP = retrieval_metrics(s_align[restrict], occ_mask[restrict], ks=(50,))["AP"]
                occ_chance = retrieval_metrics(s_rand[restrict], occ_mask[restrict], ks=(50,))["AP"]
            else:
                occ_AP = occ_chance = float("nan")
            occ_lift = occ_AP / occ_chance if occ_chance and occ_chance > 0 else float("nan")

            topk = np.argsort(-s_align)[: args.amb_k]
            amb = ((semantic[topk] == sem_k) & (instance[topk] != k)).mean()
            ndist = len(np.unique(instance[topk][instance[topk] >= 0]))

            row = dict(
                scene=name, frame=fid, inst=k, sem=sem_k, n_pts=int(gt_mask.sum()),
                vis_pts=int(vis_k.sum()), occ_pts=occ_n,
                align_AP=m_align["AP"], align_IoU=m_align["best_IoU"],
                align_rec200=m_align["rec@200"], box_AP=box_AP,
                geo_AP=m_geo["AP"], geo_rec200=m_geo["rec@200"],
                rand_AP=m_rand["AP"],
                occ_AP=occ_AP, occ_chance=occ_chance, occ_lift=occ_lift,
                amb_sameclass_otherinst=float(amb), amb_distinct_inst=int(ndist))
            row.update(jit)
            rows.append(row)
            jit_str = " ".join(f"{kk.replace('_AP','').replace('box','b')}={vv:.3f}"
                               for kk, vv in jit.items())
            print(f"[{name} {fid} inst{k} sem{sem_k} n={row['n_pts']} "
                  f"vis={row['vis_pts']} occ={occ_n}] "
                  f"align AP={m_align['AP']:.3f} IoU={m_align['best_IoU']:.3f} | "
                  f"box AP={box_AP:.3f} jit[{jit_str}] | "
                  f"occ AP={occ_AP:.3f} lift={occ_lift:.2f} | amb={amb:.2f}")

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
    for k in ("align_AP", "box_AP", "align_IoU", "align_rec200", "geo_AP",
              "geo_rec200", "rand_AP", "occ_AP", "occ_chance", "occ_lift",
              "amb_sameclass_otherinst", "amb_distinct_inst"):
        v = arr(k)
        print(f"  {k:26s} mean={np.nanmean(v):.3f} median={np.nanmedian(v):.3f}")
    a, g, r = arr("align_AP"), arr("geo_AP"), arr("rand_AP")
    bx = arr("box_AP")
    amb = arr("amb_sameclass_otherinst")
    oa, ol = arr("occ_AP"), arr("occ_lift")
    print("\n  Verdict:")
    print(f"    alignment vs chance:   {np.nanmean(a):.3f} vs {np.nanmean(r):.3f}  "
          f"-> {'PASS' if np.nanmean(a) > 3 * np.nanmean(r) else 'WEAK'}")
    print(f"    mask vs box Stage-1:   align {np.nanmean(a):.3f} vs box "
          f"{np.nanmean(bx):.3f}  (box keeps "
          f"{100 * np.nanmean(bx) / max(np.nanmean(a), 1e-6):.0f}% -- the rest is the "
          f"bounding-box penalty)")
    print(f"    OCCLUDED-part recovery: occ_AP {np.nanmean(oa):.3f} vs chance "
          f"{np.nanmean(arr('occ_chance')):.3f} (lift {np.nanmean(ol):.2f})  "
          f"-> {'RECOVERS unseen parts' if np.nanmean(ol) > 2 else 'mostly visible-region only'}")
    print(f"    instance ambiguity: {np.nanmean(amb):.2f}  "
          f"-> {'LOW (good)' if np.nanmean(amb) < 0.2 else 'HIGH (duplicates pollute)'}")
    # box-jitter robustness curve (detector-noise simulation)
    jcols = [k for k in rows[0] if k.startswith("boxj")]
    if jcols:
        print("    box-jitter robustness (mean AP; detector-noise sim):")
        print(f"      tight={np.nanmean(bx):.3f}  "
              + "  ".join(f"{c.replace('box','').replace('_AP','')}={np.nanmean(arr(c)):.3f}"
                          for c in jcols))
        fc = [c for c in jcols if c.endswith("fg_AP")]
        mn = [c for c in jcols if c.endswith("_AP") and not c.endswith("fg_AP")]
        if fc and mn:
            print(f"      -> foreground filter at largest jitter: "
                  f"{np.nanmean(arr(fc[0])):.3f} vs plain {np.nanmean(arr(mn[-1])):.3f} "
                  f"(fg {'helps' if np.nanmean(arr(fc[0])) > np.nanmean(arr(mn[-1])) else 'no gain'})")
    print(f"\n  CSV -> {out_csv}")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Default: stagev2 full pretrain ckpt for BOTH backbone + patch_proj (matched pair).
    p.add_argument("--backbone_ckpt",
                   default="/group-volume/Utonia/pretrain-utonia-v1m1-0-base-stagev2.pth",
                   help="backbone weights. A full pretrain ckpt (stagev2) holds both "
                        "the student backbone and patch_proj. The released utonia.pth "
                        "is backbone-only (no patch_proj).")
    p.add_argument("--patch_proj_ckpt", default=None,
                   help="ckpt holding `patch_proj`; defaults to --backbone_ckpt. "
                        "Only needed separately if backbone comes from utonia.pth.")
    p.add_argument("--scene_dir",
                   default="/group-volume/3Ddataset/data/scannet/val/scene0011_00")
    p.add_argument("--image_dir", default=None,
                   help="defaults to <root>/images/<split>/<scene> derived from scene_dir")
    p.add_argument("--scenes", default="",
                   help="comma-separated scene NAMES (siblings of --scene_dir) to "
                        "aggregate over, e.g. scene0011_00,scene0050_00 (A: stability)")
    p.add_argument("--scene_glob", default="",
                   help="glob of scene dirs to aggregate, e.g. '/.../scannet/val/scene*'")
    p.add_argument("--frames", default="",
                   help="comma-separated frame ids; empty = all frames in the scene")
    p.add_argument("--max_frames", type=int, default=0,
                   help="even-subsample to at most this many frames per scene (0=all)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="utonia.transform.default coord scale; MATCH your extraction")
    p.add_argument("--min_inst_pts", type=int, default=200)
    p.add_argument("--min_occ_pts", type=int, default=50,
                   help="min occluded points to score the occluded-subset AP")
    p.add_argument("--min_patches", type=int, default=3)
    p.add_argument("--max_targets", type=int, default=20)
    p.add_argument("--bp_radius", type=float, default=0.10)
    p.add_argument("--amb_k", type=int, default=200)
    p.add_argument("--jitter_levels", default=[0.25, 0.5],
                   type=lambda s: [float(x) for x in s.split(",") if x != ""],
                   help="box expand+shift fractions for detector-noise robustness "
                        "(comma list, e.g. 0.25,0.5). Adds boxj* columns.")
    p.add_argument("--out_csv", default="viewpoint_probe_results.csv")
    return p


if __name__ == "__main__":
    run_probe(build_argparser().parse_args())
