# Viewpoint probe — is Utonia's 2D↔3D alignment good enough for 2D→3D localization?

A **feasibility test** to run *before* building the full "find the object in the 3D
map from a 2D image" pipeline. It measures whether the learned 2D-3D alignment
(`enc2d` distillation → `patch_proj`) lets you localize an object in the 3D map by
matching it to the **DINO feature of that object in one 2D image**, *without* using
camera pose/depth for the lift.

> ⚠️ Not yet executed in the authoring environment (no GPU/torch/data/ckpt there).
> Written against this repo's APIs + the concerto ScanNet preprocessing layout.
> Treat the first run as bring-up. The numbers, not the code's existence, are the
> deliverable.

## What it does (per `frame, instance`)

1. **3D side** — frozen backbone on the point cloud → up-cast 3 levels (1332-ch
   `enc2d` feature) → trained `patch_proj` (1332→1536) → `cos_shift` ⇒ per-point
   features **in the DINO space**.
2. **2D side** — DINOv2-giant-with-registers on the raycast RGB frame → patch
   features in the same DINO space.
3. **Stage-1 surrogate** — uses the preprocessor's **raycast correspondence**
   (`correspondence/<f>.npy`, occlusion-aware) to pick the DINO patches that *see*
   instance `k`. Query `q` = mean of those patch features. (This isolates *alignment*
   quality from 2D-detector quality; swap in a real CLIP/detector later.)
4. **(A) learned-alignment retrieval** — `cosine(q, F3D)` over all map points →
   AP / best-IoU / recall@200 vs the **full** instance mask.
5. **(B) single-frame geometric coverage** — the raycast-visible part of the
   instance. A coverage reference, **not** an absolute ceiling: it cannot include
   parts occluded/out-of-frame in this view.
6. **Diagnostics** — random-query chance floor; instance-ambiguity (how much of the
   top-K is the same semantic class but a *different* instance).

## Why these comparisons answer the question

- **align AP ≫ random AP** → alignment carries accessible object signal →
  training-free 2-stage (2D detect → 3D cosine match) is realistic.
- **align recall@200 > geo recall@200** → alignment lit up instance parts **not
  visible in this frame** → the "knows the occluded object's 3D location" capability.
- **instance-ambiguity HIGH** → cosine lights up *all* chairs, not *the* chair →
  duplicates can't be resolved without geometry / a trained instance head.

## Prerequisites

| Need | Detail |
|---|---|
| **Full pretrain ckpt** | `Utonia-v1m1` with **both** `module.student.backbone.*` and `module.patch_proj.*`. Backbone-only weights abort the script (the aligned space only exists after `patch_proj`). Default path: `/group-volume/Utonia/utonia.pth`. |
| **DINOv2 teacher** | `facebook/dinov2-with-registers-giant` (auto-downloaded). |
| **Preprocessed scene** | `scene_dir` with `coord/color/normal/segment20/instance.npy`; `image_dir` with `color/<f>.png` + `correspondence/<f>.npy`. depth/pose/intrinsic are **not** needed (correspondence is already raycast/occlusion-aware). |
| **Env** | `utonia` conda env (`flash_attn`, `torch_scatter`) + `scipy`, `imageio`, `torchvision`. The script auto-adds the repo root to `sys.path`, so `pip install -e .` is optional. |

## Run

Defaults already point at your paths:

```bash
python tools/viewpoint_probe/probe_2d3d_alignment.py
# == explicit ==
python tools/viewpoint_probe/probe_2d3d_alignment.py \
  --pretrain_ckpt /group-volume/Utonia/utonia.pth \
  --scene_dir     /group-volume/3Ddataset/data/scannet/val/scene0011_00 \
  --frames        ""          # empty = all frames in the scene \
  --out_csv       results_scene0011.csv
```

`image_dir` is auto-derived as `<root>/images/<split>/<scene>`
(→ `/group-volume/3Ddataset/data/scannet/images/val/scene0011_00`); override with
`--image_dir` if your layout differs.

`--scale` must match the coord scale used when you extracted your stored features
(standalone demos use `utonia.transform.default(0.5)`; default here is `1.0`).

## Reading the result

```
alignment vs chance:   <align_AP> vs <rand_AP>   (PASS/WEAK)
alignment vs frame-coverage (recall@200): <ar> vs <gr>
instance ambiguity: <frac>   (LOW (good) / HIGH (duplicates pollute))
```

- **PASS + LOW ambiguity + align recall ≥ geo recall** → green light for the
  training-free 2-stage design.
- **PASS but HIGH ambiguity** → works at the *class* level only; add a
  viewing-direction constraint from the posed frame, or a light instance head, to
  pick *the* specific object.
- **WEAK** → DINO-distilled alignment isn't discriminative enough at patch
  granularity; fall back to pose+depth back-projection, or re-distill with a richer
  teacher (RADIO) before relying on feature matching.

## Known limitations (by construction)

- Alignment is **patch-pooled and lossy** → coarse regions, not sharp boundaries.
- DINO features are **class-semantic, not instance-discriminative** → the ambiguity
  diagnostic quantifies exactly this.
- The Stage-1 selection uses correspondence (≈ ground truth) to isolate alignment
  quality. Replace with your real 2D detector/CLIP step once alignment passes.
