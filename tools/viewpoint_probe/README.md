# Viewpoint probe — is Utonia's 2D↔3D alignment good enough for text/2D→3D localization?

A **feasibility test** to run *before* building the full "find the object in the 3D
map from a 2D image" pipeline. It measures, quantitatively, whether the learned
2D-3D alignment (`enc2d` distillation → `patch_proj`) lets you localize an object in
the 3D map by matching it to the **DINO feature of that object in one 2D image**,
*without* using camera pose/depth for the lift.

> ⚠️ Not yet executed. This was written against the repo's APIs and configs but
> could not be run in the authoring environment (no GPU / torch / data / checkpoint).
> Treat the first run as a bring-up: expect to fix data-path and intrinsic details
> for your export layout. The numbers, not the code's existence, are the deliverable.

## What it does

For each `(frame, instance)`:

1. **3D side** — runs the frozen Utonia backbone on the scene point cloud, up-casts
   3 levels to the 1332-ch `enc2d` feature, applies the trained `patch_proj`
   (1332→1536) and `cos_shift`, giving per-point features **in the DINO space**.
2. **2D side** — runs DINOv2-giant-with-registers on the (posed) RGB frame → patch
   features in the same DINO space.
3. **Stage-1 surrogate** — selects the DINO patches that *see* instance `k`
   (via depth+pose+GT labels; this stands in for "a 2D detector found the object").
   Query vector `q` = mean of those patch features.
4. **(A) learned-alignment retrieval** — `cosine(q, F3D)` over all map points →
   rank → AP / best-IoU / prec@K against the **full** instance mask.
5. **(B) single-frame geometric coverage** — back-projects the selected patches; this
   recovers only the part of the instance visible in this frame. It is a *coverage*
   reference, **not** an absolute ceiling (see below).
6. **Diagnostics** — random-query chance floor; instance-ambiguity (how much of the
   top-K is the same semantic class but a *different* instance — the known weak spot
   of DINO-style features).

## Why these comparisons answer the question

- **align AP ≫ random AP** → the alignment carries real, accessible object signal →
  training-free 2-stage (2D detect → 3D cosine match) is realistic.
- **align recall > geo recall** (geo = single-frame coverage) → the alignment lights
  up parts of the instance **not visible in this frame** → this is exactly the
  "knows the occluded object's 3D location" capability you want.
- **instance-ambiguity HIGH** → cosine lights up *all* chairs, not *the* chair →
  you cannot disambiguate duplicates without geometry / a trained instance head.
  (Matches your earlier "localize all entities of a class" framing.)

## Prerequisites

| Need | Detail |
|---|---|
| **Full pretrain ckpt** | `Utonia-v1m1` checkpoint containing **both** `module.student.backbone.*` **and** `module.patch_proj.*`. The released standalone weight is backbone-only and will abort the script — the DINO-aligned space *only* exists after `patch_proj`. |
| **DINOv2 teacher** | `facebook/dinov2-with-registers-giant` (auto-downloaded by `transformers`). |
| **One posed scene** | point cloud (`coord/color/normal`) + GT `instance`/`semantic` + ≥1 frame (`rgb`, `depth[m]`, depth-intrinsic, `cam2world`). ScanNet/ScanNet++/Structured3D all work. |
| **Env** | the `utonia` conda env (`flash_attn`, `torch_scatter`); `pip install -e .` in repo root; plus `scipy`, `imageio`, `torchvision`. |

Adapt `load_scene` / `load_frame` in the script to your export layout — only the
returned dict keys matter.

## Run

```bash
python tools/viewpoint_probe/probe_2d3d_alignment.py \
  --pretrain_ckpt /path/to/utonia_v1m1_pretrain.pth \
  --scene_dir     /path/to/scenes/scene0011_00 \
  --frames        0,300,600,900 \
  --scale         1.0 \
  --out_csv       results_scene0011.csv
```

`--scale` **must match** the coord scale you used when extracting your stored 3D
features (the standalone demos use `utonia.transform.default(0.5)`).

## Reading the result

The summary prints means/medians and two verdict lines:

```
learned-alignment vs chance:   <align_AP> vs <rand_AP>   (PASS/WEAK)
learned-alignment vs geometry: <align_AP> vs <geo_AP>    (ratio ...)
instance ambiguity ...:        <frac>     (LOW (good) / HIGH (duplicates pollute))
```

Rough decision:

- **PASS + LOW ambiguity + align recall ≥ geo recall** → green light for the
  training-free 2-stage design; the alignment is the lift.
- **PASS but HIGH ambiguity** → works at the *class* level only; add geometry
  (viewing-direction constraint from the posed frame) or a light instance head to
  pick *the* specific object.
- **WEAK** → the DINO-distilled alignment is not discriminative enough at the patch
  granularity; fall back to pose+depth back-projection (you have posed frames), or
  re-distill with a richer/aligned teacher (RADIO) before relying on feature matching.

## Known limitations (by construction)

- Alignment is **patch-pooled and lossy** → coarse regions, not sharp boundaries.
- DINO features are **class-semantic, not instance-discriminative** → the ambiguity
  diagnostic exists precisely to quantify this.
- If your real query frames are **posed with depth**, back-projection is exact and
  strictly better than this feature matching — the matching path is only worth it for
  *unposed* query images.
- The "Stage-1" object selection here uses GT labels to isolate **alignment** quality
  from 2D-detection quality. Swap in your real 2D detector/CLIP step once alignment
  passes.
