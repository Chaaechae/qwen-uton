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
| **Checkpoints** | **`patch_proj` lives ONLY in a full pretrain ckpt (e.g. stagev2).** The released `utonia.pth` is backbone-only (standalone format, no `patch_proj`) so it cannot define the aligned space by itself. Use `--backbone_ckpt` for the backbone (utonia.pth *or* a pretrain ckpt) and `--patch_proj_ckpt` for a pretrain ckpt that holds `patch_proj`. Full pretrain ckpts are loaded with `weights_only=False` (they carry optimizer/EMA state), which fixes the `UnpicklingError: Weights only load failed`. |
| **DINOv2 teacher** | `facebook/dinov2-with-registers-giant` (auto-downloaded). |
| **Preprocessed scene** | `scene_dir` with `coord/color/normal/segment20/instance.npy`; `image_dir` with `color/<f>.png` + `correspondence/<f>.npy`. depth/pose/intrinsic are **not** needed (correspondence is already raycast/occlusion-aware). |
| **Env** | An interpreter that has the utonia deps (`torch`, `spconv`, `flash_attn`, `torch_scatter`, `transformers`, `timm`, `huggingface_hub`) + `scipy`, `imageio`, `torchvision`. **No `conda activate` needed** — see Run. The script auto-adds the repo root to `sys.path` (same as the repo's `export PYTHONPATH=./` convention), so `pip install -e .` is not required either. |

## Run (no `conda activate`)

Same convention as the repo's demos (`export PYTHONPATH=./` from the repo root),
or just call the interpreter that has the deps by its absolute path. The script also
puts the repo root on `sys.path` itself, so the path part needs no setup.

```bash
# Option A — repo convention, run from repo root:
export PYTHONPATH=./
python tools/viewpoint_probe/probe_2d3d_alignment.py        # defaults to your paths

# Option B — point straight at the env's python, no activate, run from anywhere:
/path/to/envs/utonia/bin/python tools/viewpoint_probe/probe_2d3d_alignment.py

# explicit args (defaults already match your layout):
python tools/viewpoint_probe/probe_2d3d_alignment.py \
  --backbone_ckpt /group-volume/Utonia/pretrain-utonia-v1m1-0-base-stagev2.pth \
  --scene_dir     /group-volume/3Ddataset/data/scannet/val/scene0011_00 \
  --frames        0,300,600          # omit / "" = all frames in the scene
```

The default uses the **stagev2 full pretrain ckpt for both the backbone and
`patch_proj`** (the matched pair). `patch_proj` is read from the same ckpt unless
`--patch_proj_ckpt` is given. The ckpt is loaded with `weights_only=False`, fixing
the `UnpicklingError: Weights only load failed`.

### A — stability across scenes/frames

One scene can be a fluke. Aggregate over many:

```bash
# several named scenes (siblings of --scene_dir):
python tools/viewpoint_probe/probe_2d3d_alignment.py \
  --scenes scene0011_00,scene0050_00,scene0231_00 --max_frames 8

# or glob every val scene, ~8 frames each:
python tools/viewpoint_probe/probe_2d3d_alignment.py \
  --scene_glob '/group-volume/3Ddataset/data/scannet/val/scene*' --max_frames 8
```

The SUMMARY then reflects all `(scene, frame, instance)` pairs; the CSV gains a
`scene` column. Watch whether `align_AP`, `occ_lift`, and `amb` hold up at scale.

It also reports `box_AP` (mask vs loose-box Stage-1 penalty) and a **box-jitter
robustness curve** — the box is expanded + randomly shifted by `--jitter_levels`
(default `0.25,0.5`) to mimic the loose / mis-aligned boxes a real detector (Qwen-VL)
produces, with mean and max aggregation. This is the cheap proxy for "will a real
detector box still work?" before wiring up an actual VLM:

```bash
python tools/viewpoint_probe/probe_2d3d_alignment.py \
  --scene_glob '/.../scannet/val/scene*' --max_frames 8 --jitter_levels 0.25,0.5
```

Read it as `tight=… j0.25=… j0.5=… j0.5_max=…`: how AP decays as the box gets looser/
offset, and how much per-point **max aggregation** (`--agg max` in the demo) recovers.

## Stage-1 → 3D demo (B): localize from a real 2D image, no GT

`localize_from_2d.py` runs the actual pipeline the probe validated, but Stage-1 is
**image-only** (no GT correspondence). It writes a heatmap point cloud + top-K indices,
and (optionally) AP/IoU vs a GT instance so you can read the degradation from the
probe's GT-surrogate numbers.

```bash
# point prompt: click a pixel, grow by DINO self-similarity
python tools/viewpoint_probe/localize_from_2d.py --frame 300 \
  --mode point --point 640 360 --eval_instance 12 --out /tmp/loc

# box from any external detector (incl. Qwen-VL in this repo):
python tools/viewpoint_probe/localize_from_2d.py --frame 300 \
  --mode box --box 410 220 690 540 --out /tmp/loc

# detector dump (text -> box): {frame_id: [{label, box:[x0,y0,x1,y1]}]}
python tools/viewpoint_probe/localize_from_2d.py --frame 300 \
  --mode json --boxes_json dets.json --text chair --out /tmp/loc

# open-vocab text (EXPERIMENTAL, needs `pip install open_clip_torch`):
python tools/viewpoint_probe/localize_from_2d.py --frame 300 \
  --mode text --text "a chair" --tau 0.6 --out /tmp/loc
```

Modes: `box`/`json` = realistic 2D-detector boxes; `auto` = box auto-derived from
`--eval_instance`'s correspondence (detector stand-in, auto-picks a frame that shows
it); `auto_mask` = pixel-accurate patches (== the probe's selection) to isolate the
box penalty; `point` = click + DINO self-similarity; `text` = experimental open-vocab.

### The bounding-box penalty (important)

On scene0011 the probe's pixel-perfect selection gives align AP ≈ 0.31, but a loose
`auto` **box** drops to ≈ 0.06 with prec@100 = 0 — the box mixes floor/wall/other
objects into the query, so a mean query matches large background structures. The
alignment is fine; the **bbox Stage-1 is the bottleneck**. Mitigations:

- `--mode auto_mask` — confirms the pipeline (should recover ≈ the probe number).
- `--agg max` — per-point max cosine over patches; robust to a few background patches.
- `--fg` — 2-means foreground filter inside the box (keeps the central cluster).
- Real fix: a **mask** (SAM / segmentation) Stage-1, not just a box.

The `[stage-1 diag]` line prints which GT instances actually sit under the selected
patches — use it to see box contamination. Open `<out>.ply` to see the 3D heatmap.

If `utonia` imports but a dependency is missing, the script tells you exactly which
package and which interpreter — no `conda activate` involved.

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
