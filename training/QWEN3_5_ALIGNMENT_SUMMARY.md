# Utonia ↔ Qwen3.5-VL alignment: experiment summary

Final record for the cleaned-up branch. It documents why we re-aligned the
Utonia point encoder to the **Qwen3.5-VL vision tower** (instead of the original
DINOv2 target), the A→I ablation sweep that led to the two kept recipes
(**H** and **I**), how they were evaluated, the result, and the analysis.

The two kept recipes live in `training/configs/utonia/`:

| file | what it is |
|---|---|
| `distill-utonia-v1m3-H-indoor-qwen3_5-4b.py` | merged-scale alignment, SSL **off** |
| `distill-utonia-v1m3-I-indoor-qwen3_5-4b.py` | H **+ light SSL** re-enabled — **this is the evaluated checkpoint** |

Model module: `training/pointcept/models/utonia/utonia_v1m3b_qwen3_5_distill_ema.py`
(`Utonia-v1m3b_qwen3_5_distill_ema`). The earlier align-only variant
(`utonia_v1m3a_qwen3_5_align_only.py`) backed sweep version A.

---

## 1. Motivation — why Qwen3.5 ViT instead of DINOv2

Utonia is a cross-domain PTv3 encoder (Sonata/Concerto lineage), originally
distilled against a **DINOv2** 2D teacher. The downstream target here is
**spatial reasoning on VSI-bench / re-VSI-bench**, where the VLM backbone is
**Qwen3.5-VL** (and a VG-LLM variant). Hypothesis:

> If Utonia's 3D features are aligned to the **same vision manifold the VLM
> already consumes (Qwen3.5 ViT)** rather than DINOv2's, the VLM should
> integrate the 3D features more naturally → better VSI accuracy.

So the 2D distillation teacher was swapped **DINOv2 → Qwen3.5-4B vision tower**.

---

## 2. The A→I ablation sweep

Each version changed one thing relative to the previous (full rationale lives in
each config's docstring; the failed iterations A–G were removed in this cleanup,
their findings are preserved here):

| ver | change | takeaway |
|---|---|---|
| **A** | pure Qwen↔Utonia alignment, **no SSL** (`v1m3a`) | baseline align-only |
| **B** | alignment + **EMA-fixed 3D SSL** regularization (`v1m3b`) | joint SSL+align tried |
| **C** | same as B, alternate enc2d loss formulation | — |
| **D** | **batch-centered cosine** pull | — |
| **E** | **cross-scene (batch-wide) InfoNCE** | contrastive beats plain cosine |
| **F** | E + **K-subsampling** + **MLP patch_proj** | memory + stability |
| **G** | **two-tower (bidirectional) projection**, SSL off | A–F showed joint SSL+align hurt alignment metrics ~2× → SSL turned off |
| **H** | **align at Qwen's MERGED 16×16 / LLM-dim scale** (`use_full_merger=True`) | see §3 |
| **I** | **H + light SSL** re-enabled as a regularizer | evaluated checkpoint |

### Why H moved to the merged scale (key insight)

The Qwen3.5 ViT **per-patch (32×32)** features were diagnosed as effectively
**rank ≈ 12** even after `merger.norm` — they are an intermediate representation
the tower was trained to push through its 2×2 `merger`
(norm → spatial-merge → linear_fc1 → act → linear_fc2 → LLM hidden dim). At the
**merged 16×16 scale the rank jumps to 100+**, and that is the representation the
LLM actually consumes. So H aligns there:

- `use_full_merger=True` → `ENC2D_forward` runs the full merger, returns
  16×16 tokens at **Qwen LLM hidden dim = 2560**.
- `enc2d_head_in_channels=2560`, `enc2d_layer_idx=-1`.
- It reuses the existing 32×32 point↔image correspondences and just halves
  row/col in `feature_index`, so each 16×16 target token aggregates the four
  32×32 source patches in its 2×2 footprint — matching Qwen's own merge.

### Why I re-enabled SSL

H is alignment-only, so the backbone is shaped *solely* by "match Qwen's vision
token," with nothing preserving a generally-useful 3D representation. It got away
with it for 5 short epochs only because it warm-starts from `utonia.pth`
(SSL-pretrained). I adds light SSL as a **regularizer against collapse** to a
"Qwen-friendly but feature-poor" backbone, while keeping alignment dominant:

- `enc2d_loss_weight = 3/4`, `mask = 1/16`, `roll_mask = 1/16`, `unmask = 1/8`.
- mask/unmask heads are instantiated (start random; warm up their prototypes).

---

## 3. Training setup (H / I)

- **Module:** `Utonia-v1m3b_qwen3_5_distill_ema` (EMA teacher).
- **2D target:** Qwen3.5-4B vision tower, **merged 16×16 @ 2560-d**
  (`use_full_merger=True`, `enc2d_layer_idx=-1`).
- **Alignment loss:** cross-scene InfoNCE (`enc2d_loss_type="infonce_batch"`,
  `infonce_temperature=0.07`), **two-tower** projection into a `common_dim=512`
  space, MLP `patch_proj`, K-subsampling.
- **3D student/teacher:** PTv3 Utonia base `(54,108,216,432,576)`,
  `enc2d_upcast_level=3` → `backbone_out_channels=1332`.
- **Warm-start:** `utonia.pth` (backbone only); align/SSL heads random.
- **LR:** layer-grouped — backbone `base_lr*0.05` w/ decay `0.9`, new modules
  full `base_lr`; OneCycleLR.
- **Schedule:** 5 epochs, `batch_size=64` (8×H100), `MultiViewGenerator`
  `max_size=enc2d_max_size=16384`.
- **Datasets:** indoor multi-dataset — **ScanNet, ScanNet++, ArkitScenes,
  Structured3D** (`SkipOnErrorImagePointDataset`, `${DATASET_ROOT}/data/...`).
  (Originally ScanNet-only; broadened — append s3dis / hm3d / re10k the same way.)
- **Image norm:** Qwen preprocessor stats `mean=std=(0.5,0.5,0.5)`.

---

## 4. Evaluation

- **Alignment / representation eval:** `training/tools/eval_alignment_full.py`
  (+ `eval_alignment.py`, `debug_alignment.py`) — CKA / linear-probe of the
  distilled encoder vs the Qwen target; `judge_qwen_utility.py` probes whether
  Qwen ViT is a usable target vs a DINOv2 reference
  (`eval-utonia-v1m1-dinov2-scannet.py`).
- **Downstream spatial reasoning:** VSI-bench and re-VSI-bench, in two stacks:
  1. `Qwen3.5 + CFG + Utonia encoder`
  2. `VG-LLM + CFG + Utonia encoder`
- **Baseline:** the original (DINOv2-aligned) Utonia encoder in the same stacks.

---

## 5. Results

The Qwen3.5-aligned encoder (**I**) **regressed** vs. the original Utonia encoder:

| downstream | vs. original Utonia |
|---|---|
| `Qwen3.5 + CFG + Utonia encoder` | regressed |
| `VG-LLM + CFG + Utonia encoder` | regressed **by a large margin** |

(Fill in exact VSI / re-VSI numbers.) The hypothesis — "match the VLM's own
vision manifold and the 3D branch will help" — did not hold even after fixing the
two most obvious pitfalls (merged-scale target, SSL regularizer).

---

## 6. Analysis — likely causes

Note: the sweep already addressed the two pitfalls one would first suspect — it
aligns at the **merged** scale (not the rank-collapsed pre-merge one), and I
keeps **SSL** as an anti-collapse regularizer. So the remaining explanations are:

1. **Aligning a 3D geometric encoder to a 2D image-text manifold removes the
   complementary signal.** Qwen3.5 ViT encodes 2D/semantic, image-text-aligned
   features. The VLM **already has that** via its own vision path. The 3D
   encoder's value is *geometry* the 2D path lacks; pulling it toward the Qwen
   manifold makes the 3D branch more redundant with the 2D branch and less
   geometrically discriminative → worse on spatial questions. This is the
   leading hypothesis and is directly checkable (see §7).

2. **5-epoch, indoor-only fine-tuning narrows Utonia's cross-domain generality.**
   VSI/re-VSI span diverse egocentric scenes; aligning on a few indoor datasets
   for few epochs trades away the breadth the original Utonia had.

3. **VG-LLM's larger drop = geometry reliance × CFG amplification.** VG-LLM leans
   more on the 3D encoder for grounding than Qwen3.5 (which has a strong 2D
   fallback). Degraded 3D features hurt it more, and CFG amplifies the weakened
   conditioning.

4. **Possible train/inference feature mismatch (encoder side).** Training shapes
   the `enc2d_upcast_level=3` (1332-d) representation; confirm the downstream
   precompute (`demo/11_precompute_external.py`) extracts the *same* composition
   and not the open-ended 1386-d concat.

5. **SSL-vs-alignment tension (H vs I).** A–F found joint SSL+align hurt
   alignment metrics ~2×. I's diagnostic is exactly: train H and I side-by-side
   on identical data/steps; if `I.mIoU ≈ H.mIoU` but `I.CKA` lower, SSL fights
   alignment and H was right; if `I.mIoU > H.mIoU` at similar CKA, SSL retention
   helps downstream.

---

## 7. Recommendations / next steps

1. **Test the leading hypothesis directly:** compare downstream VSI with the 3D
   branch features (a) original Utonia, (b) H, (c) I — and measure CKA between
   each encoder and *both* the Qwen ViT target and a geometry probe (e.g. depth /
   normal / 3D-semseg linear probe). If alignment ↑ CKA-to-Qwen but ↓ geometry
   probe and ↓ VSI, the 2D target is the wrong objective for this encoder.
2. **Evaluate H and I side-by-side** per I's built-in diagnostic (§6.5).
3. **Verify downstream feature composition** matches training (1332-d, 3-level
   upcast) end-to-end.
4. **Ablate alignment strength** (`enc2d_loss_weight`) and longer schedules to
   separate "alignment hurts" from "under-training / domain-narrowing hurts."
5. **Reconsider the target.** If the goal is helping a VLM that already has a
   strong 2D path, a *geometry-preserving* objective (or aligning only a small
   adapter while freezing the Utonia backbone) may beat pulling the whole
   encoder onto the 2D manifold.

---

## Cleanup note

This branch removed the failed/superseded distill configs (`v1m3-0-base`,
`v1m3-1-scannet-only`, and sweep versions `A`–`G`), keeping only the two final
recipes **H** and **I** (renamed from `-scannet-only-` to `-indoor-` to reflect
the actual multi-dataset training) plus the eval tooling. The model modules
`utonia_v1m2_qwen3_5_distill.py` (used only by the removed `0/1` configs) and
`utonia_v1m3a_qwen3_5_align_only.py` (used only by removed `A`) are now orphaned
and can be removed in a follow-up if desired.
