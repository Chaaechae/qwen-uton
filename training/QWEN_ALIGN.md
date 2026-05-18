# Qwen3.5 ViT × Utonia(PTv3) Alignment — 종합 정리

이 문서는 `claude/analyze-qwen-uton-repo-sYaPf` 브랜치에서 진행한
모든 실험·진단·코드 변경의 종합 정리입니다. 최종 목표는 **Utonia
PTv3 백본의 3D point 표현을 Qwen3.5 ViT의 2D patch 표현과 정렬해서
Video-3D-LLM의 3D positional encoding 대체로 사용**할 수 있게 만드는 것.

---

## 1. 학습 setup

문서 전반에서 사용하는 약어:

| 약어 | 의미 | 차원 |
|------|------|------|
| **f2** / `f2_qwen` | feature **2D** = Qwen ViT patch token | 1024-d |
| `f2_proj` | (G에서만) `qwen_proj` 후 공통공간 표현 | 512-d |
| **f3** / `f3_raw` | feature **3D** = PTv3 backbone 출력 (patch_proj 전) | 1332-d |
| `f3_proj` | `patch_proj` 후 표현 | 1024-d (F까지) / 512-d (G) |
| `_bc` 접미사 | batch-centered (batch 평균 빼고 측정) |  |
| `pos_bc` / `neg_bc` | 올바른/잘못된 pair의 batch-centered cosine | scalar |
| `discrim_gap_bc` | `pos_bc − neg_bc` (정렬 품질 핵심 지표) | scalar |
| `K` | batch당 unique한 (point, patch) pair 개수 | int |

```
┌──────────────────────────────────────────────────────────────┐
│  2D teacher (frozen)                                         │
│    Qwen3.5-4B `model.visual`                                 │
│    img(512×512, mean=std=0.5) → 1024-d patch token × 32×32   │
│    *** merger.norm 적용 후 사용 (이전엔 안 했음, §5 참고) *** │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  qwen_proj  : Linear(1024 → 512) + LN   (G에서 추가, trainable) │
└────────────────────────────▲─────────────────────────────────┘
                             │ common 512-d space
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  patch_proj : MLP(1332 → 2048 → 512) + LN    (G; F는 1024-d) │
└────────────────────────────▲─────────────────────────────────┘
                             │ projected 3D feature
                             │
┌──────────────────────────────────────────────────────────────┐
│  student PTv3 backbone (Utonia base, warm-start)             │
│    enc_channels=(54,108,216,432,576) → 1332-d (upcast)       │
└──────────────────────────────────────────────────────────────┘

학습:
  loss = enc2d_loss (cosine / cosine_bc / infonce / infonce_batch)
       + mask/unmask SSL (선택)
  - point ↔ image correspondence는 dataset 단에서 매번 다시 계산
  - InfoNCE 종류는 (3D point, 2D patch) pair를 positive, 같은 batch
    의 다른 patch 들을 negatives 로 사용
```

---

## 2. 코드 구조 — 무엇이 어디에 있나

### 모델 (3종류)

| 파일 | 모델 type | 용도 |
|------|-----------|------|
| `pointcept/models/utonia/utonia_v1m2_qwen3_5_distill.py` | `Utonia-v1m2_qwen3_5_distill` | **원본 (수정 없음)**. SSL 버그 2개 있음 (teacher init 미실행 + EMA pass), v1m3-0/1 config가 사용 |
| `pointcept/models/utonia/utonia_v1m3a_qwen3_5_align_only.py` | `Utonia-v1m3a_qwen3_5_align_only` | Align-only 변종. SSL 완전 제거 → 정렬만 학습. v1m3-A가 사용 |
| `pointcept/models/utonia/utonia_v1m3b_qwen3_5_distill_ema.py` | `Utonia-v1m3b_qwen3_5_distill_ema` | SSL 버그 두 개 모두 수정 + EMA 정상화. **현재 메인 모델**, v1m3-B/C/D/E/F/G 가 모두 사용. 그동안 모든 loss 변형·architecture 옵션이 여기 누적됨 |

### Config (9개)

| Config | 모델 type | 핵심 차이 | 결과 |
|--------|-----------|----------|------|
| `v1m3-0-base` | v1m2 | 원본. SSL 버그 그대로 | (사용 안 함) |
| `v1m3-1-scannet-only` | v1m2 | 0의 ScanNet-only 변종. 동일한 SSL 버그 보유 | (사용 안 함) |
| `v1m3-A` | v1m3a | SSL 완전 off, alignment만 | pos_bc 0.01 |
| `v1m3-B` | v1m3b | SSL fix + EMA 정상화 + cosine pull (`(1-cos)·10`) | mean-direction 붕괴, pos_bc 0.97/neg 0.97 |
| `v1m3-C` | v1m3b | 같은 모델, **per-scene InfoNCE** (τ=0.03) | pos_bc 0.006, 천장 |
| `v1m3-D` | v1m3b | **cosine_bc** (batch-centered cosine pull) | pos_bc 0.006, 천장 |
| `v1m3-E` | v1m3b | **cross-scene InfoNCE** (τ=0.5, K~22k) | loss ≈ log K, 학습 안됨 |
| `v1m3-F` | v1m3b | E + **K-subsample 1024** + **MLP patch_proj** (2048-d hidden) | pos_bc 0.01, 천장 |
| `v1m3-G` | v1m3b | F + **two-tower (qwen_proj)** + **SSL 완전 off** + **enc2d_layer_idx=-2** + **merger.norm 적용** | 학습 예정 |

### Tools

| 파일 | 용도 |
|------|------|
| `tools/eval_alignment.py` | 기본 pos/neg cosine histogram |
| `tools/eval_alignment_full.py` | 종합 평가: R@K, MRR, CKA, batch-centered cosine. validation scene 50개 사용 |
| `tools/debug_alignment.py` | **단일 batch 깊은 진단**: K survival, feature stats, PCA, gradient flow, single-batch overfit, correspondence 시각화, Qwen ViT layer-by-layer rank probe |

### 데이터 / 인프라

| 파일 | 용도 |
|------|------|
| `pointcept/datasets/skip_on_error_dataset.py` | corrupted sample skip + JSONL 로깅 |
| `pointcept/engines/launch.py` | `DIST_BACKEND=gloo` 지원 (cluster의 NCCL 깨짐 대응) |
| `run_train.sh` | 1-shot launcher. flag 어디든 위치 가능. variant 자동 해석. 데이터 symlink |
| `install_into_pointcept.sh` | third_party/Pointcept으로 우리 파일 symlink |
| `demo/compare_pca.py` | 3개 모델 PCA 시각화 (matplotlib / plotly / open3d) |

---

## 3. 실험의 흐름 — 그동안 무엇을 시도했나

### Phase 1: SSL 버그 수정 (v1m3-A, B)
- 원본 v1m2의 두 버그:
  1. `teacher.{mask,unmask}_head` 가 student head 가중치로 초기화 안 됨 → SK target이 random prototype
  2. `after_step`이 빈 `pass` → teacher 가 EMA 업데이트 안 됨
- A: 두 버그 모두 회피하기 위해 SSL 자체를 제거
- B: 두 버그 정상 수정 + cosine pull 로 alignment 진행
- **결과**: B는 "mean-direction collapse" — 모든 f3가 Qwen 평균 방향으로 끌려가서 pos=neg=0.97. R@1=0.004. 학습은 됐지만 의미 없는 상태

### Phase 2: Loss 형식 탐색 (v1m3-C, D, E, F)
- cosine pull의 trivial solution (μ_qwen으로 다 끌고 가기)을 막기 위해:
  - **C**: per-scene InfoNCE (K~150-2k) — 정체 (pos_bc 0.006)
  - **D**: cosine_bc — batch-center 후 cosine pull. C와 비슷한 정체
  - **E**: cross-scene InfoNCE (K~22k, τ=0.5) — log(K) 천장에서 못 내려옴
  - **F**: K-subsample 1024 + patch_proj MLP — 여전히 pos_bc 0.01
- **결과**: 5개 변종 모두 pos_bc ≤ 0.01 동일 천장. loss formulation 변경으로는 못 풀림

### Phase 3: 디버깅 도구 (debug_alignment.py)
"loss 문제가 아닐 가능성"이 짙어져서 단일 batch 깊은 진단 도구 작성:
1. K survival → 91% 정상
2. Feature stats → `f2_qwen pairwise_cos = 0.95` (강한 비등방성)
3. Gradient flow → `|g|/|w| ≈ 1e-4` (소실은 아님)
4. **단일 batch overfit** → 100 step만에 pos_bc 0.018→0.066. **학습은 가능**
5. Correspondence 시각화 → 정상

→ 핵심 발견: **batch 내에서는 학습되는데 scene 사이에서 일반화가 안 됨**. patch_proj가 batch마다 다른 방향으로 끌려감.

### Phase 4: PCA 진단 — 진짜 원인 발견
PCA effective rank 분석:
```
f2_qwen (Qwen 마지막 block 출력) : eff_rank = 1.0  ← !!
f3_raw  (PTv3 출력)              : eff_rank = 6.1
f3_proj (patch_proj 출력)        : eff_rank = 5.2
```

**Qwen feature가 사실상 rank 1**. 모든 patch가 `μ + λᵢ·v` 형태로 한 방향에 줄지어 있음. 이건 어떤 loss로도 정렬 학습 불가능 — 정렬할 정보 자체가 없는 것.

### Phase 5: ViT layer probe — 누락된 단계 발견
모든 ViT block에서 rank < 5. patch_embed부터 이미 붕괴. 가설: **post-block norm을 빼먹음**.

Qwen3.5 visual 구조 확인:
- `pos_embed`, `rotary_pos_emb`, `merger: Qwen3_5VisionPatchMerger`
- `merger` 자식: `norm`, `linear_fc1`, `act_fn`, `linear_fc2`

`merger.norm` 적용 결과: **rank 1 → 12.58**. 12배 좋아짐. fc1/fc2는 spatial 2x2 pooling이 내장돼 있어서 per-patch에 적용 불가 (4096-d 입력 요구).

→ **`ENC2D_forward`가 `merger.norm`을 호출하지 않은 게 5개 변종 실패의 진짜 원인**. 1줄 fix.

---

## 4. 핵심 발견 한 줄 정리

> **원본 코드는 Qwen ViT의 `merger.norm` 후처리를 빠뜨려서, alignment loss를 사실상 rank-1 feature 에 대해 학습시키고 있었다. 이것이 그동안 모든 loss 변형(B~F)이 같은 천장에 부딪힌 진짜 이유.**

---

## 5. 코드 변경 누적 목록 (model file 기준)

`utonia_v1m3b_qwen3_5_distill_ema.py`에 시간순으로 누적된 변경:

| 추가된 옵션 | 도입 시점 | 효과 |
|-------------|---------|------|
| 기본 (v1m3-B 초기) | — | SSL 버그 2개 fix, cosine pull |
| `enc2d_loss_type` (`cosine`/`infonce`) | C | per-scene InfoNCE 추가 |
| `cosine_bc` 분기 | D | batch-centered cosine |
| `infonce_batch` 분기 | E | cross-scene InfoNCE |
| `infonce_batch_subsample` | F | K-subsample로 dilution 해소 |
| `patch_proj_hidden_channels` | F | patch_proj를 MLP로 |
| `common_dim` + `qwen_proj` | G | two-tower (양방향 projection) |
| `enc2d_layer_idx` | G | 중간 ViT block 선택 가능 |
| **`merger.norm` 적용 in `ENC2D_forward`** | **G (가장 중요)** | rank 1 → 12, 모든 config에 자동 반영 |

---

## 6. v1m3-0/1 (원본) vs 현재 작업 비교

| 항목 | v1m3-0 / v1m3-1 (원본) | v1m3-G (현재) |
|------|----------------------|---------------|
| 모델 type | `Utonia-v1m2_qwen3_5_distill` (버그 보유) | `Utonia-v1m3b_qwen3_5_distill_ema` |
| SSL 버그 | 있음 (teacher 미초기화 + EMA pass) | 수정됨 |
| Qwen `merger.norm` | **누락** (rank 1 짜리 feature로 학습) | **적용** (rank 12) |
| Alignment loss | cosine pull `(1-cos)·10` | symmetric InfoNCE_batch + K-sub 1024 |
| Loss weights | mask=1/8, unmask=2/8, enc2d=4/8 (50% noise) | enc2d=1.0, SSL=0 |
| patch_proj | Linear(1332→1024) | MLP(1332→2048→512) + LN |
| Qwen-side proj | 없음 | `qwen_proj`: Linear(1024→512) + LN, trainable |
| Common space | Qwen 고정 1024-d | 학습 가능한 512-d |
| ViT layer | 마지막 block (rank 1, collapse) | block -2 + merger.norm (rank 12) |
| 사용 가능한 진단 도구 | 없음 | eval_alignment_full + debug_alignment |
| Multi-GPU | NCCL (cluster에서 깨짐) | gloo 옵션 추가 |
| Error handling | crash on bad sample | SkipOnError + JSONL 로깅 |

---

## 7. Alignment 성공을 위한 핵심 체크리스트

학습 목표 — **Qwen ViT patch feature와 Utonia/PTv3 3D feature를 의미 있게
정렬**해서 다음 metric들이 의미 있는 값을 가지도록:

| Metric | 목표값 | 의미 |
|--------|--------|------|
| `pos_bc` | ≥ 0.2 | 정확한 pair의 batch-centered cosine |
| `neg_bc` | ≤ 0.05 | 잘못된 pair의 cosine (낮을수록 discriminative) |
| `discrim_gap_bc` (pos - neg) | ≥ 0.15 | 둘의 차이 |
| `R@1` | ≥ 0.10 | scene 내에서 정확한 patch를 1순위로 retrieve |
| `MRR` | ≥ 0.15 | mean reciprocal rank |
| `CKA(proj, qwen)` | ≥ 0.05 | Centered Kernel Alignment |

### 핵심적으로 필요한 것들

1. **건강한 feature 추출** ✅ (G에 적용)
   - Qwen ViT는 단순히 block 출력 쓰면 안 됨. **반드시 `merger.norm` 적용**.
   - 더 좋은 옵션: 16×16 grid로 correspondence 재구성해서 `merger` 전체 적용 가능하게 (rank 100+ 예상). G가 실패하면 이쪽.

2. **양방향 학습 가능 projection (two-tower)** ✅ (G에 적용)
   - 단일 head로 Qwen 고정 공간을 맞히는 건 너무 narrow. CLIP/SimCLR 처럼 양쪽 다 학습 가능한 공통 공간으로.

3. **SSL 간섭 제거** ✅ (G에서 SSL=0)
   - A vs F 비교에서 SSL이 alignment 학습을 살짝 방해함을 확인. 정렬 학습이 안정될 때까지는 끄는 게 낫다. 나중에 PTv3 다시 일반화 학습이 필요하면 SSL을 약하게 켜는 fine-tune 단계 추가 가능.

4. **적절한 contrastive 형식** ✅ (G: infonce_batch + sub=1024)
   - cosine pull 단독은 mean-direction trivial solution 으로 붕괴
   - InfoNCE는 negatives를 명시적으로 밀어내서 안전. K가 너무 크면 log(K) 천장에 막히니까 1024 정도가 적절.

5. **검증 도구로 학습 중 모니터링**
   - `debug_alignment.py --probe-all-layers` — 모델 자체의 representation 건강성 확인
   - `eval_alignment_full.py` — held-out scene에서 일반화 측정
   - 학습 중간(매 N epoch)에 eval 돌려서 pos_bc/CKA 추이 보기

### 만약 G 도 막히면

순서대로 시도할 백업 plan:

1. **16×16 patch grid + 전체 merger 사용** — 진짜 LLM-aligned representation
   - 데이터 transform의 correspondence를 P=32 로 재계산
   - `ENC2D_forward`가 `merger(hidden, grid_thw)` 전체 호출
   - `patch_h = patch_w = 16` (256 tokens per image)
   - rank 100+ 예상되므로 alignment 학습 훨씬 쉬워짐

2. **DINOv2 sanity check** — framework 자체가 동작하는지 한 번 더 확인
   - `image_weight_name="dinov2_vitb14"` 로 바꿔서 같은 G config로 학습
   - DINOv2는 visual SSL이라 patch level에서 의미 있는 신호 보장됨
   - 만약 DINOv2로는 잘 되는데 Qwen으로는 안 되면 → Qwen-specific 이슈

3. **Patch_proj capacity 더 키우기** — Linear(1332→4096→2048→512) 3-layer MLP

---

## 8. 다음 단계

### 즉시
```bash
# 1. G의 probe 한 번 더 — merger.norm 적용된 ENC2D_forward 의 layer-by-layer rank 확인
python tools/debug_alignment.py \
  --config-file configs/utonia/distill-utonia-v1m3-G-scannet-only-qwen3_5-4b.py \
  --weight     exp/utonia_q35_f/model/model_last.pth \
  --out-dir    exp/utonia_q35_g/debug_probe \
  --probe-all-layers

# 2. G 학습 시작
bash training/run_train.sh G --num-gpus=8

# 3. 학습 중 / 끝나면 eval
python tools/eval_alignment_full.py \
  --config-file configs/utonia/distill-utonia-v1m3-G-scannet-only-qwen3_5-4b.py \
  --weight     exp/utonia_q35_g/model/model_last.pth \
  --baseline-weight /group-volume/Utonia/utonia.pth \
  --num-scenes 50 \
  --out-dir    exp/utonia_q35_g/alignment_eval_full
```

### 의사결정 기준
- `pos_bc ≥ 0.15` 이고 `R@1 ≥ 0.05` 면 → **성공**, downstream Video-3D-LLM 으로 진행
- 그 미만이면 → §7의 백업 plan 1 (16×16 grid + 전체 merger) 진행

---

## 9. 청산하면 좋은 잔재

선택사항이지만 정리하면 깔끔:

- `v1m3-0`, `v1m3-1` config: **삭제 권장**. SSL 버그 있는 원본이라 사용 안 함. README 한 줄로 "원본 보존용으로만 남겨두고 학습은 G 사용" 설명 추가하는 정도면 충분
- `v1m3-B/C/D/E`: 실패한 변종이지만 학습 시 무엇이 안 되는지 보여주는 reference로 유용. 삭제하지 말 것 (지금까지 어떻게 도달했는지 기록)
- `v1m3-F`: G의 직접 선조. 비교용으로 유지

코드 자체에서:
- 모델 파일에 누적된 옵션들은 모두 backward compatible. 기존 config는 그대로 동작
- 새 변종 만들 때는 F나 G를 복사해서 시작 권장
