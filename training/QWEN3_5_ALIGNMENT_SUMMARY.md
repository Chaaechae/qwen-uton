# Utonia ↔ Qwen3.5-VL 정렬(distillation) 정리

> Utonia(PTv3) 3D 포인트 인코더를 **Qwen3.5-VL 비전 타워에 정렬(align)** 시켜,
> 정렬된 3D feature를 Qwen3.5 / VG-LLM 기반 공간추론(VSI / re-VSI)에 넣었을 때
> 성능을 올려보려 한 실험의 기록입니다. 처음 보는 사람도 따라올 수 있도록
> **동기 → 무엇을 어떻게 학습했는지(loss·실행법 포함) → 평가 → 결과 → 시도한
> 것들과 알게 된 사실 → 분석**의 순서로 정리했습니다.

학습 레시피 2개는 `training/configs/utonia/` 에 있습니다.

| 파일 | 한 줄 설명 |
|---|---|
| `distill-utonia-v1m3-indoor-noSSL-qwen3_5-4b.py` | **정렬만** (SSL 끔) |
| `distill-utonia-v1m3-indoor-SSL-qwen3_5-4b.py` | 정렬 **+ 가벼운 SSL** (정규화). **평가에 쓴 체크포인트** |

모델 모듈: `training/pointcept/models/utonia/utonia_v1m3b_qwen3_5_distill_ema.py`
(`Utonia-v1m3b_qwen3_5_distill_ema`).

---

## 1. 배경 / 동기 — 왜 DINOv2가 아니라 Qwen3.5 ViT인가

Utonia는 여러 도메인에 걸쳐 사전학습된 PTv3 인코더(Sonata/Concerto 계열)이고,
원래는 **DINOv2** 2D teacher에 distill되어 있었습니다.

이번 다운스트림 목표는 **VSI-bench / re-VSI-bench 공간추론**이고, 거기서 쓰는
VLM 백본이 **Qwen3.5-VL**(및 VG-LLM 변형)입니다. 가설은:

> Utonia의 3D feature를 **VLM이 이미 소비하는 비전 manifold(Qwen3.5 ViT)** 에
> 맞추면, VLM이 3D feature를 더 자연스럽게 통합해 VSI 정확도가 오를 것이다.

그래서 2D distill teacher를 **DINOv2 → Qwen3.5-4B 비전 타워**로 교체했습니다.

---

## 2. 학습 레시피 — 어떻게 학습하고, loss는 무엇이며, 어떻게 실행하나

### 2.1 두 레시피의 공통 구조

- **모델:** PTv3 student/teacher (Utonia base 채널 `(54,108,216,432,576)`),
  teacher는 student의 EMA. `enc2d_upcast_level=3` → 정렬용 3D feature는 stage
  1~4 concat = **1332-d**(`backbone_out_channels=1332`).
- **2D 타깃:** Qwen3.5-4B 비전 타워의 **merged(16×16, LLM hidden=2560)** feature
  (§3 참고). `use_full_merger=True`, `enc2d_head_in_channels=2560`,
  `enc2d_layer_idx=-1`.
- **정렬 loss:** 장면 간(batch-wide) **InfoNCE** (`enc2d_loss_type="infonce_batch"`,
  `infonce_temperature=0.07`) — **two-tower** 투영으로 3D/2D를 공통
  `common_dim=512` 공간에 보낸 뒤(3D쪽 MLP `patch_proj`, 2D쪽 `qwen_proj`),
  배치 내 고유 패치를 K-subsample 해서 대조학습.
- **warm-start:** `utonia.pth` (backbone만 로드; 정렬/SSL head는 random).
- **LR:** layer-grouped — backbone `base_lr*0.05`(층별 0.9 decay), 새 모듈은 full
  `base_lr`; OneCycleLR.
- **스케줄/자원:** 5 epoch, `batch_size=64` (8×H100),
  `MultiViewGenerator(max_size=enc2d_max_size=16384)`.
- **데이터셋(indoor 멀티):** ScanNet, ScanNet++, ArkitScenes, Structured3D,
  S3DIS, HM3D (`SkipOnErrorImagePointDataset`, `${DATASET_ROOT}/data/<name>`).
- **이미지 정규화:** Qwen 전처리기 통계 `mean=std=(0.5,0.5,0.5)` (ImageNet 아님).

### 2.2 레시피 A — 정렬만 (no SSL) · `…-indoor-noSSL-…`

backbone을 **오직 "Qwen 비전 토큰과 일치"라는 정렬 목표로만** 형성합니다.
SSL(mask / roll-mask / unmask)은 모두 끔.

- loss 비중: `enc2d=1.0`, SSL=0.

### 2.3 레시피 B — 정렬 + 가벼운 SSL · `…-indoor-SSL-…` (평가에 사용)

레시피 A에 SSL을 **약하게 다시 켠** 버전. 정렬이 지배적이되, SSL이 backbone이
"Qwen에 맞추기만 하고 feature가 빈약해지는 붕괴"를 막는 정규화로 작동합니다.

- loss 비중: `enc2d=3/4`, `unmask=1/8`, `mask=1/16`, `roll_mask=1/16`.
- mask/unmask head는 여기서 생성되며(weight=0이면 모델 빌드가 생략) random에서
  시작 → 클러스터 prototype 수렴까지 warmup 필요.

### 2.4 InfoNCE loss (요지)

3D 포인트 feature를 대응되는 Qwen merged 토큰과 묶어, 같은 (3D,2D) 쌍은
당기고 배치 내 다른 패치들은 밀어냅니다. 공통 512-d 공간에서 정규화 후

```
L = - (1/N) Σ_i  log [ exp(sim(z3d_i, z2d_i)/τ) / Σ_j exp(sim(z3d_i, z2d_j)/τ) ]
```

`τ=0.07`, `sim`=코사인. (초기 버전에서 쓰던 단순 코사인/배치중심 코사인보다
이 cross-scene InfoNCE가 정렬에 유리했음 — §6.)

### 2.5 실행 방법

```bash
# 1) 외부 리소스 경로
export QWEN3_5_4B_PATH=/path/to/Qwen3.5-4B
export UTONIA_PRETRAINED_CKPT=/path/to/utonia.pth   # warm-start (강력 권장)
export DATASET_ROOT=/path/to/3Ddataset              # ${DATASET_ROOT}/data/<name> 로 접근

# 2) Utonia 측 모듈/레시피를 Pointcept 트리에 심볼릭 링크 (repo 루트에서, 재실행 무해)
bash training/install_into_pointcept.sh

# 3) 학습 (둘 중 하나 선택)
cd third_party/Pointcept
python tools/train.py \
    --config-file configs/utonia/distill-utonia-v1m3-indoor-noSSL-qwen3_5-4b.py \
    --num-gpus 8 --options save_path=exp/utonia_q35_indoor_nossl
# 또는
python tools/train.py \
    --config-file configs/utonia/distill-utonia-v1m3-indoor-SSL-qwen3_5-4b.py \
    --num-gpus 8 --options save_path=exp/utonia_q35_indoor_ssl
```

학습 직후 ~100 step에서 확인할 것: `loss` 유한·감소, `enc2d_loss`(정렬) 0이 아니고
감소, (SSL 켠 경우) `mask/unmask_loss` NaN 없음.

> 사전 점검: `python tools/test_qwen3_5_vit_path.py --model "$QWEN3_5_4B_PATH"` 로
> Qwen ViT forward 경로가 통과(`[4/4 PASS]`)하는지 먼저 확인.

---

## 3. 정렬 타깃을 왜 "merged 16×16"으로 잡았나 (핵심)

Qwen3.5 ViT의 **per-patch(32×32)** feature는 `merger.norm` 이후에도 사실상
**rank ≈ 12**로 붕괴해 있습니다. Qwen ViT는 image-text 정렬되어 학습됐고, 변별
정보를 자신의 2×2 `merger`(norm → spatial-merge → linear_fc1 → act → linear_fc2
→ LLM hidden)를 통과시켜 내보내도록 훈련됐기 때문입니다. **merged 16×16
스케일에서는 rank가 100+로 뛰며, 이게 LLM이 실제로 소비하는 표현**입니다. 그래서
정렬 타깃을 거기로 옮겼습니다.

- 32×32 point↔image 대응을 다시 계산하지 않으려고, 기존 32×32 대응값을 그대로
  쓰되 loss 경로의 `feature_index`에서 row/col을 2로 나눕니다. 그러면 16×16 타깃
  토큰 하나가 자기 2×2 영역에 들어오는 (최대) 4개의 32×32 패치를 모으게 되는데,
  이는 Qwen이 내부적으로 하는 merge와 동일한 의미라 정합합니다.

---

## 4. 평가 방법

- **정렬/표현 평가:** `training/tools/eval_alignment_full.py`
  (+ `eval_alignment.py`, `debug_alignment.py`) — distill된 인코더와 Qwen 타깃
  간 CKA / linear-probe. `judge_qwen_utility.py`는 Qwen ViT가 distill 타깃으로
  쓸 만한지를 DINOv2 기준(`eval-utonia-v1m1-dinov2-scannet.py`)과 비교.
- **다운스트림 공간추론:** VSI-bench, re-VSI-bench, 두 스택에서
  1. `Qwen3.5 + CFG + Utonia 인코더`
  2. `VG-LLM + CFG + Utonia 인코더`
- **기준선:** 원본(DINOv2 정렬) Utonia 인코더를 같은 스택에 넣은 경우.

---

## 5. 결과

Qwen3.5 정렬 인코더(SSL 레시피, 평가본)는 원본 Utonia 대비 **오히려 하락**:

| 다운스트림 | 원본 Utonia 대비 |
|---|---|
| `Qwen3.5 + CFG + Utonia 인코더` | 하락 |
| `VG-LLM + CFG + Utonia 인코더` | **큰 폭으로 하락** |

(정확한 VSI / re-VSI 수치는 여기에 채워 넣을 것.) "VLM 자신의 비전 manifold에
맞추면 3D 분기가 도움 될 것"이라는 가설은, 가장 의심되던 두 함정(타깃 스케일,
SSL 붕괴)을 고친 뒤에도 성립하지 않았습니다.

---

## 6. 시도한 것들 & 알게 된 사실 (lessons)

정렬 방식을 단계적으로 바꿔가며 ablation을 돌렸습니다(개별 버전 이름은 생략).
남은 교훈:

- **타깃 스케일이 결정적.** pre-merge(32×32)는 rank≈12로 붕괴 → distill 타깃으로
  부적합. merged(16×16, 2560-d)로 옮기니 rank 100+이고 LLM이 실제 쓰는 표현이라
  정렬 지표가 크게 개선됨. → **§3의 merged 타깃이 최종 선택.**
- **단순 코사인 < cross-scene InfoNCE.** 패치별 코사인/배치중심 코사인보다,
  배치 전체에 걸친 InfoNCE 대조가 정렬에 유리.
- **two-tower(양방향) 투영 + 512-d 공통공간 + K-subsample + MLP patch_proj** 가
  메모리/안정성에 필요했음(특히 8×H100 / batch=64에 맞추려고 point cap을 16384로).
- **SSL과 정렬을 동시에 강하게 주면 정렬 지표가 ~2배 나빠짐.** → 정렬만 하는 게
  정렬엔 유리. 다만 정렬-only는 backbone의 일반적 3D 표현을 보존할 압력이 없어,
  짧은 5 epoch에선 warm-start(utonia.pth) 덕에 버티지만 길게/scratch면 geometry가
  무너질 위험 → **약한 SSL을 다시 켠 정규화 버전**을 별도로 둠.
- **상관 트릭:** 32×32 대응을 재계산하지 않고 row/col만 2로 나눠 16×16에 매핑해도
  Qwen 내부 merge와 의미가 같음.
- **데이터 범위:** ScanNet 단독으로 시작 → indoor 멀티(ScanNet/++/ArkitScenes/
  Structured3D/S3DIS/HM3D)로 확장.
- **주의(관측된 함정):** ScanNet에서 이미지가 없는 배치(`img_num==0`)가 생기면
  정렬 항이 SSL loss로 대체되어 **정렬이 실제로 발동하지 않는** 경우가 있었음
  (관련 진단 커밋 존재). 데이터 준비 시 image/correspondence 누락을 점검할 것.

**가장 큰 미해결 사실:** merged 타깃 + SSL 정규화까지 해도 다운스트림 VSI는 원본
Utonia보다 떨어졌고, 특히 VG-LLM에서 낙폭이 컸음.

---

## 7. 분석 — 왜 떨어졌나

가장 의심되던 두 함정(merged 타깃, SSL)은 이미 처리한 상태이므로, 남는 설명:

1. **3D geometry 인코더를 2D image-text manifold에 맞추면 상보적 신호가 사라짐.**
   Qwen ViT는 2D/semantic 표현이고, VLM은 그걸 **이미 자기 비전 경로로** 갖고
   있음. 3D 인코더의 가치는 2D에 없는 *geometry*인데, 이를 Qwen manifold로
   끌어당기면 3D 분기가 2D와 중복되고 geometry 변별력은 줄어듦 → 공간추론에
   불리. **유력 1순위 가설**(§8에서 직접 검증 제안).
2. **5 epoch · indoor-only 미세조정이 Utonia의 도메인 일반화를 좁힘.** VSI/re-VSI는
   다양한 egocentric 장면 → 좁아진 표현이 불리.
3. **VG-LLM의 큰 낙폭 = geometry 의존 × CFG 증폭.** VG-LLM은 Qwen3.5보다 3D
   인코더 의존이 큼(강한 2D fallback이 없음). 3D feature가 나빠지면 더 타격이
   크고, CFG가 약해진 conditioning을 증폭.
4. **학습/추론 feature 구성 불일치 가능성(인코더 측).** 학습은
   `enc2d_upcast_level=3`(1332-d)을 형성하므로, 다운스트림 precompute
   (`demo/11_precompute_external.py`)가 동일 구성(1386-d open-ended가 아니라
   1332-d)을 추출하는지 확인 필요.

---

## 8. 다음 단계

1. **1순위 가설 직접 검증:** 원본 Utonia / 정렬-only / 정렬+SSL 세 인코더로 (a)
   다운스트림 VSI, (b) Qwen 타깃과의 CKA, (c) geometry probe(depth/normal/3D
   semseg linear-probe)를 함께 측정. 정렬이 CKA-to-Qwen은 올리되 geometry probe와
   VSI를 내리면, 2D 타깃 정렬이 이 인코더엔 잘못된 목표라는 결론.
2. **정렬-only vs 정렬+SSL**을 동일 데이터/스텝으로 나란히 학습해 mIoU·CKA 비교.
3. **학습↔추론 feature 구성 일치** 확인(1332-d, 3-level upcast).
4. **정렬 강도(`enc2d_loss_weight`)·스케줄 ablation**으로 "정렬이 해로움" vs
   "미세조정/도메인 협소화가 해로움"을 분리.
5. **타깃 재고:** 이미 강한 2D 경로를 가진 VLM을 도우려면, 인코더 전체를 2D
   manifold로 끌기보다 **geometry 보존형 목표** 또는 **backbone freeze + 작은
   adapter만 정렬**이 더 나을 수 있음.

---

## 부록 — 정리 내역

이 브랜치는 실패/구버전 distill 설정(`v1m3-0-base`, `v1m3-1-scannet-only`, ablation
스윕 다수)을 제거하고, **최종 2개 레시피**(정렬만 / 정렬+SSL)만 남겼습니다. 두
레시피는 실제 학습 데이터를 반영해 ScanNet 단독에서 indoor 멀티(ScanNet, ScanNet++,
ArkitScenes, Structured3D, S3DIS, HM3D)로 확장했습니다. eval 도구
(`training/tools/eval_alignment*.py`, `judge_qwen_utility.py`,
`eval-utonia-v1m1-dinov2-scannet.py`)는 유지했습니다. 제거된 구버전 설정만 쓰던
모델 모듈(`utonia_v1m2_qwen3_5_distill.py`, `utonia_v1m3a_qwen3_5_align_only.py`)은
현재 어떤 레시피에서도 쓰이지 않으니, 원하면 후속 정리에서 제거할 수 있습니다.
