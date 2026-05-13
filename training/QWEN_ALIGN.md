# Qwen × Utonia: 코드 역할과 학습 방향

이 문서는 `claude/analyze-qwen-uton-repo-sYaPf` 브랜치에서 추가한 두 학습
레시피(v1m3-A, v1m3-B)와, 그것을 *Qwen3.5 ViT ↔ Utonia(PTv3)* alignment
목표에 어떻게 운용할지를 정리합니다.

원본 v1m2 레시피는 alignment 손실(`enc2d_loss`)은 살아 있지만, 함께 켜져
있던 3D SSL 손실 쪽에 두 가지 치명적 버그가 있어 손실 가중치의 50% 가량이
사실상 noise 신호였습니다. v1m3-A / v1m3-B는 그 문제를 각각 다른 방식으로
해결한 두 갈래입니다.

---

## 1. 학습 setup 한 장 요약

```
┌──────────────────────────────────────────────────────────────┐
│  2D teacher (frozen)                                         │
│    Qwen3.5-4B `model.visual`                                 │
│    img(512×512, mean=std=0.5) → 1024-d patch token × 32×32   │
└────────────────────────────┬─────────────────────────────────┘
                             │ cosine sim target
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  patch_proj : Linear(1332 → 1024)        ◄── 유일한 bridge   │
└────────────────────────────▲─────────────────────────────────┘
                             │ projected 3D feature
                             │
┌──────────────────────────────────────────────────────────────┐
│  student PTv3 backbone (Utonia base, warm-start)             │
│    encoder out (576-d) → up_cast×3 → 1332-d @ stage-1        │
└──────────────────────────────────────────────────────────────┘
                       (선택) 3D self-distill
                       student.head ↔ teacher.head (EMA)
                       v1m3-A: 없음
                       v1m3-B: 있음 (가중치 ¼)
```

핵심 사실:

* **Qwen ViT는 어떤 레시피에서도 학습 대상이 아닙니다.** `requires_grad=False`,
  EMA loop에도 포함되지 않습니다. 항상 frozen target.
* **alignment의 결과물은 backbone 가중치에 남습니다.** `patch_proj`는 학습
  중에만 쓰이고 downstream(추론)에서는 backbone만 가져갑니다.
* **enc2d_upcast_level=3** → backbone encoder 출력에서 3번 upcast해 stage 1
  해상도의 1332-d feature를 align 타겟으로 씁니다. downstream에서 이 ckpt를
  쓸 때도 정확히 3번 upcast 해야 학습 때 표현과 일치합니다.

---

## 2. 두 레시피

### v1m3-A — alignment only (단순/안전)

**목적**: Qwen feature 분포에 backbone을 끌어당기는 가장 단순한 경로.

* 모델: `Utonia-v1m3a_qwen3_5_align_only`
* 학습 가능 모듈: `student.backbone` (낮은 LR) + `patch_proj` (full LR)
* 손실: `enc2d_loss` 단 하나 (1 − cosine_similarity).
* 제거된 것: teacher PTv3, EMA, mask/unmask/roll 손실, sinkhorn-knopp,
  모든 SSL 관련 scheduler, dead modules.

**언제 쓰나**
* "Qwen alignment가 학습되는지" 자체를 깨끗하게 검증할 때.
* 첫 학습 / 디버깅 / 작은 데이터셋(ScanNet-only) 스모크 테스트.
* `enc2d_loss` 단조감소 곡선만 보고 진단하고 싶을 때.

### v1m3-B — alignment + EMA-fixed 3D self-distill

**목적**: v1m3-A에 *기능하는* SSL regularizer를 더해 single-objective
mode-collapse 위험을 줄이고, backbone이 Qwen 특이적으로 과적합되는 것을
방지.

* 모델: `Utonia-v1m3b_qwen3_5_distill_ema`
* 학습 가능 모듈: `student.backbone` + `student.{mask,unmask}_head` + `patch_proj`
* teacher: `teacher.backbone` (EMA) + `teacher.{mask,unmask}_head` (EMA)
* 손실 비중: `{mask, roll, unmask, enc2d} = {1/16, 1/16, 1/8, 3/4}`
  (v1m2의 `{1/8, 1/8, 2/8, 4/8}`보다 alignment 비중을 ¾로 올렸음 — Qwen은
  DINOv2와 달리 image-text 학습된 VLM 표현이라 2D 신호가 더 강한 의미를
  싣고 있기 때문).

**v1m2 대비 패치 2개**
1. **init**: `teacher[k].load_state_dict(student[k].state_dict())` — teacher
   head를 student head와 동일 초기값으로 동기화.
2. **after_step**: DINO-style EMA 복원 — teacher backbone과 head 모두 매
   step student의 EMA로 갱신. Qwen ViT는 `self.enc2d_model`로 따로 살고
   이 loop에 안 잡힘.

**언제 쓰나**
* A가 안정적으로 수렴하는 걸 확인한 뒤 규모를 키울 때.
* 16-dataset 전체 학습으로 가는 단계.
* alignment가 너무 한 방향으로만 학습돼 다른 downstream(예: pure 3D
  segmentation linear-probe)이 떨어지는 신호가 보일 때.

---

## 3. 파일별 역할

### 신규 추가

| 파일 | 역할 |
|---|---|
| `training/pointcept/models/utonia/utonia_v1m3a_qwen3_5_align_only.py` | v1m3-A 모델 클래스 `Utonia-v1m3a_qwen3_5_align_only`. enc2d_loss 단일 경로만 남긴 미니멀 구현. |
| `training/pointcept/models/utonia/utonia_v1m3b_qwen3_5_distill_ema.py` | v1m3-B 모델 클래스 `Utonia-v1m3b_qwen3_5_distill_ema`. v1m2에 두 EMA 버그 패치 적용. |
| `training/configs/utonia/distill-utonia-v1m3-A-scannet-only-qwen3_5-4b.py` | v1m3-A용 ScanNet 단독 1-GPU 스모크 레시피. |
| `training/configs/utonia/distill-utonia-v1m3-B-scannet-only-qwen3_5-4b.py` | v1m3-B용 ScanNet 단독 스모크 레시피. 손실 가중치 `{1/16, 1/16, 1/8, 3/4}`. |

### 같이 수정한 파일

| 파일 | 무엇이 바뀌었나 |
|---|---|
| `training/configs/utonia/distill-utonia-v1m3-0-base-qwen3_5-4b.py` | 모든 `data_root`를 `${DATASET_ROOT}/data/<name>` 패턴으로 통일. (default `/group-volume/chaewon.yun/dataset`) |
| `training/configs/utonia/distill-utonia-v1m3-1-scannet-only-qwen3_5-4b.py` | 동일. |
| `training/pointcept/models/utonia/__init__.py` | 새 두 모델 모듈 import 등록. |
| `training/install_into_pointcept.sh` | 새 두 모델 파일 symlink 추가. config는 기존 `distill-utonia-v1m3-*.py` 글롭에 자동 포함. |

### 기존 자료 (참고)

| 경로 | 무엇 |
|---|---|
| `utonia/` | 추론·시각화용 패키지. 학습과 무관. |
| `demo/*.py` | PCA / similarity / sem-seg 등 데모. 학습 후 ckpt를 끼워 평가하는 용. |
| `third_party/Pointcept/` | Pointcept 서브모듈. 학습은 이 트리에서 `tools/train.py`로 실행. |
| `training/pointcept/models/utonia/utonia_v1m2_qwen3_5_distill.py` | 원본 fork 모델. 두 EMA 버그가 남아 있어, 단독 사용 시 SSL 신호가 noise. 비교용으로만 유지. |

---

## 4. Qwen-Utonia alignment를 위한 학습 방향

### 추천 순서 (위에서 아래로)

1. **사전 점검**
   ```bash
   python tools/test_qwen3_5_vit_path.py --model "$QWEN3_5_4B_PATH"
   # 기대: [4/4 PASS] with strategy=[manual: position_embeddings=(cos,sin) ...]
   ```
2. **A 레시피로 ScanNet 1-GPU 5-epoch 스모크**
   * 목표: `enc2d_loss`가 단조 감소하고 학습 그래프가 깨지지 않는지.
   * 100-200 step 내에 enc2d_loss가 의미 있게 떨어지면 alignment 자체가
     동작하는 것.
3. **B 레시피로 동일 ScanNet 스모크**
   * 추가로 `mask_loss / unmask_loss`가 nan 없이 finite하고 천천히
     떨어지는지 확인. (teacher head init copy + EMA가 동작한다는 신호)
   * `enc2d_loss` 감소 속도가 A 대비 비슷하거나 약간 빠르면 SSL이
     regularizer로 잘 동작.
4. **A 또는 B를 base recipe(16-dataset)로 확장**
   * 데이터 다양성이 alignment의 일반화에 가장 큰 영향. 단일 도메인
     (ScanNet only)은 indoor에만 휘므로 base recipe 전환이 중요.
5. **downstream 평가**
   * 추론 측에서 정확히 **3 upcast**를 돌려 1332-d feature 추출.
   * Sonata/Concerto가 사용하던 linear-probe 평가 (ScanNet semseg 등) 로
     align ckpt가 원본 Utonia 대비 손해를 보지 않는지 확인.

### Loss 곡선에서 봐야 할 것

* `enc2d_loss` — **이 작업의 핵심 시그널**.
  * 학습 초반에 빠르게 떨어지다가 plateau가 정상. cos 거리 손실의 ×10
    스케일이라 0~20 범위에서 시작해 5~10 부근으로 안정화되면 OK.
  * 단조 증가하면: warm-start ckpt 누락 / image normalize mean·std 불일치
    / `correspondence` 데이터 누락 의심.
* `mask_loss`, `unmask_loss` (B만)
  * `log(num_prototypes) ≈ log(4096) ≈ 8.3` 부근에서 시작해 천천히 감소.
  * 0에 너무 빨리 붙으면 mode-collapse 의심 → momentum을 올려 보거나
    SSL 가중치를 더 낮춤.
* `loss` (합산)
  * v1m3-B에서 `enc2d_loss × 0.75 + SSL × 0.25` 가 한 자리수 후반에서
    한 자리수 중반으로 내려가는 모양이 정상.

### Hyperparameter 조정 레버 (영향이 큰 순)

1. **`backbone_lr_scale`** (기본 0.05)
   * warm-start ckpt가 있을 때 0.05가 안전. 없으면 1.0으로 올려 backbone을
     처음부터 학습.
   * alignment가 너무 약하게 학습되면(enc2d_loss 정체) 0.1로 올리는 것도
     선택지. 단 너무 올리면 backbone이 원본 Utonia의 강점을 잃음.
2. **`enc2d_loss_weight`** (v1m3-B에서 3/4)
   * downstream에서 의미적(semantic) 성능이 더 필요하면 더 올림. 기하적
     강건성(geometric)을 지키고 싶으면 내림.
3. **`mask_jitter`, `mask_ratio_base`** (B만)
   * SSL 신호의 난이도 조절. mask_ratio_base를 0.6으로 내리면 SSL이 좀 더
     관대해져 alignment와 충돌이 줄어듦.
4. **`enc2d_upcast_level`** (기본 3)
   * 3 → 1332-d at stage-1. 더 fine한 alignment를 원하면 4(1386-d stage-0).
     단 downstream upcast 횟수도 반드시 같이 바꿔야 함.
5. **`momentum_base`** (B만, 기본 0.994)
   * EMA 속도. 손실이 noisy하면 0.996으로 올림.

### Failure mode 와 진단

| 증상 | 가능한 원인 | 확인/대응 |
|---|---|---|
| `enc2d_loss` 증가 또는 plateau (시작부터) | warm-start ckpt 미적용 | `[v1m3{a,b} warm-start] ... loaded=K` 로그에서 K가 0이 아닌지 확인 |
| `enc2d_loss` nan | image normalize 불일치 (ImageNet stats로 들어감) | config의 `Imgnormalize` mean/std가 `(0.5,0.5,0.5)`인지 확인 |
| `enc2d_loss` finite, but `mask_loss` noisy / 안 떨어짐 (B만) | EMA 또는 head-init 패치가 적용 안 됨 | `Utonia-v1m2_qwen3_5_distill` (구버전) 을 쓰고 있지 않은지 config의 `model.type` 확인 |
| 학습은 되는데 downstream linear-probe 손해 | alignment가 너무 강해 backbone 표현이 Qwen-특이적으로 휨 | A→B로 전환, 또는 `enc2d_loss_weight`를 0.5~0.6으로 내림 |
| 메모리 부족 | Qwen ViT가 무거움 (4B) | batch_size 줄이거나, `crop_h/w` 를 384 로 내림 (단 `patch_h/w` 도 같이) |

### A vs B 선택 가이드 (한 줄)

* **데이터/시간이 빠듯하고 alignment 결과만 깔끔하게 보고 싶다 → A**
* **base 16-dataset으로 본 학습을 돌릴 거고, 다중 도메인에서 backbone이
  지나치게 한쪽으로 휘는 걸 방지하고 싶다 → B**

---

## 5. 실행 명령 (cheat sheet)

```bash
# 1회 셋업
git submodule update --init --recursive
bash training/install_into_pointcept.sh

cd third_party/Pointcept
export QWEN3_5_4B_PATH=/data/hf/Qwen3.5-4B
export UTONIA_PRETRAINED_CKPT=/data/ckpt/utonia.pth
# DATASET_ROOT 는 기본값 /group-volume/chaewon.yun/dataset

# v1m3-A : alignment only
python tools/train.py \
    --config-file configs/utonia/distill-utonia-v1m3-A-scannet-only-qwen3_5-4b.py \
    --num-gpus 1 --options save_path=exp/utonia_q35_align_only

# v1m3-B : alignment + EMA-fixed SSL regularizer
python tools/train.py \
    --config-file configs/utonia/distill-utonia-v1m3-B-scannet-only-qwen3_5-4b.py \
    --num-gpus 1 --options save_path=exp/utonia_q35_align_ssl

# 16-dataset 본 학습으로 확장 (필요시 base recipe의 model.type을
# Utonia-v1m3b_qwen3_5_distill_ema 로 바꿔 사용)
python tools/train.py \
    --config-file configs/utonia/distill-utonia-v1m3-0-base-qwen3_5-4b.py \
    --num-gpus 8 --options save_path=exp/utonia_q35_base
```
