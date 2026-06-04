# Qwen3.5 ViT distillation 성능 저하 원인 가설

DINOv2/SigLIP/RADIO → **Qwen3.5 vision tower** 로 distillation 타깃을 교체한 뒤,
finetune한 Utonia를 VSI / re-VSI bench에 평가한 결과 **기존 Utonia보다 성능이 하락**했다.

- `qwen3.5 + cfg + utonia encoder` : 성능 저하 (소폭~중폭)
- `vg-llm + cfg + utonia encoder` : 대폭 저하

아래는 그 원인에 대한 **가설 모음**이다. (검증/개선은 하지 않음. 분석 기록 목적.)

분석 대상 코드는 distill 모듈의 두 부분으로 한정한다.
- **H** : `ENC2D_forward` — Qwen3.5 ViT teacher feature 추출
  (`training/pointcept/models/utonia/utonia_v1m2_qwen3_5_distill.py:351-412`)
- **I** : `forward` 내부 `enc2d_loss` 블록 — 2D↔3D cosine 정렬 loss
  (`...utonia_v1m2_qwen3_5_distill.py:766-865`)

관련 진단 스크립트: `training/tools/judge_qwen_utility.py`

---

## 가설 1. distillation 타깃이 "LM이 실제로 쓰는 feature"가 아니다 (pre-merge raw block 출력)

H(`ENC2D_forward`)는 Qwen ViT의 block 마지막 출력(1024-dim)을 그대로 타깃으로 쓴다.
`merger.norm`, 2×2 spatial merge, `merger.mlp`(LLM hidden dim projection)를 **전혀 거치지 않는다.**

```python
# ENC2D_forward (H), :404-412
for blk in self.enc2d_model.blocks:
    hidden = blk(hidden, cu_seqlens=..., position_embeddings=...)
hidden = hidden.view(B, h * w, -1)   # merger 없이 여기서 종료
return hidden
```

- 모듈 docstring(`:5-8`)은 이걸 *"lands in the same manifold as the LM's input ViT features"* 로 정당화하지만,
  **LM이 실제 입력으로 받는 ViT feature는 merger를 통과한 post-merge 토큰**이다.
  pre-merge raw block 출력과 LM 입력 사이에는 `norm + 2×2 merge + MLP`가 끼어 있다.
- 즉 "VSI-bench가 Qwen을 쓰니 Qwen에 align하면 좋아질 것"이라는 전제가, 실제로는
  **Qwen이 쓰지도 않는 중간 representation**에 align하는 것으로 구현돼 있다.

### 보강 증거 (코드 내부 불일치)

진단 스크립트 `judge_qwen_utility.py:99-132`의 `qwen_merged_features`는
`merger.norm → 2×2 reorder → merger.mlp`를 **모두 적용**하고,
주석에 *"use_full_merger=True so we measure exactly what H/I distills from"* 라고 적혀 있다.

→ **진단 스크립트는 full-merger 출력을 측정하는데, 정작 H는 pre-merge를 distill한다.**
   측정 대상과 실제 distill 대상이 서로 다른 representation이다.

---

## 가설 2. Qwen raw-block feature 자체가 dense distillation 타깃으로 부적합 (rank collapse / anisotropy)

`judge_qwen_utility.py` docstring(`:8-14`)이 직접 기술:

- **DINOv2** : iBOT + multi-crop DINO 학습 → 각 patch가 self-contained semantic unit
  (high rank, well-spread, locally discriminative) → dense per-patch distillation에 이상적.
- **Qwen3.5 ViT** : LLM 소비용으로 학습, 유효 정보는 2×2 merger 이후 / LLM-hidden-dim 스케일에 존재.
  *"per-patch features are anisotropic and rank-collapsed at the raw-block level."*

→ H가 타깃으로 쓰는 것이 바로 그 "raw-block level"의 rank-collapse된 feature.
   DINOv2 대비 **정보량이 적고 방향성이 치우친(anisotropic)** 신호를 per-point로 강제하는 셈.

---

## 가설 3. 정렬 방식이 robustness를 잃었다 — cluster head 미사용 + plain cosine

I(`enc2d_loss`)는 `__init__`에서 `enc2d_head_student` / `enc2d_head_teacher`
(OnlineCluster = prototype + sinkhorn)를 만들고 prototype까지 load 하지만(`:229-235`),
**loss 계산에서 한 번도 호출하지 않는다 (dead code).**

실제 loss(`:854-862`):

```python
if self.enc2d_cos_shift:
    feature2d_mask = feature2d_mask - feature2d_mask.mean(-1, keepdim=True)
    feature3d_mask = feature3d_mask - feature3d_mask.mean(-1, keepdim=True)
loss = (1 - cos(feature2d_mask, feature3d_mask)).mean() * 10
```

- `patch_proj`(1332→1024) 출력과 raw ViT feature 사이의 **plain cosine regression**.
- 원래 2D-3D align이 prototype-assignment(분포 매칭) 기반이었다면, 그 robustness를 버리고
  직접 회귀로 바뀐 형태.

---

## 가설 4. anisotropic feature + cosine → 소수 outlier 채널이 loss를 지배

- rank-collapse / anisotropic feature는 소수의 massive-activation 채널이 norm을 독식한다.
- `cos_shift`의 채널-mean 빼기로는 이 outlier가 제거되지 않는다.
- cosine은 scale-invariant → 결국 **그 1~2개 outlier 채널 방향만 맞추는** 쪽으로 학습이 진행.
- 3D encoder가 기하학적으로 의미 있는 고차원 feature 공간을 버리고
  **저rank·anisotropic manifold로 끌려간다.**
- VSI / re-VSI가 의존하는 **geometric fidelity가 이 과정에서 손상**된다.

---

## 가설 5. 잘못된 타깃에 대한 정렬 loss의 가중치가 과도하게 크다

config(`distill-utonia-v1m3-0-base-qwen3_5-4b.py:158`):

```python
enc2d_loss_weight = 4 / 8   # = 0.5
```

게다가 loss 내부에 `* 10`(`:862`)이 곱해진다.

→ 전체 loss의 절반이 "부적합한 타깃에 대한 cosine 회귀"이며,
   backbone을 그 방향으로 **강하게** 끌고 간다.
   (warm-start로 보존하려던 Utonia의 geometric feature를 그만큼 강하게 덮어쓴다.)

---

## 가설 6. encoder feature 분포/스케일 변화 → downstream(vg-llm)에서 대폭 저하

- H가 final norm 없는 raw block manifold로 끌고 가면서, encoder 출력의 **분포/스케일 자체가 변한다.**
- `vg-llm + cfg + utonia encoder` 파이프라인은 3D encoder feature를 LLM 쪽으로 더 직접 흘려보내는데,
  downstream projection은 **원래 Utonia 통계**에 맞춰져 있어 정합이 크게 깨진다.
- anisotropic 방향으로의 collapse는 토큰 다양성(유효 정보량)을 줄여 LLM의 공간 추론 입력을 더 크게 훼손.
- → `qwen3.5+cfg`(encoder feature를 비교적 간접 사용)보다 `vg-llm+cfg`에서 **저하 폭이 훨씬 큰** 현상과 일치.

---

## 가설 7. (확인 필요) windowed attention / token order 미처리로 teacher feature 자체가 부정확할 가능성

H는 모든 block에 **이미지 전체 단일 `cu_seqlens`** 만 넘기고(`:399-409`),
Qwen2.5-VL 계열에서 쓰는 `window_index` 재배열 및 window별 `cu_seqlens`를 적용하지 않는다.

- Qwen3.5 ViT가 일부 layer에서 windowed attention을 쓴다면, teacher feature가 **틀리게 계산**된다.
- "에러 없이 forward 된다 ≠ 의미상 올바른 feature" — `cu_seqlens`만 받도록 fix 되어 돌아가더라도
  windowing이 빠지면 출력이 달라진다.
- 확인 지점: `tools/test_qwen3_5_vit_path.py`의 `[4/4 INFO] block forward sig` 출력,
  그리고 config의 `fullatt_block_indexes` / `window_size` 존재 여부.

---

## 종합

핵심 전제("VSI-bench가 Qwen을 쓰니 Qwen에 align하면 좋아진다")의 함정:

> VSI-bench는 Qwen을 **LM(언어 추론) backbone**으로, **post-merger 2D 토큰** 위에서 사용한다.
> 3D encoder의 가치는 "Qwen ViT patch 공간을 흉내내는 것"이 아니라 **geometric fidelity**에 있으며,
> 그것은 DINOv2 distillation이 더 잘 보존했다.

그런데 실제 구현(H/I)은 Qwen의 가장 불리한 버전(**pre-merge, rank-collapsed raw block**)을,
**robustness 없는 방식**(unused cluster head + plain cosine),
**과도한 가중치**(0.5 × 내부 10배)로 align했다.
→ 기존 Utonia 대비 성능이 하락하고, encoder feature를 직접 소비하는 vg-llm에서
   저하 폭이 더 커지는 결과로 이어졌다는 것이 본 문서의 가설이다.
