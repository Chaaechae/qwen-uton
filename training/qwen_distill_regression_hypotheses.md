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

## 가설 1. merge level을 어떻게 잡아도 좋은 dense 타깃이 안 나온다 (rank vs manifold 트레이드오프)

> **정정 (저자 기억):** pre-merge는 실수가 아니라 **의도된 선택**이었다.
> post-merge feature로 시도했을 때 **rank가 너무 낮아서** pre-merge(raw block 출력)로 바꿨다.

이 트레이드오프 자체가 핵심 문제다. 두 선택지 모두 dense distillation 타깃으로 결함이 있다.

| 타깃 | 토큰 수(512/16) | 문제 |
|------|----------------|------|
| **post-merge** (LM이 실제 쓰는 것) | 16×16 = 256 | rank가 너무 낮음 (2×2+MLP로 압축·LM 정렬되어 dense 신호로 부적합) |
| **pre-merge raw block** (H가 쓰는 것) | 32×32 = 1024 | rank는 높지만 anisotropic, 그리고 **LM이 실제로 쓰는 manifold가 아님** |

H(`ENC2D_forward`)는 후자를 택한다 — `:364` 주석대로 *"bypass the spatial merger"*,
`:411`에서 `(B, h*w, -1)` = `(B, 32×32, 1024)` 토큰을 그대로 반환한다.

```python
# ENC2D_forward (H), :404-412
for blk in self.enc2d_model.blocks:
    hidden = blk(hidden, cu_seqlens=..., position_embeddings=...)
hidden = hidden.view(B, h * w, -1)   # merger 없이 여기서 종료
return hidden
```

- 모듈 docstring(`:5-8`)은 pre-merge를 *"lands in the same manifold as the LM's input ViT features"* 로 정당화하지만,
  엄밀히는 **LM이 입력으로 받는 ViT feature는 merger 통과 후(post-merge) 토큰**이다.
  pre-merge와 LM 입력 사이에는 `norm + 2×2 merge + MLP`가 끼어 있다.
- 즉 "rank를 살리려고 pre-merge" vs "LM이 실제 쓰는 건 post-merge" 사이에서,
  **rank를 택했지만 그 대가로 LM이 쓰지 않는 off-manifold representation에 align**하게 됐다.
- 정리: post-merge는 rank가 없어서 못 쓰고, pre-merge는 LM manifold가 아니라서 효과가 약하다.
  **Qwen ViT에는 "rank도 충분하면서 LM이 실제로 쓰는" 좋은 dense 타깃 layer가 없다**는 것이
  더 근본적인 가설이다 (→ 가설 2와 직결).

### 미해소 불일치 — 2×2 merge 적용 여부

저자 기억상으로는 "완전한 pre-merge가 아니라 2×2 merge도 끼어 있었다"고 한다.
그러나 **현재 브랜치에 커밋된 `ENC2D_forward`에는 2×2 merge가 없다** (1024 토큰 그대로 반환,
`feature_index`(`:839-841`)도 `patch_h*patch_w`=32×32 해상도). 2×2 merge가 등장하는 곳은
진단 스크립트 `judge_qwen_utility.py:127-129`의 full-merger 경로뿐이다.

→ 따라서 둘 중 하나다:
  1. 실제 평가한 run이 이 커밋이 아니라 **2×2 merge가 들어간 미커밋 변형 버전**이었다, 또는
  2. 기억이 진단 스크립트 쪽과 섞였다.

  만약 (1)이라면 평가된 모델의 실제 토큰 수/해상도는 16×16=256이며, 위 표의 "그 중간"
  (rank는 post-merge보다 약간 높고 manifold는 여전히 off) 지점에 해당한다. **재현·결론을 위해
  실제 평가 run의 merge 설정을 먼저 확정해야 한다.**

---

## 가설 2. Qwen ViT feature 자체가 dense distillation 타깃으로 부적합 (rank collapse / anisotropy — merge level 무관)

`judge_qwen_utility.py` docstring(`:8-14`)이 직접 기술:

- **DINOv2** : iBOT + multi-crop DINO 학습 → 각 patch가 self-contained semantic unit
  (high rank, well-spread, locally discriminative) → dense per-patch distillation에 이상적.
- **Qwen3.5 ViT** : LLM 소비용으로 학습, 유효 정보는 2×2 merger 이후 / LLM-hidden-dim 스케일에 존재.
  *"per-patch features are anisotropic and rank-collapsed at the raw-block level."*

핵심: **rank 문제는 어느 merge level에서도 해소되지 않는다.**
- post-merge → rank가 너무 낮음 (가설 1에서 pre-merge로 바꾼 이유).
- pre-merge raw block → rank는 상대적으로 높지만 **anisotropic & rank-collapsed** (docstring 표현).

→ 결국 Qwen ViT는 **DINOv2와 달리 어떤 layer를 잘라도 좋은 dense per-patch 타깃이 되지 못한다.**
   이것이 "DINOv2 → Qwen 교체"가 성능을 떨어뜨린 가장 근본적인 원인 가설이다.

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

그런데 Qwen ViT는 **어느 merge level에서도 좋은 dense 타깃이 없다** (post-merge는 rank가 너무 낮아
pre-merge로 갔지만, pre-merge는 anisotropic & off-manifold). 여기에 더해 정렬 방식마저
**robustness 없는 방식**(unused cluster head + plain cosine), **과도한 가중치**(0.5 × 내부 10배)로 구현됐다.
→ 기존 Utonia 대비 성능이 하락하고, encoder feature를 직접 소비하는 vg-llm에서
   저하 폭이 더 커지는 결과로 이어졌다는 것이 본 문서의 가설이다.

> 주의: pre-merge 선택 자체는 저자의 **의도된 트레이드오프**(post-merge rank 부족 회피)였으며,
> 단순 구현 실수가 아니다. 또한 평가 run이 실제로 2×2 merge를 포함했는지(가설 1의 미해소 불일치)에 따라
> 일부 가설의 강도가 달라질 수 있으므로, 결론 확정 전 실제 평가 setup의 merge 설정 확인이 필요하다.
