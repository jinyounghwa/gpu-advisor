# 강화학습 학습 루프 심층 분석 (GPU Advisor)

> 본 문서는 GPU Advisor의 강화학습 학습 과정을 이론과 코드 수준에서 상세히 설명합니다.
> 2026-03-22 기준: 30일 실데이터 학습 완료, 릴리즈 태그 `release-agent-20260322-105138` 배포.

---

## 1. 학습 전체 흐름

```
실제 시장 데이터 (30일)
        │
        ▼
feature_engineer.py → 256D 상태 벡터 생성
        │
        ▼
AgentFineTuner._build_samples()
  ├── 연속된 두 날의 상태 (s_t, s_{t+1})
  ├── 가격 변화율 → 행동 레이블 (BUY/WAIT/HOLD/SKIP)
  └── 실제 보상값 r_t
        │
        ▼
AgentFineTuner._train_step()  ← 500 steps × batch 32
  ├── h(s_t) → z_t (잠재 상태)
  ├── g(z_t, a_t) → ẑ_{t+1}, μ, log σ²
  ├── f(z_t) → π_logits, v
  ├── a(z_t) → action_prior_logits
  └── 5개 손실 합산 → AdamW.step()
        │
        ▼
alphazero_model_agent_latest.pth 저장
        │
        ▼
AgentEvaluator.evaluate() → 7개 품질 게이트
        │
        ▼
ReleasePipeline.run() → docs/reports/ 리포트 생성
```

---

## 2. 샘플 생성 로직

### 2.1 연속 날짜 쌍 구성

```python
# fine_tuner.py: _build_samples()
for i in range(len(dates) - 1):
    date_t   = dates[i]    # 현재 날짜
    date_t1  = dates[i+1]  # 다음 날짜

    state_t  = load_state(date_t)    # 256D 벡터
    state_t1 = load_state(date_t1)   # 256D 벡터 (타겟)

    # 각 GPU 모델마다 독립 샘플 생성
    for gpu_model in gpu_models:
        price_t  = get_price(date_t,  gpu_model)
        price_t1 = get_price(date_t1, gpu_model)
        pct_change = (price_t1 - price_t) / price_t  # 가격 변화율

        # 행동 레이블: 내일 가격이 올랐으면 오늘 BUY가 좋았다
        action = _action_from_delta(pct_change)
        reward = pct_change (BUY_NOW의 경우) 또는 -pct_change (WAIT의 경우)
```

### 2.2 행동 레이블링 규칙

```python
def _action_from_delta(pct_change: float) -> int:
    if pct_change >= 0.02:    return 0  # BUY_NOW  — 2% 이상 상승 예정
    if pct_change <= -0.05:   return 4  # SKIP     — 5% 이상 하락 예정
    if pct_change <= -0.02:   return 2  # WAIT_LONG — 2~5% 하락 예정
    if pct_change <= -0.005:  return 1  # WAIT_SHORT — 0.5~2% 하락 예정
    return 3                            # HOLD     — 변동 없음
```

**주의사항**: 이 레이블은 **다음 날 가격 기준의 역산(hindsight)**입니다. 즉 "어제 살 걸 알았다면" 기준의 의사 레이블(pseudo-label)입니다. 학습 신호는 노이즈가 있지만 30일 × 24 GPU = 720개 샘플에서 패턴을 추출합니다.

---

## 3. 배치 학습 스텝

### 3.1 순전파 (Forward Pass)

```python
def _train_step(self, batch):
    states, actions, rewards, next_states = batch

    # 1. 현재 상태 잠재 표현
    z_t = self.h(states)                          # (B, 256) → (B, 256)

    # 2. World Model 예측
    action_onehot = F.one_hot(actions, 5).float()
    z_hat_t1, reward_mu, reward_logvar = self.g(z_t, action_onehot)
    # z_hat_t1: 예측된 다음 상태
    # reward_mu, reward_logvar: Gaussian 보상 분포

    # 3. 다음 상태 실제 표현 (타겟)
    with torch.no_grad():
        z_t1 = self.h(next_states)                # 그래디언트 단절

    # 4. 정책/가치 예측
    policy_logits, value = self.f(z_t)

    # 5. 행동 사전 확률
    action_prior_logits, _ = self.a(z_t.detach())  # 그래디언트 단절
```

### 3.2 손실 계산

```python
    # 손실 1: World Model 상태 전이 일관성
    latent_loss = F.mse_loss(z_hat_t1, z_t1)

    # 손실 2: 정책 학습 (MCTS 타겟이 없으므로 행동 레이블 사용)
    policy_loss = F.cross_entropy(policy_logits, actions)

    # 손실 3: 가치 예측 (실제 보상을 타겟으로)
    value_loss = F.mse_loss(value.squeeze(), rewards)

    # 손실 4: Gaussian NLL (보상 불확실성 학습)
    reward_nll = 0.5 * ((rewards - reward_mu)**2 * torch.exp(-reward_logvar) + reward_logvar)
    reward_loss = reward_nll.mean()

    # 손실 5: 행동 사전 확률 (클래스 균형 보정)
    action_prior_loss = F.cross_entropy(action_prior_logits, actions, weight=class_weights)

    # 최종 합산 (가중치)
    total_loss = (
        1.0 * latent_loss
      + 1.0 * policy_loss
      + 1.0 * value_loss
      + 0.5 * reward_loss
      + 0.3 * action_prior_loss
    )
```

### 3.3 역전파 및 그래디언트 클리핑

```python
    optimizer.zero_grad()
    total_loss.backward()

    # 그래디언트 클리핑 (폭발 방지)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(self.h.parameters()) +
        list(self.g.parameters()) +
        list(self.f.parameters()) +
        list(self.a.parameters()),
        max_norm=1.0
    )
    # 500 steps 후 grad_norm ≈ 4.91 (안정적 범위)

    optimizer.step()
```

---

## 4. 학습 과정에서 각 지표의 변화

### 4.1 loss 값 해석

| 지표 | 500 스텝 결과 | 해석 |
|------|-------------|------|
| total_loss | **-0.74** | 음수는 Gaussian NLL이 지배적일 때 발생 (정상) |
| policy_loss | **0.244** | 행동 분류 정확도 개선 중 |
| value_loss | **0.016** | 보상 기댓값 예측 오차 (작을수록 좋음) |
| reward | **+0.014** | 배치 평균 보상 (양수 = 이익 구간 학습 중) |
| entropy | **0.350** | 정책 다양성 (0.25 이상 = 모드 붕괴 아님) |
| win_rate | **75%** | 에피소드 75%에서 양의 보상 달성 |
| grad_norm | **4.91** | 클리핑 1.0 기준 상대적으로 크나 안정적 |

### 4.2 총 loss가 음수인 이유

```
total_loss = latent_loss + policy_loss + value_loss + 0.5×reward_loss + 0.3×prior_loss
           ≈ 0.05 + 0.24 + 0.02 + 0.5×(-2.3) + 0.3×0.8
           ≈ 0.31 + (-1.15) + 0.24
           ≈ -0.60
```

Gaussian NLL 손실(`reward_loss`)은 `μ`와 `σ²`가 잘 보정되면 음수가 될 수 있습니다. 이는 모델이 보상의 분포를 잘 학습했다는 신호로, 학습 실패가 아닙니다.

---

## 5. 평가 지표 상세 — 7개 품질 게이트

```python
# evaluator.py: evaluate()
# day(t) 의사결정 vs day(t+1) 실현 가격으로 독립 백테스트
```

| 게이트 | 기준 | 달성값 | 의미 |
|--------|------|--------|------|
| accuracy_raw | ≥ 0.55 | **0.894** | BUY vs WAIT 방향 정확도 |
| reward_raw | > 0.0 | **+0.0064** | 평균 의사결정당 보상 |
| abstain | ≤ 0.85 | **0.788** | 관망(HOLD) 비율 허용 한도 |
| safe_override | ≤ 0.90 | **0.384** | 안전모드 개입 비율 |
| action_entropy_raw | ≥ 0.25 | **1.459** | 행동 다양성 (모드 붕괴 방지) |
| uplift_raw_vs_buy | ≥ 0.0 | **+0.0040** | always_buy 전략 대비 초과 성과 |
| no_mode_collapse_raw | True | **True** | 특정 행동 과집중 없음 |

### 5.1 방향정확도 89.4%의 의미

```
전체 631개 평가 샘플 중:
  - BUY_NOW 선택 후 다음 날 가격 상승: 올바른 결정
  - WAIT 선택 후 다음 날 가격 하락: 올바른 결정
  - 이 두 케이스가 전체의 89.4%

비교:
  always_buy: 11.1% (GPU 시장에서 구매 직후 가격 하락이 더 흔함)
  에이전트: 89.4% → 78.3%p 개선
```

### 5.2 관망비율 78.8%의 해석

에이전트는 전체 의사결정의 78.8%를 HOLD로 처리합니다. 이는 다음을 의미합니다:
- 확신이 없을 때는 행동하지 않음 (safe_mode 동작)
- 10번의 기회 중 2번만 실제 구매/대기 행동 추천
- 나머지 8번은 "지금은 판단하기 어렵다" (HOLD)

이는 금융 도메인에서 **보수적 에이전트의 건전한 특성**입니다.

---

## 6. 재학습 정책 (Post-30d)

### 6.1 7일 주기 자동 재학습

```python
# auto_training.py
if newly_accumulated_days >= retrain_every_days:  # 7일
    action = "train_release"  # 500 steps 재학습 + 릴리즈 게이트
else:
    action = "release_check"  # 드라이 체크만
```

### 6.2 재학습 시 기대 효과

| 시점 | 데이터 | 예상 개선 |
|------|--------|---------|
| 첫 학습 (2026-03-22) | 30일 (631 샘플) | 기준선 확립 |
| 재학습 1 (2026-03-29) | 37일 | 추가 패턴 학습 |
| 재학습 2 (2026-04-05) | 44일 | 계절성 패턴 포착 시작 |
| 재학습 N (60일+) | 60일+ | 안정적 프로덕션 품질 |

---

## 7. 현재 한계와 개선 방향

### 7.1 데이터 희소성

현재 학습 데이터:
- 30일 × 24 GPU 모델 = **720개 샘플** (파라미터 수 ~19M 대비 극히 적음)
- 동일 데이터가 500 스텝 × 배치 32 = 약 **22회 재사용**

개선 방향:
- 데이터 누적으로 자연 해소 (매 7일 재학습)
- 데이터 증강: 시계열 윈도우 슬라이딩

### 7.2 World Model 검증 부재

현재 릴리즈 게이트는 **에이전트 성능**(방향정확도, 보상)만 측정합니다. World Model 자체의 품질, 즉 `g(s,a) → ẑ_{t+1}`의 예측 정확도(next-state MSE)는 독립적으로 검증되지 않습니다.

향후 추가 게이트:
```python
# 제안: World Model 품질 게이트
world_model_mse = mean((z_hat_{t+1} - z_{t+1})²)
assert world_model_mse < threshold, "World Model 품질 미달"
```

---

*작성: 2026-03-22 | GPU Advisor 프로젝트 기술 문서*
