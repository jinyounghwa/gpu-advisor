# WORLD MODEL 이론과 구현 (GPU Advisor 기준)

이 문서는 GPU Advisor를 기준으로, 단순 데이터 최적화(optimizer) 사고에서 월드모델 기반 예측/계획(planning) 사고로 전환하기 위한 학습 문서입니다.

## 1. 핵심 메시지: 전환의 방향

결론부터 말하면 방향은 맞습니다.  
다만 "데이터 기반 의사결정"을 버리는 것이 아니라 다음처럼 재정의해야 합니다.

1. 데이터는 정답을 바로 뽑는 입력이 아니라, 세계를 학습하는 재료다.
2. 모델은 단일 시점 점수화기가 아니라, 미래를 전개하는 시뮬레이터다.
3. 의사결정은 점수 최대화가 아니라, 시뮬레이션 기반 계획 문제다.

즉, `optimizer 중심`에서 `world model + planner 중심`으로 무게중심을 이동하되,
데이터는 더 중요해집니다. (학습 데이터 + 운영 검증 데이터 + 반례 데이터)

## 2. 왜 Optimizer만으로는 한계가 생기는가

Optimizer 성향(예: "지금 feature에서 행동 점수 argmax")은 실무에서 빠르지만 다음 문제가 있습니다.

1. 비정상(non-stationary) 환경 취약
- 환율/재고/프로모션/정책 변화처럼 체제 변환(regime shift)이 오면 과거 패턴 최적화가 깨집니다.

2. 행동-결과 인과 분리 부족
- "무슨 일이 일어났는지"는 설명하지만, "내 행동 때문에 무엇이 달라졌는지"를 분리해 다루기 어렵습니다.

3. 멀티스텝 의사결정 손실
- 오늘의 최고점 행동이 2주 뒤에는 손해가 될 수 있는데, 단일 스텝 최적화는 이 경로 의존성을 잘 반영하지 못합니다.

4. 안전/보수적 운영 어려움
- 불확실성이 큰 구간에서 "보류"를 택해야 하는데, 확신 과대 모델은 과감한 행동을 과잉 추천할 수 있습니다.

## 3. 월드모델 관점에서의 의사결정 구조

GPU Advisor의 MuZero 스타일 분해:

1. `h` (Representation): 관측 상태 `x_t`를 잠재 상태 `z_t`로 압축
2. `g` (Dynamics): `z_t, a_t`에서 `z_{t+1}`과 보상/리스크를 전개
3. `f` (Prediction): 각 상태의 정책 prior와 가치 예측
4. MCTS (Planner): 여러 행동 시나리오를 탐색해 장기 기대값 기준으로 행동 선택

수식 요약:
- `z_t = h(x_t)`
- `(z_{t+1}, r_t) = g(z_t, a_t)`
- `(pi_t, v_t) = f(z_t)`

의사결정은 `f` 단독 출력이 아니라 `search(h, g, f)` 결과를 사용합니다.

## 4. 현재 코드와 1:1 매핑

- Representation (h): `backend/models/representation_network.py`
- Dynamics (g): `backend/models/dynamics_network.py`
- Prediction (f): `backend/models/prediction_network.py`
- Action Prior (a): `backend/models/action_model.py`
- Planner(MCTS): `backend/models/mcts_engine.py`
- Orchestration: `backend/agent/gpu_purchase_agent.py`

수식 보완:
- `z_t = h(x_t)`
- `(z_{t+1}, r_t) = g(z_t, a_t)`
- `(pi_t, v_t) = f(z_t)`
- `prior_a = a(action_id)` — ActionModel: 학습된 행동 임베딩 기반 사전 확률

운영 액션 라벨:
- `BUY_NOW`
- `WAIT_SHORT`
- `WAIT_LONG`
- `HOLD`
- `SKIP`

운영 경로 요약:
1. 상태 벡터(256D) -> `representation_network`로 256D latent 변환
2. `mcts_engine.search`가 `prediction_network`와 `dynamics_network`를 반복 호출
3. 방문수 기반 정책(45%) + 보상 신호(25%) + f-net prior(15%) + ActionModel prior(15%) 4-신호 혼합
4. 안전 게이트(신뢰도/엔트로피)를 거쳐 최종 액션 선택

## 5. 학습 프레임: 무엇을 공부해야 하는가

### 5.1 모델 구조 이해
1. `h/g/f` 분해 이유: 학습 안정성 + 계획 가능성
2. latent 차원(`state_dim`, `latent_dim`)과 액션 차원(`action_dim`) 정합성
3. reward/value/entropy의 해석

### 5.2 계획 알고리즘 이해
1. MCTS의 `select -> expand -> simulate -> backup`
2. 탐색-활용 균형(exploration coefficient)
3. `num_simulations`, `rollout_steps`, `discount_factor`의 효과

### 5.3 데이터/운영 연결 이해
1. 데이터 수집: `crawlers/run_daily.py`
2. 학습 입력: `data/processed/dataset/training_data_YYYY-MM-DD.json`
3. 운영 품질 모니터링: `docs/reports/latest_data_status.json`
4. 릴리즈 게이트: `python3 backend/run_release_ready.py`

## 6. 전환 실행안: Optimizer -> World Model 조직 습관

### 6.1 의사결정 질문 바꾸기
- Before: "지금 feature에서 점수가 가장 높은 행동은?"
- After: "행동별 미래 경로를 전개했을 때 기대가치/리스크/불확실성이 어떤가?"

### 6.2 KPI 바꾸기
- Before: 단기 정확도/즉시 reward
- After:
1. n-step 누적 보상
2. 계획 대비 실제 오차(trajectory consistency)
3. 불확실성 높은 구간에서의 보수적 의사결정 비율
4. 대기 전략(`WAIT_*`)의 사후 효용

### 6.3 운영 정책 바꾸기
1. 고신뢰 구간: 계획 결과 적극 반영
2. 저신뢰 구간: 보류/관측 모드로 전환
3. 급격한 환경변화 감지 시: 탐색 계수 및 안전 게이트 자동 상향

## 7. 반론 가능성과 균형점

### 반론 1: "월드모델은 복잡하고 비용이 크다"
맞습니다. 하지만 실환경이 멀티스텝/비정상일수록 단순 모델의 숨은 비용(오판, 재학습 반복)이 더 커집니다.

### 반론 2: "데이터 품질이 낮으면 월드모델도 무너진다"
맞습니다. 그래서 전환의 핵심은 모델 교체가 아니라 데이터 거버넌스 강화입니다.

### 균형점
최적 접근은 `Optimizer 폐기`가 아니라 `Optimizer를 월드모델 플래너의 특수 케이스로 내재화`하는 것입니다.
즉, 단기 점수화는 planner의 휴리스틱으로 유지하고, 최종 결정은 계획 기반으로 승격합니다.

## 8. 팀 적용 체크리스트

1. 문서/회의에서 "예측 정확도"만이 아니라 "계획 품질" 지표를 함께 보고 있는가
2. 실패 사례를 단일 시점 오분류가 아니라 "시뮬레이션 경로 붕괴" 관점으로 리뷰하는가
3. `latest_data_status`를 릴리즈 전 필수 게이트로 사용하는가
4. 불확실성 높은 구간에서 자동 보수 정책이 발동되는가
5. 액션별 사후 성능(`BUY_NOW`, `WAIT_*`, `HOLD`, `SKIP`)을 분리 모니터링하는가

## 9. 결론

당신의 방향(데이터 최적화 중심에서 월드모델 기반 예측/계획 중심으로 전환)은 타당합니다.  
정확한 표현은 다음입니다.

- "데이터 기반 의사결정"을 넘어서는 것이 아니라
- "데이터를 이용한 세계모사 + 계획 기반 의사결정"으로 진화한다.

GPU Advisor의 현재 구조는 이미 이 방향(`h/g/f + MCTS`) 위에 있으므로,
앞으로의 핵심 과제는 모델 구조 변경보다 `데이터 품질`, `계획 KPI`, `운영 안전정책`의 체계화입니다.

## 10. 강화학습 기초 — GPU Advisor 관점에서

### 10.1 강화학습이란

강화학습(Reinforcement Learning, RL)은 **에이전트(agent)가 환경(environment)과 상호작용하면서 보상(reward)을 최대화하는 정책(policy)을 학습**하는 방법론입니다.

```
에이전트: GPU 구매 의사결정 AI
환경:     한국 GPU 시장 (다나와 가격, 환율, 뉴스)
상태:     256D 시장 벡터 (가격·환율·뉴스·기술지표)
행동:     BUY_NOW / WAIT_SHORT / WAIT_LONG / HOLD / SKIP
보상:     다음 날 가격 변화율 기반 수익률
```

지도학습(Supervised Learning)이 "정답 레이블"을 필요로 하는 반면, 강화학습은 **행동의 결과로 얻는 보상**으로부터 학습합니다. GPU 시장에서 "최적 구매 타이밍"의 정답 레이블은 존재하지 않으므로, RL 접근이 적합합니다.

### 10.2 MDP 정식화

GPU 구매 문제를 마르코프 결정 과정(Markov Decision Process)으로 정식화:

| MDP 구성요소 | GPU Advisor 매핑 |
|-------------|----------------|
| 상태 S | 256D 시장 상태 벡터 (가격, 환율, 뉴스, 기술지표) |
| 행동 A | {BUY_NOW, WAIT_SHORT, WAIT_LONG, HOLD, SKIP} |
| 전이 T(s,a,s') | Dynamics Network g(s,a) → s' (학습된 시장 시뮬레이터) |
| 보상 R(s,a) | pct_change(가격) × 행동 방향 부호 |
| 할인율 γ | 0.99 (미래 보상의 현재 가치 할인) |
| 정책 π(a|s) | MCTS 탐색 결과 (4-신호 블렌드) |

**핵심 통찰**: 전통적 RL에서는 환경 모델이 주어지지만, GPU 시장은 모델이 없습니다. 이 프로젝트는 **World Model(Dynamics Network g)을 함께 학습**하여 이 문제를 해결합니다. 이것이 MuZero 스타일 접근의 핵심입니다.

### 10.3 모델 기반 RL vs 모델 프리 RL

| 구분 | 모델 프리 (Model-Free) | 모델 기반 (Model-Based) |
|------|----------------------|----------------------|
| 대표 알고리즘 | Q-Learning, PPO, SAC | MuZero, AlphaZero, Dreamer |
| 환경 모델 | 불필요 | 함께 학습 |
| 데이터 효율성 | 낮음 (많은 샘플 필요) | 높음 (시뮬레이션으로 보완) |
| 계획 능력 | 없음 (즉흥적 결정) | 있음 (미래 시나리오 탐색) |
| GPU Advisor 선택 이유 | - | 데이터 희소 + 멀티스텝 계획 필요 |

GPU 시장은 하루 24개 GPU × 30일 = 720개 샘플이라는 극단적 데이터 희소성을 가집니다. 모델 기반 RL은 World Model로 시뮬레이션 데이터를 생성하여 이 한계를 보완합니다.

### 10.4 AlphaZero vs MuZero vs GPU Advisor

| 특성 | AlphaZero | MuZero | GPU Advisor |
|------|-----------|--------|-------------|
| 환경 시뮬레이터 | 완벽 제공 (게임 규칙) | 학습 (World Model) | 학습 (Dynamics Network g) |
| 자기 대전 | ✅ | ✅ | ❌ (실제 시장 데이터) |
| 보상 함수 | 명확 (승/패) | 환경에서 수집 | 가격 변화율로 근사 |
| MCTS | ✅ | ✅ | ✅ (50 시뮬레이션, 5 rollout) |
| 핵심 차이 | h+g+f+MCTS | h+g+f+MCTS | **h+g+f+a+MCTS** (Action Model 추가) |

GPU Advisor는 MuZero 구조에 **Action Model(a)**을 추가하여 행동 사전 분포를 별도로 학습합니다. 이는 데이터가 극히 희소한 실세계 도메인에서 MCTS 탐색의 수렴을 돕는 역할을 합니다.

### 10.5 손실 함수 — 5개 목표의 동시 최적화

Fine-tuner는 5개의 손실 함수를 동시에 최적화합니다:

```python
total_loss = (
    1.0 * latent_loss         # World Model: 상태 전이 일관성
  + 1.0 * policy_loss         # f(s): MCTS 탐색 결과를 타겟으로
  + 1.0 * value_loss          # f(s): 기댓값 예측 정확도
  + 0.5 * reward_nll_loss     # g(s,a): Gaussian 보상 예측 (μ, σ²)
  + 0.3 * action_prior_loss   # a(s): 실제 가격 레이블 기반 행동 학습
)
```

각 손실이 학습하는 것:
- `latent_loss`: World Model이 실제 시장 전이를 얼마나 잘 시뮬레이션하는가
- `policy_loss`: f(s)가 MCTS가 찾은 최적 분포를 닮아가도록 (점차 MCTS 필요성 감소)
- `value_loss`: f(s)가 미래 기댓값을 정확히 추정하도록
- `reward_nll_loss`: g(s,a)가 보상의 평균과 불확실성(σ²)을 동시에 학습하도록
- `action_prior_loss`: a(s)가 실제 가격 데이터에서 도출된 행동 레이블을 학습하도록

### 10.6 30일 실데이터 학습 결과 (2026-03-22)

500 스텝 학습 후 달성된 성능:

| 지표 | 값 | 의미 |
|------|-----|------|
| win_rate | **75%** | 에피소드의 75%에서 양의 보상 달성 |
| 방향정확도 | **89.4%** | BUY vs WAIT 방향 예측 정확도 |
| 평균 보상 | **+0.0064** | always_buy 전략 대비 uplift: +0.0040 |
| action_entropy | **1.459** | 다양한 행동 탐색 (모드 붕괴 없음) |
| 관망비율 | **78.8%** | 불확실 구간에서 보수적 HOLD 선택 |
| 게이트 | **7/7 PASS** | 모든 품질 기준 통과 |

**해석**: 방향정확도 89.4%는 "지금 사는 것이 이득인가 손해인가"를 10번 중 9번 올바르게 판단함을 의미합니다. 관망비율 78.8%는 에이전트가 불확실한 상황에서 과감한 행동을 자제하고 보수적으로 운영됨을 보여줍니다.

## 11. World Model 심층 분석 — Dynamics Network g(s,a)

### 11.1 시장 물리학으로서의 World Model

World Model은 "GPU 시장이 어떻게 작동하는가"를 신경망으로 학습합니다.

```
입력:  현재 시장 latent state s_t (256D) + 행동 a_t (one-hot 5D)
출력:  다음 시장 상태 s_{t+1} (256D)
      보상 기댓값 μ (scalar)
      보상 불확실성 log σ² (scalar)
```

이 과정에서 신경망은 다음을 내재화합니다:
- "BUY_NOW를 선택하면 시장 상태가 어떻게 변하는가"
- "WAIT_SHORT 후 가격이 떨어질 확률과 그 폭의 분포"
- "환율 상승 구간에서 HOLD가 초래하는 기댓값 변화"

### 11.2 Gaussian NLL — 불확실성의 명시적 모델링

단순 MSE 손실 대신 **가우시안 음의 로그우도(Gaussian NLL)** 를 사용합니다:

```python
# MSE (점추정만):  loss = (r - μ)²
# Gaussian NLL (불확실성 포함):
reward_nll_loss = 0.5 * ((reward - mu)² * exp(-logvar) + logvar)
```

- `μ (mu)`: 예측 보상의 기댓값
- `exp(logvar) = σ²`: 예측의 불확실성 (분산)

**왜 중요한가**: GPU 시장은 본질적으로 예측 불가능한 이벤트(신제품 출시, 재고 쇼크)가 존재합니다. 불확실성이 큰 상황을 σ²로 표현하면, MCTS 탐색 중 에이전트가 암묵적으로 리스크를 고려할 수 있습니다.

### 11.3 잔차 연결 (Residual Connection) — 학습 안정성

모든 네트워크 블록에 `x = x + block(x)` 형태의 잔차 연결이 적용됩니다:

```
입력 x (256D)
    │
    ├──────────────────────────┐
    │                          │
    ▼                          │
Linear(256→256)               │
GELU                          │
Linear(256→256)               │
    │                          │
    ▼                          │
출력 + ────────────────────────┘ (잔차 합산)
```

잔차 연결의 역할:
1. **그래디언트 소실 방지**: 깊은 네트워크에서 그래디언트가 0으로 사라지는 현상 억제
2. **항등 사상 기본값**: 네트워크가 "아무것도 하지 않음"을 기본값으로 학습 시작
3. **표현 재사용**: 하위 레이어의 풍부한 표현을 상위 레이어가 직접 활용

## 12. 운영 아키텍처 전체도 (2026-03-22 기준)

```
[데이터 수집 레이어]
LaunchAgent (매일 00:00)
  ├── danawa_crawler.py    → data/raw/danawa/YYYY-MM-DD.json (24 GPU 모델)
  ├── exchange_rate.py     → data/raw/exchange/YYYY-MM-DD.json (USD/JPY/EUR)
  ├── news_crawler.py      → data/raw/news/YYYY-MM-DD.json (감정 분석 포함)
  └── feature_engineer.py → data/processed/dataset/training_data_YYYY-MM-DD.json

[학습 레이어]
auto_training.py → decide_auto_training_action()
  ├── < 30일: release dry-check
  ├── = 30일 최초: train_release (500 steps)  ✅ 2026-03-22 실행 완료
  └── 매 7일 누적: retrain (재학습)

[AI 모델 레이어]
  h(s): RepresentationNetwork  256D → 256D latent (6.4M params)
  g(s,a): DynamicsNetwork      256D+5D → 256D + μ,σ² (6.5M params)
  f(s): PredictionNetwork      256D → policy(5D) + value(1D) (6.0M params)
  a(s): ActionModel            256D → 5D prior (43K params)
  MCTS: 50 simulations × 5 rollout steps

[추론 레이어]
GPUPurchaseAgent.decide_from_state()
  policy = 0.45×MCTS + 0.25×Reward + 0.15×f-prior + 0.15×ActionModel
  safe_mode: confidence < 0.25 → HOLD / entropy > 1.58 → HOLD

[서비스 레이어]
FastAPI (backend/simple_server.py) → localhost:8000
Next.js 프론트엔드             → localhost:3000
릴리즈 태그: release-agent-20260322-105138
```
