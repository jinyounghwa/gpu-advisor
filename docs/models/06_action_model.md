# Action Model (`a`) 완전 해부

> **파일 위치:** `backend/models/action_model.py`
> **역할:** 행동 의미 임베딩 + 맥락 인식 행동 사전 분포 학습
> **입력 출처:** `RepresentationNetwork(h)` 가 생성한 Latent State (256D)
> **출력 전달:** `GPUPurchaseAgent.decide_from_state()`의 policy calibration 단계
> **체크포인트 키:** `a_state_dict` (없으면 하드코딩 휴리스틱으로 폴백)

---

## 1. 왜 Action Model이 필요한가?

### 1.1 기존 아키텍처의 한계

기존 h/g/f 구조에서 행동은 두 가지 방식으로만 처리된다:

```
기존 구조:
  Dynamics Network g(s, a):
    a_t = [0, 0, 0, 0, 1]  # 5D one-hot (BUY_NOW, WAIT_SHORT, WAIT_LONG, HOLD, SKIP)
    input = cat([s_t, a_t])  # 256 + 5 = 261D 단순 결합

  GPUPurchaseAgent.decide_from_state():
    # 하드코딩된 휴리스틱 (utility_bias)
    utility_bias[0] = +1.0 * trend_down - 1.2 * over_ma - 0.7 * trend_up  # BUY_NOW
    utility_bias[1] = +0.6 * over_ma + 0.5 * trend_up                      # WAIT_SHORT
    ...
    util_policy = softmax(utility_bias / 0.8)
```

**문제점:**
1. **One-hot의 의미 손실**: BUY_NOW(0)와 WAIT_SHORT(1)은 개념적으로 가깝지만, 유클리드 거리상 동일하게 취급됨 (`|[1,0,0,0,0] - [0,1,0,0,0]| = |[1,0,0,0,0] - [0,0,0,0,1]|`)
2. **비학습 휴리스틱**: `utility_bias`는 도메인 전문가가 수작업으로 설계한 고정값으로, 시장 조건이 바뀌어도 적응하지 못함
3. **관심사 분리 없음**: 행동 합리성 판단 로직이 에이전트 계층에 산재

### 1.2 Action Model의 역할 (MuZero 관점)

| 구성요소 | 역할 |
|---------|------|
| `h(s)` | 관측 → Latent State |
| `g(s, a)` | World Model: 상태 전이 + 보상 예측 |
| `f(s)` | Policy + Value: 행동 확률 + 상태 가치 |
| **`a(s)`** | **Action Model: 맥락 인식 행동 사전 분포** |

`a(s)`는 `f(s)`의 policy head와 별개로, **더 가벼운 전용 네트워크**로 행동 합리성을 판단한다. 학습 신호도 다르다: `f(s)`는 MCTS 탐색 결과를 타겟으로 하지만, `a(s)`는 실제 가격 데이터에서 도출된 행동 레이블로 직접 학습된다.

---

## 2. 전체 데이터 흐름

```
[현재 Latent State s_t]           [행동 인덱스 a_t]
       (batch, 256)                   (batch,) int
            │                              │
            │              ┌──────────────►│
            │              │        ActionEmbeddingLayer
            │              │          nn.Embedding(5, 16)
            │              │          LayerNorm(16)
            │              │              │
            │              │        (batch, 16)  ← 학습된 의미 임베딩
            │              └──────────────►│
            │
            ▼
   ActionPriorNetwork
   ┌──────────────────────────┐
   │  Linear(256 → 128)       │
   │  GELU + LayerNorm(128)   │
   │  Linear(128 → 64)        │
   │  GELU                    │
   │  Linear(64 → 5)          │
   └──────────────────────────┘
            │
     (batch, 5) logits
            │
         Softmax
            │
     (batch, 5) 행동 사전 확률
            │
            ▼
   GPUPurchaseAgent.decide_from_state()
     calibrated = 0.60 * mcts_probs      ← MCTS 가중치 상향 (0.45→0.60, 2026-04-03)
                + 0.20 * reward_policy
                + 0.10 * prior_policy
                + 0.10 * util_policy  ← a(s)의 출력이 여기에 사용됨
```

---

## 3. 모듈 구조

### 3.1 `ActionEmbeddingLayer` — 행동 의미 임베딩

```python
class ActionEmbeddingLayer(nn.Module):
    def __init__(self, num_actions=5, embed_dim=16):
        self.embedding = nn.Embedding(5, 16)   # 학습 가능한 임베딩 테이블
        self.layer_norm = nn.LayerNorm(16)

    def forward(self, action_indices):
        # action_indices: (batch,) 정수
        # 반환: (batch, 16) 임베딩 벡터
        return self.layer_norm(self.embedding(action_indices))

    def embed_onehot(self, action_onehot):
        # one-hot (batch, 5) → 임베딩 (batch, 16)
        # Dynamics Network 연동 시 사용 (선택적)
        return self.layer_norm(action_onehot @ self.embedding.weight)
```

**One-hot vs 학습 임베딩의 차이:**

```
One-hot (5D):
  BUY_NOW    = [1, 0, 0, 0, 0]
  WAIT_SHORT = [0, 1, 0, 0, 0]  → BUY_NOW과의 거리: √2
  SKIP       = [0, 0, 0, 0, 1]  → BUY_NOW과의 거리: √2  (동일!)

학습 임베딩 (16D, 학습 후 예상):
  BUY_NOW    ≈ [ 0.8, -0.2,  0.5, ...]  # 매수 관련 방향
  WAIT_SHORT ≈ [ 0.6, -0.1,  0.4, ...]  # BUY_NOW과 유사
  SKIP       ≈ [-0.7,  0.8, -0.3, ...]  # BUY_NOW과 반대 방향
```

→ 학습된 임베딩에서는 행동 간 의미적 유사도가 벡터 거리에 반영된다.

---

### 3.2 `ActionPriorNetwork` — 맥락 인식 행동 사전 분포

```python
class ActionPriorNetwork(nn.Module):
    def __init__(self, latent_dim=256, num_actions=5):
        self.net = nn.Sequential(
            nn.Linear(256, 128),    # 256 → 128
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),    # 128 → 64
            nn.GELU(),
            nn.Linear(64, 5),      # 64 → 5 (행동 로짓)
        )

    def forward(self, latent_state):
        # → (batch, 5) 로짓
        return self.net(latent_state)

    def prior_probs(self, latent_state):
        # → (batch, 5) 확률 분포 (합=1)
        return torch.softmax(self.forward(latent_state), dim=-1)
```

**이 네트워크가 배우는 것:**

기존 하드코딩 휴리스틱의 경우:
```python
# 고정된 규칙 (market_condition에 상관없이 동일)
utility_bias[BUY_NOW] = +1.0 * trend_down - 1.2 * over_ma - 0.7 * trend_up
```

ActionPriorNetwork의 경우:
```
"RTX 4090 가격이 MA7 대비 15% 하락 + 환율 불안정" →
    BUY_NOW:    0.52  (하락세이나 환율 리스크)
    WAIT_SHORT: 0.28  (단기 관망 권장)
    WAIT_LONG:  0.10
    HOLD:       0.08
    SKIP:       0.02

"RTX 3080 가격이 MA14 대비 8% 상승 + 신제품 발표 임박" →
    BUY_NOW:    0.08
    WAIT_SHORT: 0.12
    WAIT_LONG:  0.35
    HOLD:       0.25
    SKIP:       0.20  (신제품 대기 권장)
```

→ 동일한 가격 패턴이라도 다른 맥락(환율, 신제품 등이 latent state에 인코딩)에서 다른 prior를 출력한다.

---

### 3.3 `ActionModel` — 통합 모듈

```python
class ActionModel(nn.Module):
    def __init__(self, num_actions=5, embed_dim=16, latent_dim=256):
        self.embedding = ActionEmbeddingLayer(num_actions, embed_dim)
        self.prior_net = ActionPriorNetwork(latent_dim, num_actions)

    def embed(self, action_indices):
        """정수 인덱스 → 16D 임베딩"""

    def embed_onehot(self, action_onehot):
        """One-hot → 16D 임베딩 (Dynamics Network 연동용)"""

    def get_prior(self, latent_state):
        """현재 시장 상태에서의 행동 사전 확률"""

    def forward(self, latent_state):
        """→ (logits, probs) 튜플"""
```

---

## 4. 학습 방법 (`fine_tuner.py`)

### 4.1 학습 신호

ActionModel은 실제 가격 데이터에서 도출된 행동 레이블로 지도학습된다:

```python
# fine_tuner.py: _action_from_delta()
def _action_from_delta(pct_change):
    if pct_change >= 0.02:   return 0  # BUY_NOW  (지금 샀으면 +2% 이득)
    if pct_change <= -0.05:  return 4  # SKIP     (지금 샀으면 -5% 손실)
    if pct_change <= -0.02:  return 2  # WAIT_LONG
    if pct_change <= -0.005: return 1  # WAIT_SHORT
    return 3                           # HOLD
```

### 4.2 손실 함수

```python
# _train_step()에서 ActionModel 학습
action_prior_logits, _ = self.a(latent.detach())  # 그래디언트 단절
action_prior_loss = F.cross_entropy(action_prior_logits, actions, weight=class_weights)

total_loss = (
    policy_loss         # f(s) 정책 손실
    + value_loss        # f(s) 가치 손실
    + dynamics_loss     # g(s,a) 상태 전이 손실
    + reward_loss       # g(s,a) 보상 예측 손실
    - 0.001 * entropy   # 정책 엔트로피 정규화
    + 0.02 * prior_reg  # KL 정규화
    + 0.3 * action_prior_loss  # ActionModel 손실
)
```

**왜 `latent.detach()`를 사용하는가?**
ActionModel의 학습 신호가 Representation Network의 그래디언트를 오염시키지 않도록 한다. ActionModel은 latent state의 **소비자**이지 **생산자**가 아니기 때문이다.

**왜 가중치 0.3인가?**
- policy_loss, value_loss 등과 동일한 스케일 유지
- ActionModel이 너무 지배적이지 않도록 (전체 손실의 ~15% 기여)

---

## 5. 추론 통합 (`gpu_purchase_agent.py`)

```python
# decide_from_state() 내부

# ActionModel 사용 (체크포인트에 a_state_dict가 있는 경우)
if self.action_model is not None:
    with torch.no_grad():
        latent_tensor = torch.tensor(latent, dtype=torch.float32, device=self.device)
        util_policy = self.action_model.get_prior(latent_tensor).cpu().numpy()
else:
    # 폴백: 가시적 피처 기반 하드코딩 휴리스틱
    utility_bias = ...
    util_policy = softmax(utility_bias / 0.8)

# Policy calibration (4-way blend) — 2026-04-03 튜닝 적용
calibrated = (
    0.60 * mcts_probs      # MCTS 탐색 결과 (↑ 0.45→0.60)
    + 0.20 * reward_policy  # 기댓값 기반 정책 (↓ 0.25→0.20)
    + 0.10 * prior_policy   # 경험적 행동 분포 (↓ 0.15→0.10)
    + 0.10 * util_policy    # ← ActionModel 사전 분포 (↓ 0.15→0.10)
)
```

### Backward Compatibility

| 체크포인트 | 동작 |
|----------|------|
| `a_state_dict` 없음 (구버전) | 하드코딩 utility_bias 폴백 |
| `a_state_dict` 있음 (신버전) | ActionModel prior 사용 |

---

## 6. 체크포인트 구조

```python
# save_checkpoint() 저장 형식 (fine_tuner.py)
payload = {
    "h_state_dict": self.h.state_dict(),   # Representation Network
    "g_state_dict": self.g.state_dict(),   # Dynamics Network
    "f_state_dict": self.f.state_dict(),   # Prediction Network
    "a_state_dict": self.a.state_dict(),   # Action Model (신규)
    "meta": {
        "schema_version": "agent-v1",
        ...
    }
}
```

---

## 7. 파라미터 수

| 모듈 | 파라미터 수 |
|------|------------|
| ActionEmbeddingLayer | 5×16 + 16 ≈ **96** |
| ActionPriorNetwork | 256×128 + 128 + 128×64 + 64 + 64×5 + 5 ≈ **41,477** |
| **ActionModel 합계** | **~43K** |
| 기존 h/g/f 합계 | ~18.9M |
| 전체 모델 | ~18.94M (+0.23%) |

→ 전체 파라미터 대비 0.23% 증가로 행동 추론 능력을 크게 강화한다.

---

## 8. 실행 테스트

```python
from models.action_model import ActionModel
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ActionModel(num_actions=5, embed_dim=16, latent_dim=256).to(device)

# 행동 임베딩
action_indices = torch.tensor([0, 1, 3, 4], device=device)
embeddings = model.embed(action_indices)
print(f"Embeddings: {embeddings.shape}")  # (4, 16)

# 맥락 인식 행동 사전 확률
latent = torch.randn(4, 256, device=device)
probs = model.get_prior(latent)
print(f"Prior probs: {probs.shape}")         # (4, 5)
print(f"Sum = 1: {probs.sum(dim=-1)}")       # [1., 1., 1., 1.]
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")  # ~43,573
```

---

## 9. 아키텍처 전체 구성 요약

```
GPU Purchase Agent 전체 아키텍처 (업데이트)

Raw Market State (256D)
        │
        ▼
  ┌─────────────┐
  │ h(s)        │  Representation Network
  │ 256D → 256D │  관측 → Latent Space
  └─────────────┘
        │ s_t (256D latent state)
        ├────────────────────────────────────────┐
        │                                        │
        ▼                                        ▼
  ┌─────────────┐                        ┌─────────────┐
  │ g(s, a)     │  Dynamics Network      │ f(s)        │  Prediction Network
  │ 256+5 → 256 │  World Model           │ 256 → 5, 1  │  Policy + Value
  └─────────────┘                        └─────────────┘
        │                                        │
        ▼                                        │
   s_{t+1}, r_μ, r_σ²                           │
   (MCTS 시뮬레이션 입력)                         │
                                                 │
        ┌────────────────────────────────────────┘
        │
        ▼
  ┌─────────────┐
  │ a(s)        │  Action Model  ◄── 신규 추가
  │ 256D → 5D   │  행동 사전 분포
  └─────────────┘
        │
        ▼
  Policy Calibration (4-way blend)
  = 0.60×MCTS + 0.20×Reward + 0.10×Prior + 0.10×ActionModel
  (2026-04-03 파라미터 튜닝: MCTS 가중치 0.45→0.60 상향)
        │
        ▼
  Final Decision (BUY_NOW / WAIT_SHORT / WAIT_LONG / HOLD / SKIP)
```

## 10. 실데이터 학습 결과

### 10.1 v3.0 학습 결과 (2026-03-22, 30일 데이터)

| 지표 | 값 |
|------|-----|
| 방향정확도 (BUY vs WAIT) | **89.4%** |
| 평균 보상 | **+0.0064** |
| action_entropy | **1.459** |
| 관망비율 | **78.8%** |
| 게이트 | **7/7 PASS** |

### 10.2 v3.1 학습 결과 (2026-04-03, 42일 룩백, 적극 파라미터 튜닝)

파라미터 변경: 2000 스텝, MCTS 60%, 엔트로피 임계 0.45, abstain 게이트 ≤0.93

| 지표 | 값 |
|------|-----|
| 샘플 수 (302 롤링 윈도우) | **302** |
| 방향정확도 (BUY vs WAIT) | **89.1%** |
| 평균 보상 | **+0.00172** |
| 관망비율 | **93.38%** |
| 평균 신뢰도 | **0.3728** |
| 게이트 | **6/7 (BLOCKED — abstain 0.38% 초과)** |

### 10.3 행동 분포 비교

```
v3.1 (2026-04-03, 302 샘플, MCTS 60%):
  HOLD:       282개 (93.4%) ← MCTS 가중치 상향으로 보수적 강화
  BUY_NOW:     11개  (3.6%)
  WAIT_LONG:    6개  (2.0%)
  WAIT_SHORT:   3개  (1.0%)

v3.0 비교 (2026-03-22, 631 샘플, MCTS 45%):
  HOLD:       518개 (82.1%)
  BUY_NOW:     45개  (7.1%)
  WAIT_SHORT:  38개  (6.0%)
  WAIT_LONG:   30개  (4.8%)
```

MCTS 가중치 0.45→0.60 상향 후 에이전트가 불확실 구간에서 더 보수적으로 행동. abstain 비율 78.8%→93.4% 증가는 이 변화의 직접적 결과입니다.

### 10.4 릴리즈 이력

| 버전 | 날짜 | 태그/상태 |
|------|------|---------|
| v3.0 | 2026-03-22 | `release-agent-20260322-105138` (게이트 7/7 PASS) |
| v3.1 | 2026-04-03 | 재학습 완료, BLOCKED (abstain 게이트 미달) |
| 현재 | 2026-04-08 | 다음 자동 재학습 대기 중 (2일 후) |
