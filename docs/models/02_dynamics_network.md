# Dynamics Network (`g`) 완전 해부

> **파일 위치:** `backend/models/dynamics_network.py`
> **역할:** 현재 상태 + 행동 → 미래 상태 + 보상을 예측하는 **월드 모델(World Model)**
> **입력 출처:** `RepresentationNetwork(h)` 가 생성한 Latent State
> **출력 전달:** `MCTS 엔진`이 미래 시나리오를 시뮬레이션할 때 반복 호출됨
> **운영 기준 참고:** 행동 라벨은 에이전트 계층(`backend/agent/gpu_purchase_agent.py`)에서 관리되며, Dynamics는 `action_dim` 크기의 벡터만 처리합니다.

---

## 1. 전체 데이터 흐름도

```
[현재 Latent State s_t] + [Action a_t]
         ↓                      ↓
   (batch, 256)          (batch, 5) One-hot
         ↓                      ↓
         └────── 결합(cat) ─────┘
                    ↓
              (batch, 261)
                    ↓
┌─────────────────────────────────────┐
│      DynamicsNetwork (g)            │
│                                     │
│  input_layer (261 → 512)            │
│         ↓                           │
│  LayerNorm                          │
│         ↓                           │
│  Transformer Block #1 (512→2048→512)│
│         ↓                           │
│  Transformer Block #2               │
│         ↓                           │
│  Transformer Block #3               │
│         ↓                           │
│  Transformer Block #4               │
│         ↓                           │
│  LayerNorm                          │
│         ↓                           │
│  ┌──────┴──────┐                    │
│  ↓             ↓                    │
│ next_state   reward_mean            │
│ (512→256)    (512→1)                │
│              reward_logvar          │
│              (512→1)                │
└─────────────────────────────────────┘
         ↓              ↓
   s_{t+1}         예상 보상 μ, σ²
   (batch, 256)    (batch,)  (batch,)
         ↓
  → MCTS에서 다음 노드의 state로 사용됨
  → 다시 g()에 넣어 그 다음 미래도 시뮬레이션 가능
```

---

## 2. Action (행동) 표현

이 네트워크는 **`action_dim` 길이의 행동 벡터**를 입력으로 받습니다.

| 인덱스 | 벡터 예시 | 설명 |
|--------|-----------|------|
| 0~4 | `[1,0,0,0,0]` 등 | 인덱스별 의미는 에이전트/정책 계층에서 정의 |

**사용 예시:**
```python
# 특정 행동(예: index=4)을 One-hot으로 인코딩
action = torch.zeros(1, 5)
action[0, 4] = 1.0  # index 4
```

---

## 3. 클래스 코드 상세 분석

### 3.1 `__init__()` — 네트워크 구조 정의

```python
class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,     # Latent State 차원
        action_dim: int = 5,       # Action 종류 수
        hidden_dim: int = 512,     # 내부 처리 차원
        num_layers: int = 4,       # Transformer 블록 수
        dropout: float = 0.1,      # 드롭아웃 비율
    ):
        super().__init__()

        # ① 입력 차원: Latent State(256) + Action(5) = 261
        input_dim = latent_dim + action_dim

        # ② 입력 처리 레이어: 261 → 512
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        # ③ 4개의 Transformer 스타일 블록
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),     # 512 → 2048 (확장)
                nn.GELU(),                                  # 활성화 함수
                nn.Dropout(dropout),                        # 과적합 방지
                nn.Linear(4 * hidden_dim, hidden_dim),     # 2048 → 512 (축소)
                nn.LayerNorm(hidden_dim),                   # 정규화
            )
            for _ in range(num_layers)  # 이 블록을 4번 반복
        ])

        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # ④ 출력 헤드 3개
        self.next_state_head = nn.Linear(hidden_dim, latent_dim)  # 다음 상태
        self.reward_mean_head = nn.Linear(hidden_dim, 1)           # 보상 평균 μ
        self.reward_logvar_head = nn.Linear(hidden_dim, 1)         # 보상 분산 log(σ²)
```

**왜 GELU를 사용하는가? (ReLU 대신)**
```
ReLU:  x < 0 이면 무조건 0
GELU:  x < 0 이면 부드럽게 감소 (완전히 0은 아님)
```
→ GELU는 음수 근처에서 더 부드러운 그래디언트를 제공하여, 학습이 더 안정적이고 정밀합니다. 최신 Transformer 모델(GPT, BERT)에서 표준으로 사용됩니다.

**왜 `4 * hidden_dim`으로 확장하는가?**
- `512 → 2048 → 512` 패턴은 Transformer의 표준 Feed-Forward 구조입니다.
- 4배 넓은 공간에서 데이터를 처리하면 더 복잡한 패턴을 학습할 수 있습니다.

---

### 3.2 `forward()` — 미래를 예측하는 핵심 로직

```python
def forward(self, s_t: torch.Tensor, a_t: torch.Tensor):
    batch_size = s_t.size(0)

    # ① 현재 상태(256D)와 행동(5D)을 하나로 결합
    x = torch.cat([s_t, a_t], dim=-1)  # (batch, 256+5) = (batch, 261)
# 예시: s_t=[0.12, -0.34, ...256개] + a_t=[0,0,0,0,1] → 261차원 벡터(256+5)

    # ② 입력 처리: 261차원 → 512차원으로 확장
    x = self.input_layer(x)            # (batch, 512)
    x = self.layer_norm1(x)            # 값의 분포를 표준화

    # ③ 4개의 Transformer 블록을 순차적으로 통과
    for block in self.blocks:
        x = block(x)
    # 각 블록에서:
    #   512 → 2048 (확장) → GELU → Dropout → 2048 → 512 (축소) → LayerNorm

    x = self.layer_norm2(x)            # 최종 정규화

    # ④ 3개의 출력 헤드로 분기
    s_tp1 = self.next_state_head(x)                     # 다음 Latent State (batch, 256)
    reward_mean = self.reward_mean_head(x).squeeze(-1)  # 보상 평균 μ (batch,)
    reward_logvar = self.reward_logvar_head(x).squeeze(-1)  # 로그 분산

    # ⑤ Softplus로 분산 안정화 (항상 양수 보장)
    reward_logvar = F.softplus(reward_logvar, beta=1.0)
    # softplus(x) = log(1 + e^x) → 항상 양수, 부드러운 곡선

    return s_tp1, reward_mean, reward_logvar
```

**`torch.cat`이 하는 일:**
```python
s_t = [0.12, -0.34, 0.56, ...]  # 256개 값 (현재 시장 상태)
a_t = [0, 0, 0, 0, 1]           # 5개 값 (HOLD 행동)
# cat 결과:
x   = [0.12, -0.34, 0.56, ..., 0, 0, 0, 0, 1]  # 261개 값
```
→ "현재 이 시장 상태에서 특정 행동을 선택했을 때"라는 맥락을 하나의 벡터로 만듭니다.

---

## 4. 출력 해석

### 4.1 `s_tp1` (다음 상태)
```python
# 현재 상태에서 특정 행동을 하면 미래 시장은 이렇게 된다
s_tp1.shape  # (batch, 256) — 미래의 Latent State
# 이 값은 다시 g()에 넣어 더 먼 미래를 예측할 수 있음
```

### 4.2 `reward_mean` / `reward_logvar` (보상 분포)
```python
# 이 행동으로 인한 예상 수익률
reward_mean = 0.052   # → +5.2% 이득 예상
reward_logvar = 0.01  # → 불확실성 낮음 (자신감 높음)

# 불확실성이 크면:
reward_mean = 0.052   # → +5.2% 이득 예상
reward_logvar = 0.50  # → 확실하지 않음 (변동성 큼)
```

**왜 평균(μ)과 분산(σ²) 둘 다 출력하는가?**
- 단순히 "5% 이득"이라고만 하면 얼마나 확실한 예측인지 알 수 없습니다.
- 분산이 작으면 → "거의 확실하게 5% 이득"
- 분산이 크면 → "5% 이득일 수도 있지만 -3%가 될 수도 있음"
→ MCTS에서 리스크를 고려한 의사결정이 가능해집니다.

---

## 5. MCTS에서 반복 호출되는 과정

```python
# MCTS 시뮬레이션 과정에서 g()가 반복 호출됨 (운영 경로: `mcts_engine.py`)

# Step 1: "지금 action index=1을 하면?"
action_buy = torch.tensor([[0,1,0,0,0]])     # index 1
s_1, r_1, _ = g(current_state, action_buy)   # → 1주 후 상태 + 보상

# Step 2: "그 상태에서 action index=4를 하면?"
action_hold = torch.tensor([[0,0,0,0,1]])     # index 4
s_2, r_2, _ = g(s_1, action_hold)            # → 2주 후 상태 + 보상

# Step 3: "그 상태에서 또 HOLD를 하면?"
s_3, r_3, _ = g(s_2, action_hold)            # → 3주 후 상태 + 보상

# 총 보상 = r_1 + 0.99*r_2 + 0.99²*r_3 + ...
# (discount_factor=0.99로 먼 미래일수록 비중 감소)
```
→ 이렇게 **연쇄적으로 호출**하여 5단계(rollout_steps=5) 앞의 미래까지 "상상"합니다.

---

## 6. 실행 테스트 및 파라미터

```python
model = DynamicsNetwork().to(device)

s_t = torch.randn(4, 256).to(device)       # 4개 GPU의 현재 상태
a_t = torch.zeros(4, 5).to(device)          # 4개 모두 action index=0 선택
a_t[:, 0] = 1.0

s_tp1, reward_mean, reward_logvar = model(s_t, a_t)
```

**출력 결과:**
```
s_t shape:         (4, 256)   ← 입력: 현재 상태
s_{t+1} shape:     (4, 256)   ← 출력: 미래 상태
Reward mean:       tensor([ 0.0234, -0.0156,  0.0089,  0.0412])  ← 예상 보상
Reward std:        tensor([ 0.6931,  0.7123,  0.6845,  0.7001])  ← 불확실성
Total parameters:  8,671,490  ← 약 867만 개 파라미터
```

---

## 7. 보조 클래스: `TransitionModel`

```python
class TransitionModel(nn.Module):
    """행동 없이 상태만으로 다음 상태를 예측하는 단순 모델"""
    def __init__(self, latent_dim=256, hidden_dim=512):
        self.transition = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),   # 256 → 512
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   # 512 → 512
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),   # 512 → 256
        )

    def forward(self, s_t):
        return self.transition(s_t)
```
→ 행동과 무관한 **자연적인 시장 흐름**을 학습하기 위한 사전(prior) 모델입니다. DynamicsNetwork가 "행동의 영향"에 집중할 수 있도록 기본적인 시장 변동을 미리 학습해둡니다.
