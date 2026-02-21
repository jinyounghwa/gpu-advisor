# Prediction Network (`f`) 완전 해부

> **파일 위치:** `backend/models/prediction_network.py`
> **역할:** 현재 상태를 보고 즉시 "최선의 행동(Policy)"과 "상황의 유리함(Value)"을 판단하는 직관 네트워크
> **입력 출처:** `RepresentationNetwork(h)` 또는 `DynamicsNetwork(g)` 가 생성한 Latent State
> **출력 전달:** `MCTS 엔진`에서 탐색 방향을 결정하고, 리프 노드의 가치를 평가하는 데 사용됨
> **운영 기준 참고:** 실제 운영 호출 경로는 `backend/agent/gpu_purchase_agent.py` + `backend/models/mcts_engine.py`이며, 행동 라벨 의미는 에이전트 계층(`ACTION_LABELS`)에서 관리됩니다.

---

## 1. 전체 데이터 흐름도

```
[Latent State s_t]
   (batch, 256)
        ↓
┌───────────────────────────────────────┐
│      PredictionNetwork (f)            │
│                                       │
│  input_layer (256 → 512)              │
│        ↓                              │
│  LayerNorm                            │
│        ↓                              │
│  Transformer Block #1 (512→2048→512)  │
│        ↓                              │
│  Transformer Block #2                 │
│        ↓                              │
│  Transformer Block #3                 │
│        ↓                              │
│  Transformer Block #4                 │
│        ↓                              │
│  LayerNorm                            │
│        ↓                              │
│  ┌─────┴──────┐                       │
│  ↓            ↓                       │
│ Policy      Value                     │
│ Head        Head                      │
│ (512→1024   (512→512                  │
│  →5)         →1→Tanh)                 │
└───────────────────────────────────────┘
     ↓              ↓
 Policy logits   Value score
 (batch, 5)      (batch,)
     ↓              ↓
 softmax 적용후    -1 ~ +1 사이의
 MCTS에서         현재 상태
 탐색 방향         유리함 점수
 결정에 사용
```

---

## 2. 핵심 개념: Dual-Head 구조

이 네트워크의 핵심은 **하나의 몸통(Shared Trunk)**에서 **두 개의 머리(Head)**가 갈라지는 구조입니다.

```
         [Shared Trunk]          ← 공통 특징 추출
             ↓
      ┌──────┴──────┐
      ↓             ↓
  Policy Head    Value Head      ← 각각 다른 목적의 출력
  "뭘 할까?"     "얼마나 좋아?"
```

**왜 두 머리를 따로 두는가?**
- **Policy(정책)**: "5가지 행동 중 어떤 게 가장 좋을까?"에 대한 확률 분포
- **Value(가치)**: "지금 상황이 전체적으로 나한테 유리한 상황인가?"에 대한 점수
- 두 질문은 관련은 있지만 다른 성격이므로 분리합니다.

---

## 3. 코드 상세 분석

### 3.1 `__init__()` — 네트워크 구조 정의

```python
class PredictionNetwork(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,     # 입력 차원 (Latent State 크기)
        hidden_dim: int = 512,     # 내부 처리 차원
        num_layers: int = 4,       # Transformer 블록 수
        action_dim: int = 5,       # 행동 종류 수
        dropout: float = 0.1,      # 드롭아웃 비율
    ):
        super().__init__()

        # ━━━ Shared Trunk (공통 몸통) ━━━

        # 입력 레이어: 256 → 512
        self.input_layer = nn.Linear(latent_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        # 4개의 Transformer 스타일 블록
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),    # 512 → 2048
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden_dim, hidden_dim),    # 2048 → 512
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # ━━━ Policy Head (정책 머리) ━━━
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),  # 512 → 1024 (넓은 분석)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, action_dim),   # 1024 → 5 (행동 수)
        )
        # 출력: raw logits (아직 softmax 적용 전)

        # ━━━ Value Head (가치 머리) ━━━
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),       # 512 → 512
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),                # 512 → 1 (스칼라)
            nn.Tanh(),                                # -1 ~ +1 범위로 제한
        )
```

**Policy Head가 더 넓은(512→1024) 이유:**
- 주석에 "더 많은 탐색 가능하도록 크게"라고 설명되어 있습니다.
- 넓은 레이어를 사용하면 5가지 행동의 미묘한 차이를 더 정밀하게 구분할 수 있습니다.

**Value Head에 `Tanh()`를 쓰는 이유:**
```
Tanh(x)의 출력 범위: -1.0 ~ +1.0

-1.0 = "매우 불리한 상황" (손해가 예상됨)
 0.0 = "중립적 상황"
+1.0 = "매우 유리한 상황" (큰 이득 예상)
```
→ 값의 범위를 제한하여 학습이 안정적으로 진행되게 합니다.

---

### 3.2 `forward()` — 실제 실행 흐름

```python
def forward(self, s_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # ━━━ Shared Trunk ━━━

    # ① 입력 처리: 256차원 → 512차원
    x = self.input_layer(s_t)   # (batch, 256) → (batch, 512)
    x = self.layer_norm1(x)     # 정규화

    # ② 4개의 Transformer 블록 순차 통과
    for block in self.blocks:
        x = block(x)
    # 각 블록: 512 → 2048 → GELU → Dropout → 2048 → 512 → LayerNorm

    x = self.layer_norm2(x)     # 최종 정규화
    # 여기까지가 공통 몸통 — 이후 두 머리로 분기

    # ━━━ Policy Head ━━━
    # ③ 행동 확률 예측
    policy_logits = self.policy_head(x)    # (batch, 5)
    # 출력 예시: [1.23, 3.45, -0.67, 0.12, 2.34]
    # 아직 raw logits이므로 확률이 아님 (softmax 필요)

    # ━━━ Value Head ━━━
    # ④ 상태 가치 예측
    value = self.value_head(x).squeeze(-1)  # (batch, 1) → (batch,)
    # 출력 예시: 0.42 (Tanh이므로 -1~+1 사이)

    return policy_logits, value
```

---

## 4. 출력 해석 상세

### 4.1 Policy Logits → Action Probabilities

```python
# forward()가 반환한 raw logits
policy_logits = torch.tensor([1.23, 3.45, -0.67, 0.12, 2.34])

# softmax를 적용하여 확률로 변환
action_probs = F.softmax(policy_logits, dim=-1)
# 결과: [0.06, 0.56, 0.01, 0.02, 0.18]
#         ↑      ↑     ↑     ↑     ↑
#      ACTION_0 ACTION_1 ACTION_2 ACTION_3 ACTION_4
```

**해석:**
- `ACTION_1 = 56%` → "index 1 행동이 가장 유력"
- `ACTION_4 = 18%` → "index 4 행동도 후보"
- `ACTION_2 = 1%` → "index 2 행동은 우선순위 낮음"

**왜 forward()에서 softmax를 적용하지 않는가?**
- 학습 시 `CrossEntropyLoss`가 내부적으로 softmax를 적용하므로 중복 적용을 방지합니다.
- 추론 시에만 외부에서 `F.softmax()`를 호출합니다.

### 4.2 Value Score

```python
value = 0.42  # Tanh 출력

# 해석:
# +0.42 → "현재 시장 상황은 약간 유리함"
# +0.90 → "매우 유리한 상태 (역대 최저가에 근접)"
# -0.50 → "불리한 상태 (가격이 비쌈)"
```

---

## 5. MCTS에서 어떻게 사용되는가

### 5.1 노드 확장 시 — Prior(사전 확률) 결정

```python
# MCTS 구현의 노드 확장 메서드 예시 (`mcts.py`/`mcts_engine.py` 개념 공통)
def _expand_node(self, node: Node, prediction_model: nn.Module):
    with torch.no_grad():
        # f 네트워크를 호출하여 policy와 value를 얻음
        policy_logits, _ = prediction_model(node.state)
        probs = F.softmax(policy_logits, dim=-1).squeeze(0)
        # probs = [0.06, 0.56, 0.01, 0.02, 0.18]

    # 각 행동에 대한 자식 노드를 생성하고 prior(사전 확률)를 할당
    for a in range(self.action_dim):
        node.children[a] = Node(
            parent=node,
            action=a,
            prior=probs[a].item()  # ← f 네트워크가 알려준 확률
        )
    # 결과:
    # children[0] (ACTION_0) prior=0.06
    # children[1] (ACTION_1) prior=0.56
    # children[2] (ACTION_2) prior=0.01
    # children[3] (ACTION_3) prior=0.02
    # children[4] (ACTION_4) prior=0.18
```
→ prior가 높은 노드가 MCTS에서 더 자주 방문되므로, **f 네트워크의 직관이 탐색 방향을 가이드**합니다.

### 5.2 리프 노드 도달 시 — Value로 평가

```python
# MCTS search() 내부 예시
with torch.no_grad():
    _, value = prediction_model(next_state)
    leaf_value = value.item()
    # leaf_value = 0.42 → "이 노드의 미래 가치는 +0.42"

# 이 값이 backpropagate를 통해 부모 노드들에 전파됨
self._backpropagate(search_path, leaf_value)
```
→ f 네트워크의 value가 **MCTS 트리 전체의 가치 평가**에 핵심 역할을 합니다.

---

## 6. 실행 테스트 및 파라미터

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = PredictionNetwork().to(device)

s_t = torch.randn(4, 256).to(device)
policy_logits, value = model(s_t)

action_probs = F.softmax(policy_logits, dim=-1)
```

**출력 결과:**
```
Policy logits shape: (4, 5)     ← 4개 GPU에 대한 5가지 행동 raw 점수
Value shape:         (4,)       ← 4개 GPU의 상태 가치 점수

Action probabilities (GPU #1):
  ACTION_0: 12.3%
  ACTION_1: 45.6%   ← 최고
  ACTION_2:  3.2%
  ACTION_3:  1.8%
  ACTION_4: 37.1%

Value (GPU #1): +0.34   ← "약간 유리한 상황"

Total parameters: 9,330,182  ← 약 933만 개 파라미터
```

---

## 7. 학습 시 Loss 함수와의 연결

```python
# train_alphazero_v2.py 에서의 학습 코드 (개념)

# MCTS가 실제로 찾아낸 최적 행동 분포 (Ground Truth)
mcts_policy_target = [0.10, 0.60, 0.05, 0.05, 0.20]

# f 네트워크가 예측한 분포
predicted_policy = F.softmax(model(state)[0], dim=-1)
# [0.12, 0.45, 0.03, 0.02, 0.38]

# Policy Loss: 두 분포의 차이를 줄이도록 학습
policy_loss = -sum(target * log(predicted))
# → f 네트워크가 MCTS의 판단을 점점 닮아감

# Value Loss: 실제 이득과 예측 가치의 차이
value_loss = (predicted_value - actual_return) ** 2
# → f 네트워크가 실제 결과를 더 정확히 예측하게 됨
```
