# MCTS 탐색 엔진 완전 해부

> **파일 위치:** `backend/models/mcts.py` (간결 버전) + `backend/models/mcts_engine.py` (상세 버전)
> **역할:** `h`, `g`, `f` 세 네트워크를 조합하여 최적의 행동을 결정하는 **최종 의사결정 엔진**
> **입력:** `RepresentationNetwork(h)` 가 생성한 Root Latent State
> **출력:** GPU 구매 추천 점수 (Action Probability Distribution)
> **운영 기준 참고:** 실제 서비스 경로는 `backend/agent/gpu_purchase_agent.py`에서 `mcts_engine.py`를 사용합니다. `mcts.py`는 간결한 레퍼런스/실험용 구현입니다.

---

## 1. MCTS란 무엇인가?

Monte Carlo Tree Search는 **트리 구조로 가능한 미래를 탐색**하는 알고리즘입니다.

```
현재 시장 상태 (Root)
    ├── [ACTION_0]  →  미래 상태 A
    │       ├── [ACTION_1]  →  미래 A-1
    │       ├── [ACTION_4]  →  미래 A-2  ← 여기가 좋았다!
    │       └── [ACTION_2]  →  미래 A-3
    ├── [ACTION_1]  →  미래 상태 B  ← 가장 많이 방문됨
    │       ├── [ACTION_4]  →  미래 B-1
    │       └── [ACTION_3]  →  미래 B-2
    ├── [ACTION_2]  →  미래 상태 C
    ├── [ACTION_3]  →  미래 상태 D
    └── [ACTION_4]  →  미래 상태 E
```

50번의 시뮬레이션 중 ACTION_1 가지가 28번 방문되었다면 → **ACTION_1 추천 확률 = 28/50 = 56%**

---

## 2. 두 가지 구현체의 차이

| 특성 | `mcts.py` | `mcts_engine.py` |
|------|-----------|-------------------|
| 복잡도 | 간결(172줄) | 상세(343줄) |
| 데이터 타입 | `torch.Tensor` 기반 | `numpy.ndarray` 기반 |
| UCB 공식 | PUCT (AlphaZero 스타일) | 전통 UCB + prior |
| Rollout | 없음 (1단계 확장) | 5단계 롤아웃 |
| Self-play | 미포함 | `MCTSTrainer` 클래스 포함 |
| 주 용도 | 레퍼런스/실험 | 운영 추론 + 학습 데이터 생성 |

---

## 3. `mcts.py` 상세 분석 (레퍼런스/개념용)

### 3.1 데이터 구조: `MCTSConfig` 와 `Node`

```python
@dataclass
class MCTSConfig:
    num_simulations: int = 50       # 시뮬레이션 횟수 (50번 미래를 상상)
    exploration: float = 1.5        # 탐색 vs 이용 균형 계수
    dirichlet_alpha: float = 0.25   # 초기 탐색 노이즈 강도
    rollout_steps: int = 5          # 한 번에 몇 단계 미래까지 볼 것인가
    temperature: float = 1.0        # 행동 선택의 다양성 (높을수록 랜덤)

@dataclass
class Node:
    state: Optional[torch.Tensor] = None   # Latent State (1, 256)
    visits: int = 0                         # 이 노드를 방문한 횟수
    total_reward: float = 0.0               # 누적 보상값
    parent: Optional['Node'] = None         # 부모 노드
    action: Optional[int] = None            # 이 노드에 도달한 행동 (0~4)
    prior: float = 1.0                      # f 네트워크가 알려준 사전 확률
    children: Dict[int, 'Node'] = field(default_factory=dict)  # 자식 노드들

    @property
    def value(self) -> float:
        """평균 보상값 = 누적 보상 / 방문 횟수"""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits
```

**`exploration = 1.5` 가 의미하는 것:**
- 값이 크면 → 아직 방문하지 않은 노드를 더 적극적으로 탐색 (exploration)
- 값이 작으면 → 이미 좋다고 알려진 노드를 더 자주 방문 (exploitation)
- 1.5는 AlphaZero 논문의 권장 범위 내 값입니다.

---

### 3.2 `search()` — 메인 탐색 루프 (가장 중요한 함수)

```python
def search(self, root_state, dynamics_model, prediction_model):
    # ━━━ 1단계: 루트 노드 초기화 ━━━
    root = Node(state=root_state)

    # f 네트워크로 루트의 자식 노드들에 prior 할당
    self._expand_node(root, prediction_model)
    # → root.children[0~4] 에 각각 prior가 설정됨
# 예: children[1].prior = 0.56 (ACTION_1이 유력)

    # ━━━ 2단계: 50번의 시뮬레이션 ━━━
    for _ in range(self.config.num_simulations):  # 50번 반복
        node = root
        search_path = [node]

        # ── Selection (선택 단계) ──
        # 이미 확장된 노드들 사이에서 PUCT 점수가 가장 높은 자식을 따라 내려감
        while node.children and all(
            child.state is not None for child in node.children.values()
        ):
            node = self._select_child(node)  # PUCT 기준 최선의 자식
            search_path.append(node)

        # ── Expansion & Evaluation (확장 단계) ──
        if node.children:
            # 아직 state가 없는 (= 미확장) 자식 중 가장 유력한 것 선택
            best_action = -1
            best_score = -float('inf')
            for a, child in node.children.items():
                if child.state is None:
                    score = child.prior * (
                        math.sqrt(max(1, node.visits)) /
                        (1 + child.visits)
                    )
                    if score > best_score:
                        best_score = score
                        best_action = a

            target_node = node.children[best_action]
            search_path.append(target_node)

            # g 네트워크로 미래 상태 예측
            action_onehot = torch.zeros(1, self.action_dim, device=root_state.device)
            action_onehot[0, best_action] = 1.0

            with torch.no_grad():
                next_state, reward_mean, _ = dynamics_model(
                    node.state, action_onehot
                )
                # → g 네트워크: "이 행동을 하면 시장이 이렇게 된다"

                target_node.state = next_state
                self._expand_node(target_node, prediction_model)
                # → f 네트워크: 새 상태의 자식 노드들에 prior 할당

                _, value = prediction_model(next_state)
                leaf_value = value.item()
                # → f 네트워크: "이 미래 상태는 +0.35 정도 유리해"

        # ── Backpropagation (역전파) ──
        self._backpropagate(search_path, leaf_value)
        # → 지나온 모든 노드의 visits와 total_reward를 업데이트

    # ━━━ 3단계: 최종 결정 ━━━
    # 방문 횟수가 가장 많은 행동이 최선의 행동
    visit_counts = torch.tensor([
        root.children[a].visits for a in range(self.action_dim)
    ], dtype=torch.float32)

    policy_target = visit_counts / visit_counts.sum()
    # 예시: [3, 28, 2, 1, 16] / 50 = [0.06, 0.56, 0.04, 0.02, 0.32]

    best_action = torch.argmax(visit_counts).item()
    # → 1 (ACTION_1)

    return best_action, policy_target
```

---

### 3.3 `_select_child()` — PUCT 알고리즘

```python
def _select_child(self, node: Node) -> Node:
    """PUCT(Predictor + Upper Confidence bound for Trees)"""
    total_sqrt = math.sqrt(node.visits)
    best_score = -float('inf')
    best_child = None

    for action, child in node.children.items():
        # PUCT 공식:
        # Q(s,a) + c * P(s,a) * √(N(s)) / (1 + N(s,a))
        #
        # Q(s,a): 이 행동의 평균 가치 (= child.value)
        # c:      탐색 강도 (= self.config.exploration = 1.5)
        # P(s,a): f 네트워크의 사전 확률 (= child.prior)
        # N(s):   부모 방문 횟수
        # N(s,a): 자식 방문 횟수

        u_score = self.config.exploration * child.prior * (
            total_sqrt / (1 + child.visits)
        )
        score = child.value + u_score

        if score > best_score:
            best_score = score
            best_child = child

    return best_child
```

**PUCT 공식의 직관:**
```
score = child.value + 1.5 * child.prior * √(parent.visits) / (1 + child.visits)
         ↑                  ↑                                      ↑
     "이미 알려진 가치"    "f가 좋다고 한 정도"       "아직 덜 방문했을수록 보너스"

예시 - 시뮬레이션 20번째:
  ACTION_1 (방문 12번, 평균 가치 0.4, prior 0.56):
    score = 0.4 + 1.5 * 0.56 * √20 / (1+12) = 0.4 + 0.29 = 0.69
  
  ACTION_4 (방문 3번, 평균 가치 0.3, prior 0.18):
    score = 0.3 + 1.5 * 0.18 * √20 / (1+3) = 0.3 + 0.30 = 0.60
  
  ACTION_0 (방문 0번, prior 0.06):
    score = 0.0 + 1.5 * 0.06 * √20 / (1+0) = 0.0 + 0.40 = 0.40
```
→ ACTION_1(0.69) 선택. 하지만 방문이 많아지면 u_score가 줄어들어 다른 노드도 탐색하게 됩니다.

---

### 3.4 `_backpropagate()` — 경험의 전파

```python
def _backpropagate(self, search_path: List[Node], value: float):
    """리프 노드의 가치를 루트까지 거슬러 올라가며 전파"""
    for node in reversed(search_path):
        node.visits += 1            # 방문 횟수 +1
        node.total_reward += value  # 보상값 누적
```

```
시뮬레이션 경로: Root → ACTION_1 → ACTION_4
리프 가치: +0.42

전파 과정:
  ACTION_4 노드:  visits 3→4,  total_reward 1.05→1.47
  ACTION_1 노드:  visits 12→13, total_reward 5.20→5.62
  Root 노드:      visits 20→21, total_reward 7.80→8.22
```

---

## 4. `mcts_engine.py` 상세 분석 (학습용)

### 4.1 핵심 차이: `_simulate()` 롤아웃

`mcts.py`가 1단계만 확장하는 반면, `mcts_engine.py`는 **5단계 미래**까지 돌려봅니다:

```python
def _simulate(self, node, policy_network, dynamics_network, device="cpu"):
    total_reward = 0.0
    current_node = node
    discount = 1.0

    for step in range(self.config.rollout_steps):  # 5번 반복
        # ① f 네트워크로 행동 확률 예측
        state_tensor = torch.tensor(current_node.state, ...).to(device)
        with torch.no_grad():
            policy_logits, value = policy_network(state_tensor)
            policy_probs = torch.softmax(policy_logits, dim=-1)

        # ② 확률에 따라 행동 샘플링 (확률적 선택)
        action = torch.multinomial(policy_probs, num_samples=1).item()
        # 예: ACTION_1(56%)이 가장 자주 선택되지만 ACTION_4(32%)도 가능

        # ③ g 네트워크로 미래 상태와 보상 예측
        action_onehot = np.zeros(5)
        action_onehot[action] = 1.0
        with torch.no_grad():
            next_state, reward_mean, _ = dynamics_network(state_tensor, action_tensor)

        # ④ 보상을 할인하여 누적
        reward = reward_mean.squeeze(0).cpu().item()
        total_reward += discount * reward
        discount *= self.config.discount_factor  # 0.99
        # Step 1 보상: reward × 1.00
        # Step 2 보상: reward × 0.99
        # Step 3 보상: reward × 0.98
        # Step 4 보상: reward × 0.97
        # Step 5 보상: reward × 0.96

        # 미래 상태를 현재로 업데이트하고 반복
        current_node = MCTSNode(state=next_state_np)

    # ⑤ 마지막 상태의 가치를 f 네트워크로 평가
    with torch.no_grad():
        _, value = policy_network(final_state_tensor)

    # 총 가치 = 5단계 누적 보상 + 마지막 상태 가치
    return total_reward + discount * value.item()
```

**Discount Factor (0.99)의 의미:**
```
1주 후 받는 100만원의 가치:  100 × 0.99¹ = 99.0만원
1달 후 받는 100만원의 가치:  100 × 0.99⁴ = 96.1만원
3달 후 받는 100만원의 가치:  100 × 0.99¹² = 88.6만원
```
→ "지금의 100만원 > 먼 미래의 100만원" 이라는 시장의 시간 가치를 반영합니다.

---

### 4.2 `MCTSTrainer` — 자가 대전으로 학습 데이터 생성

```python
class MCTSTrainer:
    def generate_episode(self, initial_state: np.ndarray) -> List[Dict]:
        """MCTS를 200번 수행하여 하나의 에피소드 데이터 생성"""
        episode_data = []
        current_state = initial_state
        step = 0

        while step < 200:
            # ① MCTS 탐색 수행 (50번 시뮬레이션)
            action_probs, value, tree = self.mcts.search(
                current_state, self.policy_network,
                self.dynamics_network, self.device
            )
            # action_probs = [0.06, 0.56, 0.04, 0.02, 0.32]

            # ② 확률에 따라 실제 행동 선택
            action = np.random.choice(len(action_probs), p=action_probs)
            # 예: 2 (ACTION_2가 높은 비율로 선택됨)

            # ③ g 네트워크로 다음 상태 생성
            with torch.no_grad():
                next_state, reward, _ = self.dynamics_network(
                    state_tensor, action_tensor
                )

            # ④ 학습 데이터로 저장
            episode_data.append({
                "state": current_state,           # 입력 상태
                "action": action,                  # 선택한 행동
                "action_probs": action_probs,      # MCTS가 찾은 최적 분포
                "reward": reward_val,              # 받은 보상
                "value": value,                    # 상태 가치
                "next_state": next_state_np,       # 결과 상태
            })

            current_state = next_state_np
            step += 1

        return episode_data
        # → 이 데이터로 f, g 네트워크를 업데이트
```

**학습 데이터의 활용:**
```
episode_data[0]:
  state:        [현재 시장 벡터 256D]
  action_probs: [0.06, 0.56, 0.04, 0.02, 0.32]  ← f 네트워크의 학습 목표
  value:        0.42                              ← f 네트워크의 가치 학습 목표
```
→ f 네트워크가 MCTS의 결과를 점점 닮아가도록 학습됩니다.
→ f가 정확해질수록 MCTS는 더 적은 시뮬레이션으로 최적해를 찾게 됩니다.

---

## 5. 전체 실행 시나리오 (End-to-End)

```python
# === 실제 사용 시나리오 ===

# 1. 원시 데이터 로드
market_data = load_market_features("RTX_5090")
# → [state_vector ...] 체크포인트/데이터셋 기준 state_dim 벡터 (현재 운영 256)

# 2. h 네트워크: 관측 → 잠재 상태
root_latent = h(market_data)
# → 256차원 Latent State

# 3. MCTS 탐색: 50번 시뮬레이션
config = MCTSConfig(num_simulations=50)
tree = MCTSTree(config, action_dim=5)
best_action, policy = tree.search(root_latent, g, f)
# 내부에서 일어나는 일:
#   f() 호출 ~55회  (루트 + 50번 시뮬레이션)
#   g() 호출 ~50회  (시뮬레이션마다 1회)

# 4. 결과 해석
action_names = ['ACTION_0', 'ACTION_1', 'ACTION_2', 'ACTION_3', 'ACTION_4']
print(f"추천: {action_names[best_action]}")
print(f"확률 분포: {policy}")
# 추천: ACTION_1
# 확률 분포: [0.06, 0.56, 0.04, 0.02, 0.32]
# → "지금 전량 구매가 56%로 가장 유력합니다"
```

---

## 6. 실행 테스트

```python
# mcts.py 하단의 레퍼런스 테스트 코드
h = RepresentationNetwork(state_dim=256, latent_dim=256).to(device)
g = DynamicsNetwork(latent_dim=256, action_dim=5).to(device)
f = PredictionNetwork(latent_dim=256, action_dim=5).to(device)

config = MCTSConfig(num_simulations=50)
tree = MCTSTree(config, action_dim=5)

root_observation = torch.randn(1, 256).to(device)
root_latent = h(root_observation)

best_action, policy_target = tree.search(root_latent, g, f)
```

**출력 결과:**
```
Starting MCTS search...
Best action: 1                                    ← ACTION_1
Policy target: tensor([0.04, 0.48, 0.06, 0.02, 0.40])  ← 방문 비율
MCTS Search completed successfully!
```
