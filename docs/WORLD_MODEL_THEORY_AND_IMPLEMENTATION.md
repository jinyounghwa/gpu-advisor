# WORLD MODEL 이론과 구현 (GPU Advisor 기준)

이 문서는 현재 저장소의 실제 코드 기준으로 월드 모델 이론과 구현을 함께 설명합니다.

## 1. 왜 월드 모델인가

단일 예측 모델은 "지금 무엇을 할지"만 바로 출력합니다. 월드 모델은 다음을 분리합니다.
1. 상태를 압축해 표현한다 (`h`)
2. 행동 후 미래를 전개한다 (`g`)
3. 해당 상태의 정책/가치를 평가한다 (`f`)

이 구조 위에서 MCTS가 여러 미래 경로를 탐색하므로, 단순 반응형 추론보다 계획 기반 의사결정이 가능해집니다.

## 2. 현재 프로젝트 구성

- Representation: `backend/models/representation_network.py`
- Dynamics: `backend/models/dynamics_network.py`
- Prediction: `backend/models/prediction_network.py`
- Planning(MCTS): `backend/models/mcts_engine.py`
- 운영 오케스트레이션: `backend/agent/gpu_purchase_agent.py`

운영 액션 라벨(5개):
- `BUY_NOW`, `WAIT_SHORT`, `WAIT_LONG`, `HOLD`, `SKIP`

## 3. 수학적 관점 요약

- 표현 함수: `z_t = h(x_t)`
- 동역학 함수: `(z_{t+1}, r_t) = g(z_t, a_t)`
- 예측 함수: `(pi_t, v_t) = f(z_t)`

MCTS는 트리에서 UCB 계열 점수로 노드를 선택하고,
리프에서 `g`, `f`를 사용해 rollout/평가 후 값을 역전파합니다.

## 4. 구현 코드: 핵심 경로

### 4.1 Representation (`h`)

```python
# backend/models/representation_network.py
class RepresentationNetwork(nn.Module):
    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        if state_tensor.dim() == 2:
            x = state_tensor.unsqueeze(1)
        else:
            x = state_tensor

        x = self.input_embedding(x)
        x = self.layer_norm1(x)
        x = self.pos_encoding(x)
        x = self.ff1(x)
        x = self.ff2(x)
        x = self.ff3(x)
        x = self.layer_norm2(x)
        s_0 = self.output_layer(x)
        return s_0.squeeze(1)
```

핵심:
- 입력 벡터(`state_dim`)를 latent(기본 256)로 매핑
- 운영에서는 체크포인트에서 `state_dim/latent_dim`을 동적으로 로드

### 4.2 Dynamics (`g`)

```python
# backend/models/dynamics_network.py
class DynamicsNetwork(nn.Module):
    def forward(self, s_t: torch.Tensor, a_t: torch.Tensor):
        x = torch.cat([s_t, a_t], dim=-1)
        x = self.input_layer(x)
        x = self.layer_norm1(x)
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm2(x)

        s_tp1 = self.next_state_head(x)
        reward_mean = self.reward_mean_head(x).squeeze(-1)
        reward_logvar = self.reward_logvar_head(x).squeeze(-1)
        reward_logvar = F.softplus(reward_logvar, beta=1.0)
        return s_tp1, reward_mean, reward_logvar
```

핵심:
- `(latent, one-hot action)` 결합 입력
- 다음 latent와 reward 분포(평균/분산 성격)를 함께 예측

### 4.3 Prediction (`f`)

```python
# backend/models/prediction_network.py
class PredictionNetwork(nn.Module):
    def forward(self, s_t: torch.Tensor):
        x = self.input_layer(s_t)
        x = self.layer_norm1(x)
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm2(x)

        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy_logits, value
```

핵심:
- 정책 logits + 상태 가치(value) 동시 출력
- MCTS 노드 확장 prior와 leaf value 계산에 사용

### 4.4 MCTS 엔진

```python
# backend/models/mcts_engine.py
@dataclass
class MCTSConfig:
    num_simulations: int = 50
    exploration: float = 1.4142
    rollout_steps: int = 5
    discount_factor: float = 0.99

class MCTSEngine:
    def search(self, root_state, policy_network, dynamics_network, device="cpu"):
        root = MCTSNode(state=root_state)
        for _ in range(self.config.num_simulations):
            node = self._select(root)
            if not node.is_expanded:
                node = self._expand(node, policy_network, device)
            value = self._simulate(node, policy_network, dynamics_network, device)
            self._backup(node, value)
        action_probs = self._get_action_probs(root, self.config.temperature)
        return action_probs, root.value, root
```

핵심:
- `select -> expand -> simulate -> backup`
- 최종적으로 방문수 기반 정책 분포를 반환

## 5. 운영 경로에서의 실제 호출

```python
# backend/agent/gpu_purchase_agent.py (요약)
state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
latent = self.representation_network(state_tensor).squeeze(0).cpu().numpy()

mcts_probs_np, root_value, _ = self.mcts.search(
    root_state=latent,
    policy_network=self.prediction_network,
    dynamics_network=self.dynamics_network,
    device=str(self.device),
)
```

이후 에이전트는 MCTS 정책 + 보상 기반 정책 + prior + utility bias를 혼합해 최종 행동 확률을 만들고,
신뢰도/엔트로피 안전 게이트를 거쳐 최종 액션을 결정합니다.

## 6. 데이터 파이프라인과 월드 모델 연결

- 크롤링: `crawlers/run_daily.py`
- 일별 feature 파일: `data/processed/dataset/training_data_YYYY-MM-DD.json`
- 상태 보고서: `docs/reports/YYYY-MM-DD/data_status_*.{json,md}`

월드 모델 품질은 이 데이터의 누적 기간/일관성에 직접 의존합니다.

## 7. 검증 포인트

1. 체크포인트 차원 일치 확인
- `input_dim`, `latent_dim`, `action_dim`이 모델 구성과 일치해야 함

2. 일별 데이터 연속성
- `docs/reports/latest_data_status.json`의 `coverage` 확인

3. 릴리즈 게이트
- `python3 backend/run_release_ready.py` 실행 후 `pass|blocked` 확인

## 8. 정리

현재 프로젝트는 MuZero 스타일 월드 모델(`h/g/f`) + MCTS 계획을 실제 코드로 구현하고,
일별 실데이터 수집과 상태 보고서, 릴리즈 게이트까지 연결한 구조입니다.
핵심 리스크는 구조 자체보다 데이터 품질/커버리지이며, 운영에서는 이를 보고서와 게이트로 지속 점검해야 합니다.
