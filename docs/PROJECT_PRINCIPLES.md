# GPU Advisor: AI 원리 및 코드 심층 분석

이 문서는 GPU Advisor 프로젝트에 적용된 **MuZero/AlphaZero 기반의 강화학습 알고리즘**과 그 핵심 코드 구현을 심층적으로 설명합니다.

---

## 🧠 1. 핵심 원리: MuZero 스타일의 World Model

전통적인 AI는 "오늘의 날씨가 어떠니?"라는 질문에 답하기 위해 과거 데이터를 학습합니다. 하지만 이 프로젝트는 **"내가 지금 우산을 사면, 내일 비가 올 확률과 우산 가격 변동을 시뮬레이션했을 때 얼마나 이득인가?"**를 고민합니다.

이를 위해 MuZero의 3가지 핵심 신경망 구조를 채택했습니다.

### (1) Representation Network (h) - "현 상태의 해석"
- **위치:** `backend/models/representation_network.py`
- **역할:** 다나와 가격, 환율, 뉴스 점수 등 256차원의 입력 데이터를 AI가 시뮬레이션하기 좋은 형태인 **Latent State(잠재 상태)**로 압축합니다.
- **코드 특징:** `FeedForward` 블록 3개를 앙상블(Ensemble) 형태로 쌓아 복잡한 시장 상황을 정교하게 표현합니다.

### (2) Dynamics Network (g) - "내부 세계관(Physics)"
- **위치:** `backend/models/dynamics_network.py`
- **역할:** "만약 지금 안 사고 1주 대기를 선택한다면(Action), 미래의 시장 상태(Next Latent State)는 어떻게 변하며 그때의 이득(Reward)은 얼마인가?"를 **상상**합니다.
- **핵심:** 실제 세상을 직접 겪어보지 않고도, AI 내부에서 미래를 예측하는 **월드 모델** 역할을 수행합니다.

### (3) Prediction Network (f) - "직관적 판단"
- **위치:** `backend/models/prediction_network.py`
- **역할:** 현재 상태에서 어떤 행동이 가장 좋을 확률이 높은지(Policy)와, 현재 상태가 얼마나 유리한지(Value)를 출력합니다.
- **코드 특징:** Tanh 활성화 함수를 사용하여 가치를 -1(매우 불리)에서 1(매우 유리) 사이로 표현합니다.

---

## 🌲 2. MCTS (Monte Carlo Tree Search) 엔진

AI는 단순히 신경망의 출력값만 믿지 않습니다. 수십 번의 **시뮬레이션**을 거쳐 검증합니다.

### 동작 시퀀스 (`backend/models/mcts_engine.py`)
1. **Selection (선택):** UCB (Upper Confidence Bound) 점수가 가장 높은 자식 노드를 선택하여 내려갑니다.
2. **Expansion (확장):** 가보지 않은 길이라면 Prediction Network를 통해 새로운 가지를 칩니다.
3. **Simulation (시뮬레이션):** Dynamics Network를 이용해 5단계(config.rollout_steps) 앞의 미래까지 "상상 속의 플레이"를 진행합니다.
4. **Backup (역전파):** 상상 끝에 얻은 보상을 부모 노드들에게 전달하여 "이 길은 좋은 길이었다"는 것을 기록합니다.

---

## 🔗 3. 데이터에서 프론트엔드까지의 연결 고리

전체 시스템이 연결되는 과정은 다음과 같습니다.

### [Phase 1] 데이터 수집 (Daily)
- `crawlers/run_daily.py`가 작동하며 `data/raw/`에 매일의 데이터를 쌓습니다.
- `crawlers/feature_engineer.py`가 이 데이터를 256차원 벡터로 가공하여 `data/processed/`에 저장합니다.

### [Phase 2] 백엔드 서비스 (FastAPI)
- `backend/simple_server.py`가 실행될 때 `alphazero_model.pth` 파일을 로드합니다.
- **예측 API (`/api/ask`):** 사용자가 모델을 요청하면 MCTS 엔진이 가동되어 구매/대기 액션과 신뢰도, 근거를 반환합니다.
- **지표 API (`/api/training/metrics`, `/api/training/metrics/stream`):** 서버에서 AI가 학습 중인 경우, 지표 조회 및 SSE 스트리밍을 제공합니다.

### [Phase 3] 프론트엔드 (Next.js)
- `frontend/app/page.tsx`는 사용자의 입력을 백엔드로 전달합니다.
- `frontend/app/components/TrainingDashboard.tsx`는 백엔드에서 보내주는 실시간 지표를 받아 동적인 그래프로 그려냅니다.

---

## 💻 4. 핵심 코드 요약 (Inference flow)

시스템이 구매 추천을 생성하는 핵심 로직은 다음과 같습니다.

```python
# 1. 원시 데이터를 256D Feature로 변환
features = engineer.generate_features(model_name) 

# 2. Representation Network를 통해 Latent State 생성
latent_state = representation_net(features)

# 3. MCTS 엔진을 통한 미래 시뮬레이션
# 50번의 '구매/대기' 시나리오를 시뮬레이션함
action_probs, value, tree = mcts_engine.search(
    latent_state, 
    prediction_net, 
    dynamics_net
)

# 4. 가장 방문 횟수가 많은 행동(최적 행동)을 기반으로 점수 산출
buy_score = action_probs[0] * 100 # Action 0이 'Buy'인 경우
```

이 고도로 설계된 시스템은 데이터 수집(Crawler) -> 전처리(Feature) -> 시뮬레이션(AI) -> 시각화(Frontend)의 유기적인 결합체입니다.
