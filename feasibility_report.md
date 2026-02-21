# GPU Advisor World Model Feasibility Report (Current Project)

## 1. 요약
- 결론: 본 프로젝트의 월드 모델(`h/g/f`) + MCTS 접근은 **GPU 구매 시점 의사결정 보조** 목적에 대해 실현 가능하다.
- 전제: 초단타(HFT) 같은 초저지연 목적이 아니라, 일 단위/주 단위의 구매 의사결정 보조를 목표로 한다.
- 구현 상태: `backend/agent/gpu_purchase_agent.py`에서 체크포인트 기반 모델 로드, `backend/models/mcts_engine.py` 기반 계획 탐색, 안전 게이트를 포함한 최종 행동 출력까지 구성되어 있다.

## 2. 현재 코드 기준 핵심 사실

### 2.1 월드 모델 구성
- `RepresentationNetwork` (`backend/models/representation_network.py`): 입력 상태 벡터를 latent state로 변환
- `DynamicsNetwork` (`backend/models/dynamics_network.py`): `(latent, action) -> (next_latent, reward)` 예측
- `PredictionNetwork` (`backend/models/prediction_network.py`): `latent -> (policy_logits, value)` 출력

### 2.2 계획(Planning)
- `MCTSEngine` (`backend/models/mcts_engine.py`) 기본값
  - `num_simulations=50`
  - `rollout_steps=5`
  - `discount_factor=0.99`
- 에이전트는 MCTS 결과를 보정(calibration)하여 최종 정책을 만든다.

### 2.3 액션 공간
- 운영 액션 라벨(`backend/agent/gpu_purchase_agent.py`):
  - `BUY_NOW`, `WAIT_SHORT`, `WAIT_LONG`, `HOLD`, `SKIP`
- 고정 7액션(롱/숏 레버리지) 구조는 현재 프로젝트 구현이 아니다.

### 2.4 입력 차원
- 운영 입력 차원은 체크포인트에서 동적으로 로드된다.
- 현재 체크포인트(`alphazero_model_agent_latest.pth`) 기준 `input_dim=256`, `latent_dim=256`, `action_dim=5`.

## 3. 타당성 평가

### 3.1 강점
- 단순 분류/회귀가 아닌 미래 시나리오 탐색(MCTS) 기반 의사결정
- 데이터 파이프라인(`crawlers/run_daily.py`)과 상태 보고서(`crawlers/status_report.py`)로 일별 누적/품질 확인 가능
- 안전 게이트(신뢰도/엔트로피 기반) 포함으로 과신 행동 완화

### 3.2 제약
- 학습 품질은 데이터 커버리지와 분포 안정성에 크게 의존
- MCTS 시뮬레이션 수를 늘리면 품질 잠재 이득이 있으나 지연 증가
- 뉴스 0건인 날은 허용되지만, 장기적으로는 변동 요인 반영 한계가 생길 수 있음

## 4. 운영 권장 기준
- 크롤링: `setup_cron.sh`로 매일 자동 실행(기본 00:00)
- 상태 확인: `docs/reports/latest_data_status.{json,md}`
- 릴리즈 판정: `python3 backend/run_release_ready.py`
  - 30일 준비도/평가 지표/게이트 기반 `pass|blocked` 판정

## 5. 결론
현재 저장소 구현과 운영 절차를 기준으로, 본 프로젝트의 월드 모델 방식은 "GPU 구매 타이밍 보조" 문제에 대해 기술적으로 일관되고 실행 가능한 접근이다. 다만 품질의 핵심은 모델 구조보다 **실데이터 누적 일수와 지속 검증 체계**에 있다.
