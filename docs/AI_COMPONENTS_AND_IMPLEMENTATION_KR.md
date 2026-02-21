# GPU Advisor AI 구성요소 및 현재 구현 설명서 (한국어)

작성일: 2026-02-21  
대상 프로젝트: `gpu-advisor`

## 1. 문서 목적
이 문서는 본 프로젝트의 AI를 구성하는 핵심 요소를 설명하고, 각 요소가 현재 코드베이스에서 어떻게 구현되어 있는지 파일 단위로 연결해 설명합니다.

## 2. AI 시스템 전체 구조
프로젝트의 AI는 다음 5개 축으로 구성됩니다.

1. 상태 표현(Representation): 원시/가공 입력을 잠재 상태(latent state)로 변환
2. 세계 모델(World Model): 행동에 따른 다음 상태와 보상을 추정
3. 정책/가치 예측(Policy/Value): 현재 상태에서 행동 확률과 상태 가치를 예측
4. 계획(Planning, MCTS): 단순 1회 추론이 아니라 탐색 기반 의사결정 수행
5. 학습/평가/릴리즈 게이트: 모델 갱신, 성능 검증, 공개 가능 판정

## 3. 구성요소별 설명과 코드 매핑

### 3.1 Representation Network (h)
- 역할:
  - 입력 feature 벡터를 AI 내부 표현(잠재 상태)으로 변환
  - 이후 Dynamics/Prediction/MCTS가 사용하는 공통 상태 공간 생성
- 구현:
  - `backend/models/representation_network.py`
- 현재 프로젝트 동작:
  - `GPUPurchaseAgent`에서 state vector를 latent로 변환할 때 사용
  - 파인튜닝 시 학습 가능한 네트워크로 함께 업데이트

### 3.2 Dynamics Network (g)
- 역할:
  - `(현재 잠재상태, 행동)` -> `(다음 잠재상태, 예상 보상)` 추정
  - 실제 시장을 직접 실행하지 않고 내부 시뮬레이션 수행
- 구현:
  - `backend/models/dynamics_network.py`
- 현재 프로젝트 동작:
  - MCTS rollout에서 미래 전개 추정
  - 에이전트 응답의 `expected_rewards` 계산에도 사용
  - 파인튜닝 시 상태 전이/보상 손실로 학습

### 3.3 Prediction Network (f)
- 역할:
  - 잠재 상태에서 정책(policy logits)과 가치(value) 예측
  - MCTS 확장 시 prior와 value 신호 제공
- 구현:
  - `backend/models/prediction_network.py`
- 현재 프로젝트 동작:
  - MCTS의 노드 확장/시뮬레이션에 사용
  - 파인튜닝 시 정책/가치 손실을 통해 업데이트

### 3.4 MCTS Engine (Planning)
- 역할:
  - 단순 argmax 예측이 아닌 탐색 기반 의사결정
  - 각 행동 시나리오를 반복 시뮬레이션하여 최종 행동 확률 산출
- 구현:
  - `backend/models/mcts_engine.py`
- 현재 프로젝트 동작:
  - `GPUPurchaseAgent` 내부에서 `num_simulations` 기반 탐색 수행
  - 최종 행동: `BUY_NOW`, `WAIT_SHORT`, `WAIT_LONG`, `HOLD`, `SKIP`

### 3.5 Agent Orchestrator
- 역할:
  - 사용자 입력 모델명 정규화/매칭
  - 데이터셋에서 상태 벡터 조회
  - h/g/f + MCTS 결합 추론
  - 안전 게이트 적용 후 최종 행동 반환
- 구현:
  - `backend/agent/gpu_purchase_agent.py`
- 현재 프로젝트 동작:
  - API `/api/ask` 호출 시 핵심 엔진으로 사용
  - `safe_mode`, `safe_reason`, `entropy`, `confidence`를 trace로 반환

## 4. 학습 구성요소

### 4.1 Fine-tuner (실학습 루프)
- 역할:
  - 가공 데이터셋 + 가격 변화 기반 transition 생성
  - 정책/가치/동역학/보상 손실 결합 학습
  - 최신 체크포인트 저장
- 구현:
  - `backend/agent/fine_tuner.py`
- 현재 프로젝트 동작:
  - `POST /api/training/start` 호출 시 실학습 수행
  - 결과 저장: `alphazero_model_agent_latest.pth`
  - 재현성: `seed` 지원

### 4.2 학습 메타데이터
- 역할:
  - 실험 재현성과 릴리즈 추적성 보장
- 구현:
  - `fine_tuner.py`의 체크포인트 `meta`
- 저장 내용:
  - 학습 설정(`num_steps`, `batch_size`, `learning_rate`, `seed`)
  - 데이터 요약(`num_samples`, reward 통계 등)
  - 스키마 버전

## 5. 평가 구성요소

### 5.1 Backtest Evaluator
- 역할:
  - day(t) 의사결정을 day(t+1) 가격 실현값으로 평가
  - 정책 품질을 정량 지표로 제공
- 구현:
  - `backend/agent/evaluator.py`
- 지표:
  - `directional_accuracy_buy_vs_wait`
  - `avg_reward_per_decision`
  - `abstain_ratio`
  - `avg_confidence`

### 5.2 Release Pipeline
- 역할:
  - 준비도 -> 학습 -> 평가 -> 게이트 판정 -> 보고서 생성 자동화
- 구현:
  - `backend/agent/release_pipeline.py`
  - `backend/run_release_pipeline.py` (CLI)
- 산출물:
  - `docs/reports/YYYY-MM-DD/release_report_*.json`
  - `docs/reports/YYYY-MM-DD/release_report_*.md`

## 6. API 계층에서의 구현 방식

핵심 API는 `backend/simple_server.py`에 통합되어 있습니다.

1. 추론:
  - `POST /api/ask`
  - 에이전트 의사결정 + trace 반환

2. 준비도:
  - `GET /api/agent/readiness`
  - 데이터 일수 기준 30일 충족 여부 판정

3. 모델 정보:
  - `GET /api/agent/model-info`
  - 로드 체크포인트/파라미터/메타 조회

4. 평가:
  - `GET /api/agent/evaluate`
  - 백테스트 실행 및 지표 반환

5. 릴리즈 판정:
  - `GET /api/agent/release-check`
  - 품질 게이트 통과 여부 확인

6. 전체 파이프라인 실행:
  - `POST /api/agent/pipeline/run`
  - 최종 단계 자동화 실행

## 7. 프론트엔드에서의 노출
- 구현:
  - `frontend/app/page.tsx`
- 현재 표시:
  - 선택 행동, confidence, value
  - 안전 게이트 발동 여부(`safe_mode`, `safe_reason`)
  - 정책 분포, 기대보상 텍스트

## 8. 데이터 흐름(현재 구현 기준)
1. 크롤링 데이터 축적 (`data/raw/*`)  
2. feature 엔지니어링 결과 축적 (`data/processed/dataset/training_data_*.json`)  
3. 학습(`fine_tuner`)으로 최신 체크포인트 갱신  
4. 추론(`gpu_purchase_agent`)에서 최신 체크포인트 사용  
5. 평가/릴리즈 게이트로 공개 가능 여부 판정  

## 9. 현재 상태 해석
현재 시스템은 “구조적 완성도” 기준으로 다음을 갖추고 있습니다.
- 계획형 의사결정(MCTS)
- 실학습 루프
- 재현성(seed)
- 안전 게이트
- 백테스트
- 릴리즈 자동 판정 및 보고서

단, 실제 공개판정은 데이터 윈도우(예: 30일)와 게이트 통과 여부에 의해 최종 결정됩니다.

## 10. 관련 파일 빠른 참조
- 에이전트: `backend/agent/gpu_purchase_agent.py`
- 학습: `backend/agent/fine_tuner.py`
- 평가: `backend/agent/evaluator.py`
- 릴리즈 파이프라인: `backend/agent/release_pipeline.py`
- 서버 통합: `backend/simple_server.py`
- 프론트 표시: `frontend/app/page.tsx`
- 최종 개발 보고서: `docs/FINAL_DEVELOPMENT_REPORT_KR.md`
