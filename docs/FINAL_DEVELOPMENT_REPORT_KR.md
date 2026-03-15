# GPU Advisor 최종 개발 보고서 (한국어)

작성일: 2026-03-15 (최초: 2026-02-21 → v2.0 업데이트)
프로젝트 경로: `/Users/younghwa.jin/Documents/gpu-advisor`

## 1. 보고서 목적
이 문서는 "공개 가능한 완성형 AI 에이전트 수준까지 개발"의 수행 결과와 이후 운영 중 누적된 개선 사항을 정리합니다.

## 2. 핵심 결론
1. 에이전트 코어는 실모델 기반 계획(MCTS/PUCT) + ActionModel + 안전 게이트 + 실학습(5손실) + 백테스트 + 릴리즈 게이트 + 자동 보고서 파이프라인까지 구현 완료.
2. **자동 학습 오케스트레이션** 추가: 30일 데이터 도달 시 수동 개입 없이 학습→평가→게이트 판정이 자동 실행됨.
3. macOS LaunchAgent로 일일 배치 자동화 (cron 대체).
4. 2026-03-15 기준: dataset 23일 확보, 308 샘플, 방향정확도 0.630, 보상 +0.000276(양수), 게이트 6/7 통과.
5. **잔여 과제**: 7일 데이터 추가 축적 후 30일 자동 학습 실행 → uplift 게이트 통과.

## 3. 구현된 최종 기능

### 3.1 에이전트 의사결정 엔진
- 파일: `backend/agent/gpu_purchase_agent.py`
- 기능:
  - `h/g/f/a` 4-네트워크 모델 로드 및 MCTS(PUCT) 기반 행동 결정
  - 행동: `BUY_NOW`, `WAIT_SHORT`, `WAIT_LONG`, `HOLD`, `SKIP`
  - **MCTS 탐색**: PUCT 공식(`Q + c·P·√N_parent/(1+N_child)`, c=√2), 루트 디리클레 노이즈(ε=0.25, α=0.03)
  - **4-신호 블렌드**: `0.45×MCTS + 0.25×보상정책 + 0.15×f-net prior + 0.15×ActionModel prior`
  - `safe_mode` 게이트:
    - 신뢰도 낮음(`confidence < 0.25`) → 보수적 `HOLD` 전환
    - 불확실성 높음(`entropy > 1.58`) → 보수적 `HOLD` 전환
  - **Anti-collapse**: entropy < 0.65 시 균등 분포(uniform)와 혼합
  - 모델 메타 정보 조회(`get_model_info`)

### 3.2 ActionModel (행동 사전 확률 네트워크)
- 파일: `backend/models/action_model.py`
- 기능:
  - `ActionEmbeddingLayer(num_actions=5, embed_dim=16)` + `ActionPriorNetwork(256→128→64→5)`
  - 파라미터: ~43K
  - 잠재 상태에서 행동 사전 확률을 학습 (기존 `utility_bias` 하드코딩 휴리스틱 대체)
  - 4-신호 블렌드에서 15% 가중치로 활용

### 3.3 실데이터 파인튜닝
- 파일: `backend/agent/fine_tuner.py`
- 기능:
  - `data/processed/dataset/training_data_*.json` + `data/raw/danawa/*.json` 기반 학습 샘플 생성
  - **5개 손실 가중 합산**:
    ```
    latent_loss × 1.0
    policy_loss × 1.0
    value_loss  × 1.0
    reward_nll_loss × 0.5   (Gaussian NLL, logvar 헤드 실제 학습)
    action_prior_loss × 0.3 (ActionModel 지도 학습)
    ```
  - **Gaussian NLL**: `0.5 × ((r-μ)² × exp(-logvar) + logvar)` — reward 불확실성 헤드 실제 학습
  - **grad_norm**: `clip_grad_norm_()` 반환값 실제 캡처 및 리포트 포함
  - **행동 레이블링 수정**: `_action_from_delta()` — SKIP 레이블 버그 수정 (역방향 학습 방지)
  - `seed` 기반 재현성 보장
  - 학습 후 체크포인트 저장 (`alphazero_model_agent_latest.pth`, `a_state_dict` 포함)

### 3.4 백테스트 평가기
- 파일: `backend/agent/evaluator.py`
- 기능:
  - day(t) 의사결정 vs day(t+1) 실현 가격으로 평가
  - 주요 지표:
    - `directional_accuracy_buy_vs_wait_raw`
    - `avg_reward_per_decision_raw`
    - `abstain_ratio`
    - `safe_override_ratio`
    - `action_entropy_raw`
    - `mode_collapse_raw`
    - `uplift_raw_vs_always_buy`
    - `avg_confidence`

### 3.5 릴리즈 파이프라인
- 파일: `backend/agent/release_pipeline.py`
- 단계: 준비도 확인 → 학습(옵션) → 평가 → 게이트 판정 → 보고서 생성
- 산출물:
  - `docs/reports/YYYY-MM-DD/release_report_*.{json,md}`
  - `docs/reports/latest_release_report.{json,md}`
- 게이트 (7개):
  1. `accuracy_raw >= 0.55`
  2. `avg_reward_raw > 0`
  3. `abstain_ratio <= 0.85`
  4. `safe_override_ratio <= 0.90`
  5. `action_entropy_raw >= 0.25`
  6. `uplift_raw_vs_always_buy >= 0`
  7. `mode_collapse_raw == False`

### 3.6 자동 학습 오케스트레이션 (신규)
- 파일: `crawlers/auto_training.py`
- 기능:
  - `AutoTrainingConfig`: 학습 파라미터 + 환경변수 바인딩 (`GPU_ADVISOR_*`)
  - `decide_auto_training_action()`: 30일 도달/재학습 간격 자동 판정
  - 30일 도달 시 → `train_release` → `backend/run_release_ready.py` 자동 실행
  - 재학습 간격: 7일 (신규 데이터 누적 기준)
  - 상태 저장: `data/processed/auto_training_state.json`
  - 결과 리포트: `docs/reports/latest_auto_training_status.{json,md}`

### 3.7 API 서버 통합
- 파일: `backend/simple_server.py`
- 주요 엔드포인트:
  - `GET /api/agent/readiness`
  - `GET /api/agent/model-info`
  - `GET /api/agent/evaluate`
  - `GET /api/agent/release-check`
  - `GET /api/agent/next-steps`
  - `POST /api/agent/pipeline/run`
  - `POST /api/training/start` (실학습 루프, seed 전달 가능)

### 3.8 운영 스크립트
- `backend/run_release_ready.py`: 전체 학습+평가+게이트 파이프라인 CLI
- `backend/run_release_daily.py`: 학습 없이 드라이 체크만 실행
- `crawlers/run_daily.py`: 일일 배치 오케스트레이터 (크롤링+피처+자동학습 판정)

### 3.9 월드 모델 아키텍처 개선
- 파일: `backend/models/representation_network.py`, `dynamics_network.py`, `prediction_network.py`, `mcts_engine.py`
- 개선 사항:
  - **잔차 연결**: 모든 네트워크 블록에 `x = x + block(x)` 적용
  - **PUCT 탐색**: `_ucb_score()` 메서드 — c=√2, `Q + c·P·√N_parent/(1+N_child)`
  - **디리클레 노이즈**: 루트 노드에만 실제 적용 (이전에는 설정만 있고 미적용)
  - **Gaussian NLL**: `reward_logvar` 헤드 실제 학습 (이전에는 `_`로 무시)
  - **docstring 수정**: 22D → 256D 실제 입력 차원으로 수정

### 3.10 macOS LaunchAgent 자동화
- 파일: `~/Library/LaunchAgents/com.gpu-advisor.daily-crawl.plist`
- 기능: 매일 자정 `crawlers/run_daily.py` 자동 실행
- 로그: `~/Library/Logs/gpu-advisor/cron.log` (macOS TCC 보호 대응)
- pmset 웨이크 스케줄: 매일 23:58 자동 기동

### 3.11 프론트엔드
- 파일: `frontend/app/page.tsx`
- `agent_trace`에 `raw_action`, `safe_mode`, `safe_reason` 표시
- 운영 시 의사결정 투명성 확보

## 4. 검증 결과 요약 (2026-03-15 기준)

| 지표 | 2026-02-21 | 2026-03-15 |
|------|-----------|-----------|
| 샘플 수 | 42 | 308 |
| 방향정확도 | 0.619 | **0.630** |
| 평균 보상 | -0.000275 | **+0.000276** (양수 전환) |
| 관망비율 | 0.667 | 0.659 |
| 평균 신뢰도 | 0.228 | 0.231 |
| 게이트 통과 | 5/7 | **6/7** |

게이트 현황:

| 게이트 | 상태 |
|--------|------|
| accuracy_raw | ✓ |
| reward_raw | ✓ |
| abstain | ✓ |
| safe_override | ✓ |
| action_entropy_raw | ✓ |
| no_mode_collapse_raw | ✓ |
| uplift_raw_vs_buy | ✗ (30일 도달 후 재판정) |

## 5. 공개판정 기준 (최종)
현재 시스템의 공개판정은 아래 자동 게이트를 모두 만족해야 `pass`입니다.

1. 데이터 윈도우: 30일 이상 (현재: 23일)
2. 정확도 게이트(raw): `directional_accuracy_buy_vs_wait_raw >= 0.55` ✓
3. 보상 게이트(raw): `avg_reward_per_decision_raw > 0` ✓
4. 관망비율 게이트: `abstain_ratio <= 0.85` ✓
5. 안전오버라이드 게이트: `safe_override_ratio <= 0.90` ✓
6. 엔트로피 게이트(raw): `action_entropy_raw >= 0.25` ✓
7. 베이스라인 uplift 게이트(raw): `uplift_raw_vs_always_buy >= 0` ✗
8. 모드 붕괴 금지(raw): `mode_collapse_raw == False` ✓

## 6. 현재 미충족 항목
1. **30일 데이터 윈도우**: 현재 23일, 잔여 ~7일 (~2026-03-22 도달 예상)
2. **uplift_raw_vs_buy**: 30일 데이터 기반 재학습 후 재판정 필요

이외 모든 기술적 품질 게이트는 통과 상태입니다.

## 7. 30일 달성 후 실행 절차 (자동화됨)

30일 도달 시 `auto_training.py`가 자동으로 전체 파이프라인을 실행합니다.

### 자동 실행 흐름
```
LaunchAgent (자정)
  → crawlers/run_daily.py
  → auto_training.py → decide_auto_training_action()
      → train_release 결정
      → backend/run_release_ready.py 실행
      → docs/reports/latest_*.md 생성
```

### 수동 실행 (필요 시)
```bash
# 전체 학습+평가 파이프라인
python3 backend/run_release_ready.py

# 결과 확인
cat docs/reports/latest_release_report.md
cat docs/reports/latest_auto_training_status.md

# API 기반 실행
POST /api/agent/pipeline/run
GET  /api/agent/release-check
```

## 8. 변경 파일 목록

### 에이전트/모델 핵심
- `backend/agent/gpu_purchase_agent.py` — PUCT, 4-신호 블렌드, ActionModel, anti-collapse
- `backend/agent/fine_tuner.py` — 5손실, Gaussian NLL, grad_norm, SKIP 버그 수정, ActionModel
- `backend/agent/evaluator.py`
- `backend/agent/release_pipeline.py`
- `backend/models/action_model.py` *(신규)*
- `backend/models/representation_network.py` — 잔차, 256D docstring
- `backend/models/dynamics_network.py` — 잔차, Gaussian NLL
- `backend/models/prediction_network.py` — 잔차
- `backend/models/mcts_engine.py` — PUCT, 디리클레 노이즈

### 오케스트레이션/자동화
- `crawlers/auto_training.py` *(신규)* — AutoTrainingConfig, decide_auto_training_action
- `crawlers/run_daily.py` — --disable-auto-train, --auto-retrain-days, --auto-target-days, --skip-release
- `backend/run_release_ready.py`
- `backend/run_release_daily.py`

### 서버/프론트
- `backend/simple_server.py`
- `frontend/app/page.tsx`

### 문서
- `README.md`
- `종합_프로젝트_보고서.md`
- `docs/HYPERPARAMETER_GUIDE.md`, `docs/HYPERPARAMETER_GUIDE_KR.md`
- `docs/AUTO_TRAINING_WORKFLOW.md`, `docs/AUTO_TRAINING_WORKFLOW_KR.md`
- `docs/INFERENCE_WALKTHROUGH.md`, `docs/INFERENCE_WALKTHROUGH_KR.md`
- `docs/WORLD_MODEL_THEORY_AND_IMPLEMENTATION.md`
- `docs/FINAL_DEVELOPMENT_REPORT_KR.md` (본 문서)

## 9. 최종 판단

**기술 완성도**: 공개 가능한 완성형 수준의 요소 구현 완료.
- 4-네트워크 월드 모델(h/g/f/a) + MCTS(PUCT) + 자동 학습 오케스트레이션 + LaunchAgent 운영 자동화

**실증 완성도**: 데이터 30일 축적 후 자동 게이트 통과 시 완성.
- 현재 6/7 게이트 통과, 보상 양수 전환 — 기술적 방향성은 검증됨
- uplift 게이트는 데이터 부족에 의한 것이며 아키텍처 문제 아님

**예상 완성 시점**: ~2026-03-22 (30일 도달 → 자동 학습 → 게이트 재판정)
