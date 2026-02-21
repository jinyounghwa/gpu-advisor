# GPU Advisor 최종 개발 보고서 (한국어)

작성일: 2026-02-21  
프로젝트 경로: `/Users/younghwa.jin/Documents/gpu-advisor`

## 1. 보고서 목적
이 문서는 요청사항인 "크롤러 코드는 수정하지 않고, 공개 가능한 완성형 AI 에이전트 수준까지 개발"의 수행 결과를 정리합니다.

## 2. 핵심 결론
1. 크롤러(`crawlers/`)는 변경하지 않았습니다.
2. 에이전트 코어는 실모델 기반 계획(MCTS) + 안전 게이트 + 실학습 + 백테스트 + 릴리즈 게이트 + 자동 보고서 파이프라인까지 구현되었습니다.
3. 현재 데이터 기간은 30일 미만이므로 릴리즈 판정은 자동으로 `blocked` 됩니다(의도된 안전 동작).
4. 30일 충족 후 동일 파이프라인을 실행하면 공개판정(`pass/blocked`)을 자동 산출할 수 있습니다.

## 3. 구현된 최종 기능

### 3.1 에이전트 의사결정 엔진
- 파일: `backend/agent/gpu_purchase_agent.py`
- 기능:
  - `h/g/f` 모델 로드 및 MCTS 기반 행동 결정
  - 행동: `BUY_NOW`, `WAIT_SHORT`, `WAIT_LONG`, `HOLD`, `SKIP`
  - `safe_mode` 게이트:
    - 신뢰도 낮음(`min_confidence`)
    - 불확실성 높음(`max_entropy`)
    - 위 조건에서 보수적 `HOLD`로 자동 전환
  - 모델 메타 정보 조회(`get_model_info`)

### 3.2 실데이터 파인튜닝
- 파일: `backend/agent/fine_tuner.py`
- 기능:
  - `data/processed/dataset/training_data_*.json` + `data/raw/danawa/*.json` 기반 학습 샘플 생성
  - `seed` 기반 재현성 보장
  - 학습 후 체크포인트 저장:
    - `alphazero_model_agent_latest.pth`
    - 학습 설정/데이터 요약 메타 포함

### 3.3 백테스트 평가기
- 파일: `backend/agent/evaluator.py`
- 기능:
  - day(t) 의사결정 vs day(t+1) 실현 가격으로 평가
  - 주요 지표:
    - `directional_accuracy_buy_vs_wait_raw` (원정책 기준)
    - `avg_reward_per_decision_raw` (원정책 기준)
    - `abstain_ratio`
    - `safe_override_ratio`
    - `action_entropy_raw`
    - `mode_collapse_raw`
    - `uplift_raw_vs_always_buy`
    - `avg_confidence`

### 3.4 릴리즈 파이프라인(최종 단계 자동화)
- 파일: `backend/agent/release_pipeline.py`
  - 기능:
    - 단계: 준비도 확인 -> 학습(옵션) -> 평가 -> 게이트 판정 -> 보고서 생성
  - 산출물:
    - `docs/reports/YYYY-MM-DD/release_report_*.json`
    - `docs/reports/YYYY-MM-DD/release_report_*.md`
  - 게이트:
    - 정확도(raw) 최소치
    - 평균보상(raw) 최소치
    - 관망/안전오버라이드 비율 제한
    - 행동 엔트로피(raw) 하한
    - 베이스라인 대비 uplift(raw) 하한
    - 모드 붕괴(raw) 금지

### 3.5 API 서버 통합
- 파일: `backend/simple_server.py`
- 추가된 엔드포인트:
  - `GET /api/agent/readiness`
  - `GET /api/agent/model-info`
  - `GET /api/agent/evaluate`
  - `GET /api/agent/release-check`
  - `POST /api/agent/pipeline/run`
- 학습 엔드포인트 개선:
  - `POST /api/training/start`가 시뮬레이션이 아닌 실학습 루프 사용
  - `seed` 전달 가능

### 3.6 운영 보조 스크립트
- 파일: `backend/run_release_pipeline.py`
- 기능: CLI로 릴리즈 파이프라인 실행

### 3.7 프론트 표시 강화
- 파일: `frontend/app/page.tsx`
- 기능:
  - `agent_trace`에 `raw_action`, `safe_mode`, `safe_reason` 표시
  - 운영 시 의사결정 투명성 확보

### 3.8 문서 정합화
- 파일: `README.md`
- 기능:
  - 실제 API 목록으로 갱신
  - 릴리즈 파이프라인 API 반영

## 4. 검증 결과 요약
1. 컴파일: `python3 -m compileall backend` 통과
2. 파이프라인 실행 검증:
   - `POST /api/agent/pipeline/run` 동작
   - 보고서 파일 생성 확인
3. 현재 릴리즈 상태:
   - `insufficient_data_window`로 `blocked`
   - 원인: 최소 축 데이터 일수 30 미충족

## 5. 공개판정 기준(최종)
현재 시스템의 공개판정은 아래 자동 게이트를 모두 만족해야 `pass`입니다.
1. 데이터 윈도우: 30일 이상
2. 정확도 게이트(raw): `directional_accuracy_buy_vs_wait_raw >= 0.55`
3. 보상 게이트(raw): `avg_reward_per_decision_raw > 0`
4. 관망비율 게이트: `abstain_ratio <= 0.85`
5. 안전오버라이드 게이트: `safe_override_ratio <= 0.90`
6. 엔트로피 게이트(raw): `action_entropy_raw >= 0.25`
7. 베이스라인 uplift 게이트(raw): `uplift_raw_vs_always_buy >= 0`
8. 모드 붕괴 금지(raw): `mode_collapse_raw == False`

## 6. 현재 미충족 항목
1. 30일 데이터 윈도우 미충족
2. 데이터가 쌓이기 전에는 평가 지표 변동성이 크므로 최종 품질판정이 불안정

## 7. 30일 달성 후 실행 절차(운영 Runbook)
1. 학습 실행  
   `POST /api/training/start`
2. 학습 완료 확인  
   `GET /api/training/status`
3. 종합 파이프라인 실행  
   `POST /api/agent/pipeline/run`
4. 결과 확인  
   `GET /api/agent/release-check`
5. 보고서 확인  
   `docs/reports/YYYY-MM-DD/release_report_*.md`

## 8. 변경 파일 목록 (크롤러 제외)
- `backend/agent/__init__.py`
- `backend/agent/gpu_purchase_agent.py`
- `backend/agent/fine_tuner.py`
- `backend/agent/evaluator.py`
- `backend/agent/release_pipeline.py`
- `backend/simple_server.py`
- `backend/run_release_pipeline.py`
- `frontend/app/page.tsx`
- `README.md`
- `docs/FINAL_DEVELOPMENT_REPORT_KR.md` (본 문서)

## 9. 크롤러 관련 참고
본 문서 작성 시점의 변경 범위는 에이전트/서버 중심이었으며, 이후 운영 단계에서 `crawlers/`는 실데이터 수집 및 리포트 자동화 방향으로 추가 개선되었습니다.

## 10. 최종 판단
아키텍처/운영 기능 기준으로는 공개 가능한 완성형 수준에 필요한 요소를 구현 완료했습니다.  
다만 "실제 공개 승인"은 데이터 30일 충족 후 자동 게이트 통과 여부(`pass`)로 판정해야 합니다.
