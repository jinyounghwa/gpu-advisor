# GPU Advisor 프로젝트 리팩토링 계획

> **분석 기준일**: 2026-05-13  
> **대상 범위**: 프로젝트 전체 (backend, crawlers, frontend, config, tests, scripts)  
> **최종 업데이트**: 2026-05-13 (리팩토링 완료)

---

## 실행 완료 요약

### Phase 1: 프로젝트 정리 ✅ 완료

| 항목 | 조치 | 상태 |
|------|------|------|
| `scratch/` | 삭제 (일회용 스크립트) | ✅ |
| `backend/environment/` | 삭제 (빈 모듈, 미사용) | ✅ |
| `backend/inference/` | `backend/legacy/inference/`로 이동 (broken imports, 미사용) | ✅ |
| `backend/api/` | 삭제 후 `sentiment/`를 `backend/agent/sentiment/`로 이동 | ✅ |

### Phase 2: 모듈 구조 개선 ✅ 완료

| 항목 | 조치 | 상태 |
|------|------|------|
| `backend/server/` 모듈화 | 이미 완료 (app.py, routes/, schemas.py, dependencies.py, state.py) | ✅ |
| `simple_server.py` | thin wrapper로 정리 + 테스트 호환성 re-export | ✅ |
| `SecurityConfig` 중복 인스턴스 | `routes/auth.py`, `routes/system.py`가 `app.py` 싱글톤을 공유하도록 수정 | ✅ |
| Import 패턴 | `backend.*` 절대 경로로 통일 완료 | ✅ |

### Phase 3: 코드 중복 통합 ✅ 완료

| 항목 | 조치 | 상태 |
|------|------|------|
| `rewards.py` | `backend/agent/rewards.py`로 통합 (fine_tuner, evaluator 모두 사용) | ✅ |
| `data_loader.py` | `backend/agent/data_loader.py`로 통합 | ✅ |
| `quality_gates.py` | `backend/agent/quality_gates.py`로 통합 | ✅ |
| `sentiment/` | `backend/agent/sentiment/`로 이동 (의미상 agent 영역) | ✅ |

### Phase 4: 설정 관리 일원화 ✅ 완료

| 항목 | 조치 | 상태 |
|------|------|------|
| `gpu_purchase_agent.py` 매직 넘버 | `AgentConfig` dataclass로 추출 (14개 파라미터) | ✅ |
| `news_start = 80` 하드코딩 | `AgentConfig.news_start`로 이동 | ✅ |
| `PipelineConfig` ↔ `QualityGateConfig` 중복 | `PipelineConfig.resolve_quality_gates()` 메서드로 통합 | ✅ |

### Phase 5: 테스트 검증 ✅ 완료

| 테스트 | 결과 |
|--------|------|
| `test_networks.py` (11개) | ✅ PASS |
| `test_mcts.py` (7개) | ✅ PASS |
| `test_auth_and_sentiment.py` (2개) | ✅ PASS |
| `test_feature_engineer.py` (14개) | ✅ PASS |
| `test_status_report.py` (1개) | ✅ PASS |
| `test_release_*.py` (3개) | ✅ PASS |
| **총 39개 테스트** | **전원 통과** |

---

## 현재 모듈 구조 (리팩토링 후)

```
backend/
├── agent/
│   ├── __init__.py              # 공개 API export
│   ├── gpu_purchase_agent.py    # AgentConfig + GPUPurchaseAgent
│   ├── data_loader.py           # AgentDataLoader (통합)
│   ├── rewards.py               # calculate_reward (통합)
│   ├── quality_gates.py         # QualityGateConfig + check_quality_gates
│   ├── evaluator.py             # AgentEvaluator
│   ├── fine_tuner.py            # AgentFineTuner
│   ├── release_pipeline.py      # AgentReleasePipeline + PipelineConfig
│   ├── next_steps.py            # build_post_30d_next_steps
│   └── sentiment/
│       ├── __init__.py
│       └── analyzer.py          # NewsSentimentAnalyzer
├── models/
│   ├── representation_network.py
│   ├── dynamics_network.py
│   ├── prediction_network.py
│   ├── action_model.py
│   ├── mcts_engine.py
│   └── transformer_model.py
├── server/
│   ├── app.py                   # FastAPI app + SecurityConfig 싱글톤
│   ├── dependencies.py          # get_gpu_agent, get_repository
│   ├── schemas.py               # Pydantic models
│   ├── state.py                 # TrainingState, RateLimiter
│   └── routes/
│       ├── agent.py             # /api/agent/*
│       ├── training.py          # /api/training/*
│       ├── system.py            # /api/system/*
│       └── auth.py              # /api/auth/*
├── security/
│   └── auth.py
├── storage/
│   ├── base.py                  # RepositoryProtocol
│   ├── repository.py            # SQLiteRepository
│   └── postgres_repository.py   # PostgresRepository
├── data/
│   └── scaler.py
├── legacy/                      # 미사용 레거시 코드
│   ├── inference/               # broken imports, benchmark-only
│   ├── main.py, routes.py, ...
│   └── ...
├── simple_server.py             # 하위호환 진입점
└── wiki_context.py              # 개발자 유틸리티
```

---

## 향후 권장 작업 (Phase 6~7)

### Phase 6: 타입 안전성 강화
- `release_pipeline.run()` 반환값 → `ReleaseResult(TypedDict)`
- 커스텀 Exception 계층 (`GPUAdvisorError → ModelError / DataError / PipelineError`)

### Phase 7: 테스트 보강
- `gpu_purchase_agent.py` 유닛 테스트 (AgentConfig 기반 결정 로직)
- `release_pipeline.py` 통합 테스트
- `danawa_crawler.py` 테스트
