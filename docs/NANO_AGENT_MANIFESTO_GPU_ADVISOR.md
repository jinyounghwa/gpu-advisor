# GPU Advisor 기술 해설서

- 문서 버전: v2.0
- 작성일: 2026-03-15
- 프로젝트: `gpu-advisor`


---

## Executive Summary

이 프로젝트는 "GPU를 지금 사야 하는가, 기다려야 하는가"라는 일상적이지만 난해한 의사결정 문제를, **예측**이 아니라 **계획(Planning)** 문제로 재정의한다.
핵심은 AlphaZero/MuZero 계열의 아이디어를 차용한 월드 모델 기반 에이전트이며, 시장 데이터를 잠재 상태로 압축한 뒤 MCTS로 다중 시나리오를 탐색해 행동을 선택한다.

2026-03-15 기준 구현이 달성한 것:

1. 크롤러-피처-에이전트-API-리포트까지 이어지는 E2E 파이프라인 자동화 (LaunchAgent 기반)
2. 256차원 상태벡터 기반의 계획형 의사결정 엔진 (h/g/f/a 4-네트워크 월드 모델)
3. 릴리즈 게이트(정확도/보상/엔트로피/모드붕괴/베이스라인 uplift) 기반 품질 판정 체계
4. 30일 도달 시 자동 학습/재학습 오케스트레이션 (AutoTrainingConfig + decide_auto_training_action)

현재 운영 상태 (2026-03-15):
- 데이터 창: dataset 23일 확보 (목표 30일, 잔여 7일)
- 샘플 수: 308건
- 방향정확도: 0.630 / 평균 보상: +0.000276 (양수 전환)
- 게이트: 7개 중 6개 통과 (`uplift_raw_vs_buy` 데이터 부족으로 미통과)
- 종합 판정: `blocked` (데이터 창 미달)

아키텍처는 완성되고 대부분 지표도 양호하다. 현재의 핵심 과제는 **7일 추가 데이터 축적과 uplift 게이트 통과**다.

---

## Part 1. Manifesto - 선언문

### 1.1 데이터 드리븐 AI의 한계

전통적인 데이터 드리븐 접근은 대개 "정답 레이블 예측 정확도"를 우선한다. 하지만 실제 구매 의사결정은 다음 특성을 갖는다.

1. 정답이 단일값이 아니다.
   같은 가격 경로에서도 자금 상황, 대체재, 기다림 비용에 따라 최적 행동이 달라진다.
2. 의사결정은 연속적이다.
   오늘 `BUY`를 선택하지 않으면 내일 다시 선택해야 하며, 전략은 시계열적으로 연결된다.
3. 시장은 확률적이다.
   단일 시점 회귀값보다, 복수 미래 경로에서의 기대 효용이 더 중요하다.

따라서 단순 회귀/분류만으로는 "언제 사야 하는가" 문제를 충분히 해결하기 어렵다.

### 1.2 월드 모델로의 패러다임 전환

GPU Advisor는 문제를 이렇게 바꾼다.

1. 입력을 상태(state)로 본다.
2. 행동(action)을 정의한다. (`BUY_NOW`, `WAIT_SHORT`, `WAIT_LONG`, `HOLD`, `SKIP`)
3. 행동 후 전이를 모델링한다. (Dynamics Network)
4. 가치(value)와 정책(policy)을 동시에 학습한다.
5. MCTS로 행동 전개를 탐색한다.
6. ActionModel(a)로 행동 사전 분포를 학습해 4-신호 블렌드로 보정한다.

즉 "가격 예측 모델"이 아니라 "행동 최적화 모델"이다.

### 1.3 나노급 에이전트 선언

본 프로젝트의 선언은 다음과 같다.

1. 큰 파라미터보다 올바른 문제정식화가 우선이다.
2. LLM 호출을 반복하는 오케스트레이션보다, 도메인 월드 모델 + 탐색이 의사결정 품질에 유리한 구간이 분명히 존재한다.
3. 운영 가능한 소형 모델(약 18.9M + 43K 파라미터)로도 고난도 의사결정 자동화가 가능하다.
4. 에이전트는 "말 잘하는 시스템"이 아니라 "행동 품질이 검증되는 시스템"이어야 한다.

---

## Part 2. Theory - 이론적 배경

### 2.1 월드 모델이란 무엇인가

월드 모델은 환경을 잠재공간(latent space)에서 근사하는 내부 시뮬레이터다.

GPU Advisor의 구성 (4-네트워크):

1. Representation `h`: 관측 상태벡터(256D) → 잠재상태(256D)
2. Dynamics `g`: 잠재상태 + 행동 → 다음 잠재상태 + 보상(μ, σ²)
3. Prediction `f`: 잠재상태 → 정책(logits) + 가치(value)
4. Action Model `a`: 잠재상태 → 행동 사전 확률 (학습된 prior)

결과적으로 모델은 "다음 가격 숫자 하나"가 아니라, 행동에 따른 상태 전개와 장기 효용을 내재화한다.

### 2.2 AlphaZero / MuZero 철학

차용한 철학은 아래와 같다.

1. 가치함수와 정책함수를 결합해 탐색 효율을 높인다.
2. 학습된 모델을 MCTS의 휴리스틱으로 사용한다.
3. PUCT 공식(`Q + c·P·√N_parent/(1+N_child)`, c=√2)으로 탐색-활용 균형을 맞춘다.
4. 루트 노드에 디리클레 노이즈(ε=0.25, α=0.03)를 추가해 탐색 다양성을 확보한다.
5. 탐색 결과가 다시 학습 목표를 강화한다.

GPU Advisor는 완전정보 게임이 아니므로 진정한 self-play 대신, **히스토리 리플레이 기반 전이 샘플**로 학습한다.

### 2.3 나노급 에이전트의 정의와 가능성

이 문맥의 "나노급"은 대형 파운데이션 모델 대비 경량이면서도, 계획 기반으로 고품질 결정을 수행하는 에이전트를 뜻한다.

현재 체크포인트(`alphazero_model_agent_latest.pth`) 기준:

| 네트워크 | 파라미터 | 역할 |
|----------|----------|------|
| Representation (h) | ~921,600 | 256D 입력 → 256D 잠재 |
| Dynamics (g) | ~8,671,490 | 잠재 상태 전이 + 보상(μ, σ²) |
| Prediction (f) | ~9,330,182 | 정책 + 가치 |
| Action Model (a) | ~43,000 | 행동 사전 확률 |
| **총합** | **~18,966,272** | |

이 규모는 대규모 LLM 대비 매우 작지만, 특정 도메인 행동문제에서는 충분한 실용성을 보인다.

### 2.4 기존 LLM 기반 에이전트와의 차이

LLM 에이전트 중심 구조와 대비하면 다음 차이가 핵심이다.

1. 추론 중심
   - LLM 에이전트: 텍스트 추론 + 툴 호출
   - 본 시스템: 잠재 동역학 + MCTS 계획
2. 출력 형태
   - LLM 에이전트: 설명 텍스트
   - 본 시스템: 행동 확률분포, 기대보상, 가치, 안전오버라이드
3. 검증 가능성
   - LLM 에이전트: 평가가 상대적으로 정성적
   - 본 시스템: 백테스트 보상, 정확도, 엔트로피, uplift 등 정량 게이트
4. 재현성
   - 본 시스템은 데이터/체크포인트/게이트 기준으로 릴리즈 판정이 비교적 명확하다.

---

## Part 3. Case Study - GPU Advisor

### 3.1 프로젝트 동기와 목표

목표는 단순하다.

1. GPU 구매 타이밍 의사결정 자동화
2. "지금 매수 / 단기 대기 / 중기 대기 / 관망 / 회피" 선택지 제공
3. 실데이터 기반 일일 운영과 품질 게이트 자동화

### 3.2 문제 정의: 구매 타이밍을 게임 이론으로 재정의

문제 재정의:

1. 상태: 가격/환율/뉴스/공급/시간/기술지표로 구성된 256D 벡터
2. 행동: 5개 이산 행동 (BUY_NOW/WAIT_SHORT/WAIT_LONG/HOLD/SKIP)
3. 전이: 오늘 행동이 내일 상태와 보상에 미치는 영향 추정
4. 보상: 미래 가격 변화율에 따른 행동 적합도
5. 목적: 단일 시점 정확도 극대화가 아니라, 행동 연쇄의 기대 효용 극대화

### 3.3 시스템 아키텍처 전체 구조

```text
[Crawlers]
  Danawa / Exchange / News
        ↓
[Feature Engineering]
  256D state_vector (GPU별)
        ↓
[Agent Core]
  h(Representation) → g(Dynamics) → f(Prediction) + a(ActionModel) + MCTS(PUCT)
        ↓
[4-신호 블렌드 보정]
  45% MCTS + 25% 보상정책 + 15% f-net prior + 15% ActionModel prior
        ↓
[FastAPI Service]
  /api/ask, /api/agent/*, /api/training/*
        ↓
[Storage]
  SQLite or PostgreSQL
        ↓
[Reports]
  latest_data_status / latest_release_report / latest_auto_training_status
        ↓
[LaunchAgent 자동화]
  macOS LaunchAgent (자정 실행) + Auto Training Orchestration
```

### 3.4 데이터 파이프라인 상세 (크롤러 → 피처 엔지니어링 → 256차원)

일일 배치 실행: `crawlers/run_daily.py` (LaunchAgent로 자동 실행)

1. 다나와 가격 수집
2. 환율 수집 (`USD/KRW`, `JPY/KRW`, `EUR/KRW`)
3. 뉴스 + 감성 통계 수집
4. Feature Engineering 실행
5. 상태 보고서 및 릴리즈 리포트 생성
6. 자동 학습 오케스트레이션 (`auto_training.py`)

피처 256D 구성(`crawlers/feature_engineer.py`):

| 그룹 | 차원 | 내용 |
|------|------|------|
| 가격 피처 | 60D | 정규화 가격, MA7/MA14/MA30, 변화율, 변동성 |
| 환율 피처 | 20D | USD/KRW, JPY/KRW, EUR/KRW |
| 뉴스 피처 | 30D | 감성 점수, 기사 수, 긍정/부정 비율 |
| 시장 피처 | 20D | 판매자 수, 재고 상태 |
| 시간 피처 | 20D | 요일, 월, 연말 여부 |
| 기술지표 | 106D | RSI, MACD, 모멘텀, 패딩 |
| **합계** | **256D** | |

2026-03-15 기준 운영 데이터 현황:

| 소스 | 파일 수 | 범위 일수 | 최초일 |
|------|---------|-----------|--------|
| danawa | 26 | 33일 | 2026-02-11 |
| exchange | 23 | 23일 | 2026-02-21 |
| news | 23 | 23일 | 2026-02-21 |
| dataset | 23 | 23일 | 2026-02-21 |

- `current_min_days`: 23일 (목표 30일, 잔여 7일)
- `ready_for_30d_training`: `false` (약 2026-03-22 도달 예상)

### 3.5 AI 엔진 상세 (h / g / f / a 네트워크)

#### Representation Network `h` (`backend/models/representation_network.py`)

- 입력: 256D GPU 시장 상태 벡터
- `Linear(256, 256)` + LayerNorm + 잔차 FFN 블록 3개
- 출력: 256D latent state

#### Dynamics Network `g` (`backend/models/dynamics_network.py`)

- 입력: latent(256) + action one-hot(5)
- 잔차 블록 4개 (hidden 512)
- 출력:
  - next latent(256)
  - reward mean (μ)
  - reward log-variance (→ σ² = exp(logvar), Gaussian NLL 학습)

#### Prediction Network `f` (`backend/models/prediction_network.py`)

- 입력: latent(256)
- 잔차 블록 4개 (hidden 512)
- 출력:
  - policy logits(5)
  - value (tanh, -1~1)

#### Action Model `a` (`backend/models/action_model.py`)

- ActionEmbeddingLayer(5 actions, 16D embed) + ActionPriorNetwork(256→128→64→5)
- 파라미터: ~43K
- 역할: 잠재 상태에서 행동 사전 확률 예측 (기존 `utility_bias` 휴리스틱 대체)

### 3.6 MCTS 시뮬레이션 구현

`backend/models/mcts_engine.py` 기준:

| 파라미터 | 값 |
|----------|-----|
| num_simulations | 50 |
| rollout_steps | 5 |
| discount_factor | 0.99 |
| exploration (c_puct) | √2 ≈ 1.414 |
| dirichlet_epsilon | 0.25 |
| dirichlet_alpha | 0.03 |

탐색 공식 (PUCT, AlphaZero 스타일):
```
UCB(s, a) = Q(s, a) + c × P(s, a) × √N_parent / (1 + N_child)
```

루트 노드에 디리클레 노이즈 적용:
```python
policy = (1 - 0.25) × policy + 0.25 × Dirichlet([0.03] × 5)
```

추론 시(`backend/agent/gpu_purchase_agent.py`)에는 4-신호 블렌드 보정:

```python
calibrated = 0.45 × MCTS정책 + 0.25 × 보상정책 + 0.15 × f-net사전확률 + 0.15 × ActionModel사전확률
```

안전장치:
1. 낮은 confidence (< 0.25) → `HOLD` 강등
2. 높은 entropy (> 1.58) → `HOLD` 강등
3. 엔트로피 붕괴 방지: entropy < 0.65이면 균등 분포(uniform)와 혼합

### 3.7 자동화 운영 구조

```
macOS LaunchAgent (자정 00:00)
  → crawlers/run_daily.py
  → auto_training.py (decide_auto_training_action)
      → release_check (데이터 부족 / 재학습 불필요 시)
      → train_release (30일 도달 / 재학습 간격 충족 시)
```

- LaunchAgent 로그: `~/Library/Logs/gpu-advisor/cron.log`
- 상세 로그: `data/gpu-advisor/logs/daily_crawl.log`
- 상태 파일: `data/processed/auto_training_state.json`
- 재학습 간격: 7일 (`GPU_ADVISOR_AUTO_RETRAIN_EVERY_DAYS`)

---

## Part 4. Results & Validation

### 4.1 학습 결과 현황 (2026-03-15)

| 지표 | 2026-02-23 | 2026-03-15 | 비고 |
|------|-----------|-----------|------|
| 샘플 수 | 42 | 308 | 7.3× 증가 |
| 방향정확도 | 0.619 | 0.630 | 개선 |
| 평균 보상 (raw) | -0.000275 | +0.000276 | 양수 전환 |
| 관망비율 | 0.667 | 0.659 | 안정 |
| 평균 신뢰도 | 0.228 | 0.231 | 유사 |

게이트 통과 현황:

| 게이트 | 결과 |
|--------|------|
| accuracy_raw | ✓ 통과 |
| reward_raw | ✓ 통과 |
| abstain | ✓ 통과 |
| safe_override | ✓ 통과 |
| action_entropy_raw | ✓ 통과 |
| no_mode_collapse_raw | ✓ 통과 |
| uplift_raw_vs_buy | ✗ 미통과 |

현재 결론: "아키텍처 성공, 데이터 창 확보 중"

- 7개 게이트 중 6개 통과 (이전: 5개)
- 평균 보상이 양수로 전환됨 (이전: 음수)
- `uplift_raw_vs_buy` 미통과는 always-buy 베이스라인 대비 초과수익 미달 — 데이터 30일 도달 후 재판정 예정

### 4.2 베이스라인 비교 (2026-03-15)

| 지표 | 값 |
|------|-----|
| 에이전트 평균 보상 | +0.000276 |
| Always Buy 보상 | (30일 달성 후 측정 예정) |
| uplift_raw_vs_buy | 미통과 (데이터 부족) |

---

## Part 5. Vision

### 5.1 단기 실행 계획 (2026-03-15 기준)

1. **잔여 7일 데이터 수집**: LaunchAgent가 매일 자동 실행 중
2. **30일 도달 시 자동 학습**: `auto_training.py`가 `train_release` 결정 후 전체 파이프라인 실행
3. **uplift 게이트 통과**: 30일 데이터 기반 재학습 후 베이스라인 대비 초과수익 검증

### 5.2 도메인 확장 로드맵 (항공권, 중고차, 주식)

이 아키텍처는 "지금 행동 vs 대기" 의사결정 문제에 일반화 가능하다.

#### 1) 항공권

1. 상태: 노선별 운임 곡선, 좌석가용성, 시즌성, 환율, 이벤트 캘린더
2. 행동: 즉시구매/단기대기/중기대기/회피
3. 보상: 구매 후 일정 기간 대비 기회손익

#### 2) 중고차

1. 상태: 매물가격, 감가율, 재고회전, 지역별 수요, 금리
2. 행동: 즉시구매/탐색확장/대기/회피
3. 보상: 같은 조건 대비 체결가격 개선율

#### 3) 주식

1. 상태: 멀티팩터, 이벤트, 마이크로구조, 뉴스 감성
2. 행동: 매수/부분매수/관망/축소/회피
3. 보상: 리스크조정 수익 (단, 규제/리스크 체계 필수)

### 5.3 공통 확장 원칙

1. 레이블 예측보다 행동 품질 지표를 우선한다.
2. 월드 모델(h/g/f/a) + 탐색(MCTS/PUCT) + 안전오버라이드의 구조를 유지한다.
3. 릴리즈는 always-X 베이스라인 uplift 통과를 최소 조건으로 한다.
4. 데이터 창 길이와 커버리지(결측 마스크 포함)를 운영 KPI로 관리한다.

---

## 부록: 리스크 및 한계

### A. 리스크 목록

1. 데이터 드리프트: 시즌/신제품 출시 주기 변화
2. 크롤링 노이즈: 누락/파싱 실패/사이트 구조 변경
3. 보상함수 단순화: 실제 사용자 효용(기회비용, 예산제약) 반영 한계
4. 단기 데이터의 과적합 위험 (30일 미만 구간)

### B. 운영 체크리스트

1. LaunchAgent 로그: `~/Library/Logs/gpu-advisor/cron.log`
2. 소스별 `range_days` 추세
3. 게이트 통과율 (목표: 7/7)
4. 액션 분포 엔트로피 (모드 붕괴 감시)
5. DB 저장 건수 및 API 실패율

### C. 실험 계획 (30일/60일/90일)

1. **30일 (~2026-03-22)**: 기본 게이트 전체 통과 달성, 자동 학습 첫 실행
2. **60일 (~2026-04-22)**: 베이스라인 uplift의 통계적 유의성 점검, 재학습 7일 주기 검증
3. **90일 (~2026-05-22)**: 도메인 전이 실험 또는 API 서비스 공개 검토

### D. 문서-코드 정합성 (완료 항목)

| 항목 | 상태 |
|------|------|
| `representation_network.py` 22D 주석 → 256D 수정 | ✓ 완료 |
| `dynamics_network.py` `reward_std` → `reward_var` 수정 | ✓ 완료 |
| MCTS `ucb_score` 프로퍼티 deprecated 주석 추가 | ✓ 완료 |
| MCTS 루트 디리클레 노이즈 실제 적용 | ✓ 완료 |
| `grad_norm` 하드코딩 0.0 → 실제 캡처 | ✓ 완료 |
| `_action_from_delta()` SKIP 레이블 버그 수정 | ✓ 완료 |
| Anti-collapse uniform 분포 수정 | ✓ 완료 |
| ActionModel(a) 전체 통합 | ✓ 완료 |
| Gaussian NLL reward 불확실성 학습 | ✓ 완료 |
| 릴리즈 리포트 raw/override 분리 | ✓ 완료 |

---

## 결론

GPU Advisor는 "작은 모델 + 월드 모델(h/g/f/a) + MCTS(PUCT)" 조합으로, 구매 타이밍 문제를 실제 운영 가능한 에이전트 파이프라인으로 구현했다.
2026-03-15 기준으로 시스템은 기술적으로 성숙한 상태이며, 대부분의 품질 게이트를 통과했다. 현재의 핵심 과제는 **잔여 7일 데이터 축적과 uplift 게이트 통과를 통한 통계적 실증 완성**이다.

이 프로젝트는 나노급 에이전트의 가능성을 이미 보여줬고, 다음 단계는 30일 데이터 기반의 자동 학습 완료로 그 가능성을 통계적으로 입증하는 일이다.
