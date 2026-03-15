# WORLD MODEL 전환 실행 로드맵

이 문서는 월드 모델 기반 의사결정 시스템의 실행 로드맵입니다.
기준일: **2026-03-15(일)** (업데이트)

## 0. 현재 상태 요약

| 구분 | 상태 |
|------|------|
| h/g/f 잔차 블록 | ✓ 구현 완료 |
| ActionModel (a) | ✓ 구현 및 통합 완료 |
| PUCT (c=√2) | ✓ 적용 완료 |
| 디리클레 노이즈 (ε=0.25, α=0.03) | ✓ 루트 노드 적용 완료 |
| Gaussian NLL (reward 불확실성) | ✓ 구현 완료 |
| 4-신호 블렌드 (MCTS 45%/보상 25%/f-prior 15%/ActionModel 15%) | ✓ 적용 완료 |
| Anti-collapse uniform 수정 | ✓ 완료 |
| _action_from_delta() SKIP 버그 수정 | ✓ 완료 |
| grad_norm 실제 캡처 | ✓ 완료 |
| 자동 학습 오케스트레이션 (AutoTrainingConfig) | ✓ 구현 완료 |
| LaunchAgent 자동화 | ✓ 운영 중 |
| 데이터 창 | 23일 확보 (목표 30일, 잔여 ~7일) |
| 릴리즈 게이트 | 7/7 중 6 통과 (uplift 미통과) |

## 전환 목표 (재정의)

1. ~~의사결정 기준을 단일 점수 최대화에서 계획 품질 중심으로 전환~~ → **완료**
2. ~~운영 KPI를 단기 정확도 중심에서 n-step 성과 중심으로 전환~~ → **완료**
3. 저신뢰 구간에서 보수적 정책이 자동 발동 → **구현 완료, 지속 모니터링**
4. **신규**: 30일 도달 후 자동 학습 성공 및 uplift 게이트 통과 → **진행 중 (~7일)**

완료 기준(Definition of Done):
1. 릴리즈 리포트에 계획 KPI가 고정 포함된다. → **✓ 완료**
2. 액션별 사후 성능이 분리 추적된다. → **✓ 완료**
3. 저신뢰/데이터 이상 시 자동 보수 모드가 발동되고 로그로 검증 가능하다. → **✓ 완료**
4. 30일 데이터 기반 자동 학습 완료 및 uplift 게이트 통과. → **진행 중**

---

## 1. Week 1 (2026-02-23 ~ 2026-03-01): 계측/지표 전환 ✓ 완료

핵심 목표: "잘 맞췄나?"에서 "잘 계획했나?"로 관측 체계를 바꾼다.

**완료된 항목:**
- 계획 KPI (n_step_return, uncertainty_guard_rate 등) 릴리즈 리포트에 포함
- 운영 로그에 MCTS root value, policy entropy, 안전 게이트 개입 여부 기록
- LaunchAgent 자동화 설정 및 TCC 로그 경로 수정 (`~/Library/Logs/gpu-advisor/cron.log`)
- 릴리즈 리포트 7-게이트 체계 확립

---

## 2. Week 2 (2026-03-02 ~ 2026-03-08): 운영 정책 전환 ✓ 완료

핵심 목표: 불확실성이 높은 구간에서 자동 보수 정책이 작동하도록 만든다.

**완료된 항목:**
- PUCT 공식(c=√2) 실제 적용 (`_ucb_score()` 메서드 구현)
- 디리클레 노이즈(ε=0.25, α=0.03) 루트 노드에 실제 적용
- Anti-collapse regularizer: 경험적 prior → uniform 분포로 수정 (편향 제거)
- confidence < 0.25 → HOLD 강등, entropy > 1.58 → HOLD 강등 유지
- ActionModel(a) 구현 및 4-신호 블렌드 통합
- `_action_from_delta()` SKIP 레이블 버그 수정 (역방향 학습 방지)

---

## 3. Week 3 (2026-03-09 ~ 2026-03-15): 모델-데이터 정합성 강화 ✓ 완료

핵심 목표: 월드모델 품질 저하의 주원인인 데이터 리스크를 구조적으로 줄인다.

**완료된 항목:**
- `representation_network.py` docstring 22D → 256D 수정
- `dynamics_network.py` `reward_std` → `reward_var` (σ² 명확화)
- INFERENCE_WALKTHROUGH의 22D 오류 수정
- Gaussian NLL (`0.5 * ((r-μ)² * exp(-logvar) + logvar)`) reward 불확실성 실제 학습
- `grad_norm` 하드코딩 0.0 → `clip_grad_norm_()` 실제 반환값 캡처
- `auto_training.py` JSON 파싱 통일 (`_extract_first_json_object()`)
- AutoTrainingConfig + decide_auto_training_action() 자동 학습 오케스트레이션
- 문서 전체 업데이트 (README, 종합보고서, HYPERPARAMETER_GUIDE, AUTO_TRAINING_WORKFLOW 등)

**2026-03-15 기준 데이터 현황:**
- dataset: 23일 확보 (목표 30일, 잔여 7일)
- 방향정확도: 0.630, 평균 보상: +0.000276 (양수)
- 게이트: 6/7 통과

---

## 4. Week 4 (2026-03-16 ~ 2026-03-22): 30일 도달 및 첫 자동 학습

핵심 목표: 30일 데이터 창 달성 후 자동 학습을 실행하고 uplift 게이트를 통과한다.

**실행 항목:**
1. 일일 수집 유지 (LaunchAgent 자동 실행 중)
   - 매일 `~/Library/Logs/gpu-advisor/cron.log` 확인
   - 데이터 이상 시 `python3 crawlers/run_daily.py` 수동 실행

2. 30일 도달 자동 학습 모니터링
   - `auto_training.py`가 `decide_auto_training_action()` → `train_release` 결정
   - 실행: `backend/run_release_ready.py`
   - 결과: `docs/reports/latest_auto_training_status.md`

3. 게이트 판정 후 조치
   - `pass`: 릴리즈 후보 태그 생성 고려
   - `blocked`: 실패 게이트 원인 분석 → 하이퍼파라미터 조정 → 재학습 (7일 인터벌)

**검증 기준:**
1. `auto_training_state.json`에 `last_trained_data_date` 기록됨
2. 릴리즈 게이트 7/7 통과 (`uplift_raw_vs_buy` 포함)
3. 2주 연속 운영에서 치명적 정책 역전(저신뢰 구간 공격적 매수) 0건

**산출물:**
1. 첫 자동 학습 결과 (`alphazero_model_agent_latest.pth`)
2. 30일 기반 릴리즈 리포트
3. 자동화 상태 리포트 (`latest_auto_training_status.md`)

---

## 5. 역할 분담 가이드

1. Data/Crawler 담당
   - 데이터 커버리지, 결측/이상치, 드리프트 감시
   - LaunchAgent 실행 로그 모니터링

2. Model/Planning 담당
   - h/g/f/a 성능, MCTS 파라미터, 리플레이셋 회귀
   - Gaussian NLL logvar 헤드 학습 추세 확인

3. Agent/Backend 담당
   - 정책 게이트, 추론 로그, 릴리즈 체크 자동화
   - AutoTrainingConfig 파라미터 관리

4. Product/Ops 담당
   - KPI 해석, 액션 사후 성과 분석, 정책 임계치 의사결정

---

## 6. 리스크와 대응

1. 리스크: 30일 도달 후에도 uplift 게이트 미통과
   - 대응: 학습 스텝 증가(`--steps 1000`), lookback_days 조정, reward 함수 검토

2. 리스크: 데이터 품질 부족으로 계획 성능 불안정
   - 대응: 데이터 게이트를 릴리즈 필수 조건으로 강제, 이상치 필터링 강화

3. 리스크: 보수 정책 과도 적용으로 기회 손실
   - 대응: `uncertainty_guard_rate`와 사후 성과를 같이 최적화
   - 현재 관망비율(0.659)은 허용 범위(< 0.85) 내

4. 리스크: LaunchAgent 실행 실패
   - 대응: 매일 `launchctl print gui/$(id -u)/com.gpu-advisor.daily-crawl`로 상태 확인

---

## 7. 다음 실행 항목 (2026-03-16~)

1. LaunchAgent 정상 실행 및 데이터 수집 유지 확인 (매일)
2. `current_min_days >= 30` 도달 시 `auto_training.py`가 자동 학습 트리거 확인
3. 30일 기반 릴리즈 리포트에서 uplift 게이트 통과 여부 판정
4. `pass`이면 v0.3.0 릴리즈 태그 검토; `blocked`이면 실패 게이트 원인 분석

---

참조 문서:
- `docs/WORLD_MODEL_THEORY_AND_IMPLEMENTATION.md`
- `docs/SAFETY_MECHANISMS.md`
- `docs/AUTO_TRAINING_WORKFLOW_KR.md`
- `docs/POST_30D_NEXT_STEPS_KR.md`
- `docs/HYPERPARAMETER_GUIDE_KR.md`
