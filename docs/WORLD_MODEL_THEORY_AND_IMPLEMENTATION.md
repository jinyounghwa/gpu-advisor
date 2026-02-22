# WORLD MODEL 이론과 구현 (GPU Advisor 기준)

이 문서는 GPU Advisor를 기준으로, 단순 데이터 최적화(optimizer) 사고에서 월드모델 기반 예측/계획(planning) 사고로 전환하기 위한 학습 문서입니다.

## 1. 핵심 메시지: 전환의 방향

결론부터 말하면 방향은 맞습니다.  
다만 "데이터 기반 의사결정"을 버리는 것이 아니라 다음처럼 재정의해야 합니다.

1. 데이터는 정답을 바로 뽑는 입력이 아니라, 세계를 학습하는 재료다.
2. 모델은 단일 시점 점수화기가 아니라, 미래를 전개하는 시뮬레이터다.
3. 의사결정은 점수 최대화가 아니라, 시뮬레이션 기반 계획 문제다.

즉, `optimizer 중심`에서 `world model + planner 중심`으로 무게중심을 이동하되,
데이터는 더 중요해집니다. (학습 데이터 + 운영 검증 데이터 + 반례 데이터)

## 2. 왜 Optimizer만으로는 한계가 생기는가

Optimizer 성향(예: "지금 feature에서 행동 점수 argmax")은 실무에서 빠르지만 다음 문제가 있습니다.

1. 비정상(non-stationary) 환경 취약
- 환율/재고/프로모션/정책 변화처럼 체제 변환(regime shift)이 오면 과거 패턴 최적화가 깨집니다.

2. 행동-결과 인과 분리 부족
- "무슨 일이 일어났는지"는 설명하지만, "내 행동 때문에 무엇이 달라졌는지"를 분리해 다루기 어렵습니다.

3. 멀티스텝 의사결정 손실
- 오늘의 최고점 행동이 2주 뒤에는 손해가 될 수 있는데, 단일 스텝 최적화는 이 경로 의존성을 잘 반영하지 못합니다.

4. 안전/보수적 운영 어려움
- 불확실성이 큰 구간에서 "보류"를 택해야 하는데, 확신 과대 모델은 과감한 행동을 과잉 추천할 수 있습니다.

## 3. 월드모델 관점에서의 의사결정 구조

GPU Advisor의 MuZero 스타일 분해:

1. `h` (Representation): 관측 상태 `x_t`를 잠재 상태 `z_t`로 압축
2. `g` (Dynamics): `z_t, a_t`에서 `z_{t+1}`과 보상/리스크를 전개
3. `f` (Prediction): 각 상태의 정책 prior와 가치 예측
4. MCTS (Planner): 여러 행동 시나리오를 탐색해 장기 기대값 기준으로 행동 선택

수식 요약:
- `z_t = h(x_t)`
- `(z_{t+1}, r_t) = g(z_t, a_t)`
- `(pi_t, v_t) = f(z_t)`

의사결정은 `f` 단독 출력이 아니라 `search(h, g, f)` 결과를 사용합니다.

## 4. 현재 코드와 1:1 매핑

- Representation: `backend/models/representation_network.py`
- Dynamics: `backend/models/dynamics_network.py`
- Prediction: `backend/models/prediction_network.py`
- Planner(MCTS): `backend/models/mcts_engine.py`
- Orchestration: `backend/agent/gpu_purchase_agent.py`

운영 액션 라벨:
- `BUY_NOW`
- `WAIT_SHORT`
- `WAIT_LONG`
- `HOLD`
- `SKIP`

운영 경로 요약:
1. 상태 벡터 -> `representation_network`로 latent 변환
2. `mcts_engine.search`가 `prediction_network`와 `dynamics_network`를 반복 호출
3. 방문수 기반 정책 + 보조 정책/바이어스 + 안전 게이트를 거쳐 최종 액션 선택

## 5. 학습 프레임: 무엇을 공부해야 하는가

### 5.1 모델 구조 이해
1. `h/g/f` 분해 이유: 학습 안정성 + 계획 가능성
2. latent 차원(`state_dim`, `latent_dim`)과 액션 차원(`action_dim`) 정합성
3. reward/value/entropy의 해석

### 5.2 계획 알고리즘 이해
1. MCTS의 `select -> expand -> simulate -> backup`
2. 탐색-활용 균형(exploration coefficient)
3. `num_simulations`, `rollout_steps`, `discount_factor`의 효과

### 5.3 데이터/운영 연결 이해
1. 데이터 수집: `crawlers/run_daily.py`
2. 학습 입력: `data/processed/dataset/training_data_YYYY-MM-DD.json`
3. 운영 품질 모니터링: `docs/reports/latest_data_status.json`
4. 릴리즈 게이트: `python3 backend/run_release_ready.py`

## 6. 전환 실행안: Optimizer -> World Model 조직 습관

### 6.1 의사결정 질문 바꾸기
- Before: "지금 feature에서 점수가 가장 높은 행동은?"
- After: "행동별 미래 경로를 전개했을 때 기대가치/리스크/불확실성이 어떤가?"

### 6.2 KPI 바꾸기
- Before: 단기 정확도/즉시 reward
- After:
1. n-step 누적 보상
2. 계획 대비 실제 오차(trajectory consistency)
3. 불확실성 높은 구간에서의 보수적 의사결정 비율
4. 대기 전략(`WAIT_*`)의 사후 효용

### 6.3 운영 정책 바꾸기
1. 고신뢰 구간: 계획 결과 적극 반영
2. 저신뢰 구간: 보류/관측 모드로 전환
3. 급격한 환경변화 감지 시: 탐색 계수 및 안전 게이트 자동 상향

## 7. 반론 가능성과 균형점

### 반론 1: "월드모델은 복잡하고 비용이 크다"
맞습니다. 하지만 실환경이 멀티스텝/비정상일수록 단순 모델의 숨은 비용(오판, 재학습 반복)이 더 커집니다.

### 반론 2: "데이터 품질이 낮으면 월드모델도 무너진다"
맞습니다. 그래서 전환의 핵심은 모델 교체가 아니라 데이터 거버넌스 강화입니다.

### 균형점
최적 접근은 `Optimizer 폐기`가 아니라 `Optimizer를 월드모델 플래너의 특수 케이스로 내재화`하는 것입니다.
즉, 단기 점수화는 planner의 휴리스틱으로 유지하고, 최종 결정은 계획 기반으로 승격합니다.

## 8. 팀 적용 체크리스트

1. 문서/회의에서 "예측 정확도"만이 아니라 "계획 품질" 지표를 함께 보고 있는가
2. 실패 사례를 단일 시점 오분류가 아니라 "시뮬레이션 경로 붕괴" 관점으로 리뷰하는가
3. `latest_data_status`를 릴리즈 전 필수 게이트로 사용하는가
4. 불확실성 높은 구간에서 자동 보수 정책이 발동되는가
5. 액션별 사후 성능(`BUY_NOW`, `WAIT_*`, `HOLD`, `SKIP`)을 분리 모니터링하는가

## 9. 결론

당신의 방향(데이터 최적화 중심에서 월드모델 기반 예측/계획 중심으로 전환)은 타당합니다.  
정확한 표현은 다음입니다.

- "데이터 기반 의사결정"을 넘어서는 것이 아니라
- "데이터를 이용한 세계모사 + 계획 기반 의사결정"으로 진화한다.

GPU Advisor의 현재 구조는 이미 이 방향(`h/g/f + MCTS`) 위에 있으므로,  
앞으로의 핵심 과제는 모델 구조 변경보다 `데이터 품질`, `계획 KPI`, `운영 안전정책`의 체계화입니다.
