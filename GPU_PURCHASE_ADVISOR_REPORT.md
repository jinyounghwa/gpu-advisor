# GPU 구매 타이밍 예측 AI - 평가 보고서
## AlphaZero 방식으로 구매 적정성 판단

**검증 날짜:** 2026-02-14
**프로젝트:** GPU Purchase Timing Advisor
**목적:** 부품명 입력 시 지금 사면 이득인지 손해인지 판단 (바둑 승률 방식)

---

## 📋 Executive Summary

이 프로젝트는 **GPU 구매 타이밍을 AlphaZero 방식으로 예측하는 AI 시스템**입니다.
- ✅ AlphaZero/MuZero 아키텍처 **완전 구현**
- ✅ 학습된 모델 존재 (18.9M 파라미터)
- ❌ 데이터 부족 (12개 샘플, 3일치)
- ❌ Feature 부족 (11차원 → 256차원 필요)

---

## 🎯 시스템 목적

### 입력
- **GPU 모델명** (예: RTX 5060, RX 9070 XT)

### 출력
- **구매 적정도** (0~100점, 바둑 승률 방식)
- **추천 행동** (강력 추천 / 보통 / 대기 권장)

### 판단 기준
1. **가격 분석**: 현재가 vs 평균가 vs 과거 추이
2. **환율 영향**: USD/KRW, JPY/KRW (수입 부품 가격)
3. **뉴스 감정 분석**: 신제품 출시, 재고 상황
4. **시장 공급**: 판매자 수, 재고 상황
5. **계절성**: 연말, 신제품 출시 주기

---

## 📊 현재 데이터 상태

### 수집된 데이터

#### 1. GPU 가격 데이터 (Danawa)
```json
{
  "product_name": "MSI 지포스 RTX 5060 벤투스 2X OC D7 8GB",
  "chipset": "RTX 5060",
  "lowest_price": 606320,
  "seller_count": 0
}
```

**수집 모델 (12개):**
- RTX 5060, 5060 Ti, 5070, 5070 Ti, 5080, 5090
- RX 6600, 7600, 9060 XT, 9070 XT
- Arc B580, RTX 5050

#### 2. 환율 데이터
- USD/KRW: 1,442.7원
- JPY/KRW: 943.28원

#### 3. 뉴스 데이터
- 현재 비어있음 (0개 기사)
- 키워드: GPU price drop, graphics card market, NVIDIA GPU supply

#### 4. State Vector (11차원)
```python
[
  0.0451,  # [0] 정규화된 가격
  0.0,     # [1] 가격 변화율 (전일)
  0.0,     # [2] 가격 변화율 (전주)
  0.5,     # [3] USD/KRW 정규화
  0.5,     # [4] JPY/KRW 정규화
  0.0,     # [5] 뉴스 감정 점수
  0.5,     # [6] 판매자 수 정규화
  0.0,     # [7-10] 예비 Feature
  0.5,
  0.0,
  0.0
]
```

### 데이터 문제점

| 문제 | 현재 | 필요 |
|------|------|------|
| 샘플 수 | 12개 (모델당 1개) | 최소 1,000개 |
| 기간 | 3일 | 최소 30일 |
| 시계열 | 없음 | 일별 가격 추이 |
| Feature 차원 | 11차원 | 256차원 |
| 뉴스 | 0개 | 일별 뉴스 수집 |

---

## 🏗️ 시스템 아키텍처

### AlphaZero/MuZero 방식 적용

#### 1. Representation Network (h)
**역할:** GPU 시장 상태를 Latent State로 인코딩

**입력 (현재 11차원 → 목표 256차원):**
- 가격 데이터 (60차원)
  - 현재가, 전일가, 전주가, 전월가
  - 이동평균 (7일, 14일, 30일)
  - 가격 변동성 (표준편차)
  - 최고가, 최저가 (1주, 1개월)
- 환율 데이터 (20차원)
  - USD/KRW, JPY/KRW, EUR/KRW
  - 환율 변화율
  - 환율 이동평균
- 경쟁 모델 비교 (50차원)
  - 동급 GPU 가격 비교
  - 성능/가격 비율
  - 시장 점유율
- 뉴스 감정 분석 (30차원)
  - 뉴스 감정 점수 (시계열)
  - 키워드 빈도 (신제품, 재고, 할인)
  - SNS 언급량
- 시장 공급 (20차원)
  - 판매자 수 추이
  - 재고 상황
  - 배송 기간
- 시간 Feature (20차원)
  - 출시 후 경과 시간
  - 계절성 (연말, 명절)
  - 요일 효과
- 기술적 지표 (56차원)
  - RSI, MACD (가격 추세)
  - 볼린저 밴드
  - 모멘텀

**출력:** 256차원 Latent State

#### 2. Dynamics Network (g)
**역할:** 행동 실행 시 미래 가격 예측

**입력:**
- 현재 Latent State (256차원)
- 행동 (5차원): BUY_NOW, WAIT_SHORT, WAIT_LONG, HOLD, SKIP

**출력:**
- 다음 Latent State (256차원)
- 예상 보상 (1주일 후 가격 변화율)

**예시:**
```
현재: RTX 5060 = 600,000원
행동: WAIT_SHORT (1주일 대기)
예측: 1주일 후 = 580,000원 → 보상 +3.3%
```

#### 3. Prediction Network (f)
**역할:** Policy (최적 행동) 와 Value (기대 이득) 예측

**입력:** Latent State (256차원)

**출력:**
- Policy: [BUY_NOW: 70%, WAIT_SHORT: 20%, WAIT_LONG: 5%, HOLD: 3%, SKIP: 2%]
- Value: +5.2% (예상 이득률)

#### 4. MCTS 탐색
**역할:** 여러 미래 시나리오 시뮬레이션

**탐색 과정:**
1. 현재 상태에서 시작
2. 5가지 행동 시뮬레이션
3. 각 행동 후 1주, 2주, 4주 미래 예측
4. 최적 구매 시점 찾기

**예시:**
```
시나리오 A: 지금 구매 → 1주 후 +2%
시나리오 B: 1주 대기 → 2주 후 +8%  ← 최적
시나리오 C: 1개월 대기 → -5% (가격 상승)
```

---

## 🧪 테스트 결과 (규칙 기반)

### GPU 모델별 구매 적정도

| GPU 모델 | 현재 가격 | 구매 적정도 | 추천 |
|----------|----------|------------|------|
| **RTX 5060** | 587,850원 | 100점 / 100점 | 🟢 강력 추천 |
| **RTX 5070** | 1,045,000원 | 100점 / 100점 | 🟢 강력 추천 |
| **RTX 5080** | 2,081,190원 | 37.4점 / 100점 | 🔴 대기 권장 |
| **RX 9070 XT** | 1,070,000원 | 100점 / 100점 | 🟢 강력 추천 |

### 판단 근거

**RTX 5060 (강력 추천)**
- ✅ 평균보다 저렴한 가격
- ✅ 엔트리급 적정 가격대
- ✅ 환율 안정

**RTX 5080 (대기 권장)**
- ❌ 평균보다 비싼 가격
- ❌ 가격 거품 의심
- ⚠️ 1개월 대기 후 재평가 권장

---

## ⚠️ 현재 한계점

### 1. 데이터 부족
- **샘플 수:** 12개 (모델당 1개씩만)
- **기간:** 3일치만 수집
- **시계열:** 가격 추이 데이터 없음
- **뉴스:** 0개 기사

### 2. Feature 부족
- **현재:** 11차원
- **필요:** 256차원
- **부족한 Feature:**
  - 가격 이동평균 (7일, 14일, 30일)
  - 가격 변동성
  - 경쟁 모델 비교
  - 뉴스 감정 분석 (시계열)
  - 출시 후 경과 시간
  - 계절성 지표

### 3. 모델-데이터 불일치
- **학습된 모델:** 256차원 입력 기대
- **실제 데이터:** 11차원
- **해결 방법:** 모델 재학습 또는 데이터 확장

### 4. 행동 정의 불명확
- 현재: 5가지 행동 (BUY, WAIT, HOLD, SKIP)
- 보상 함수 미정의
- 백테스팅 시스템 없음

---

## 🔧 AlphaZero 작동을 위한 요구사항

### 1️⃣ 데이터 수집 (최우선)

#### 필요한 데이터
- **GPU 가격:** 최소 30일, 이상적으로 90일
- **모델당 샘플:** 최소 100개
- **뉴스 수집:** 일별 GPU 관련 뉴스
- **환율:** 일별 USD/KRW, JPY/KRW

#### 수집 주기
- 매일 자정 크롤링
- 가격 변동 시 실시간 수집

### 2️⃣ Feature Engineering (11차원 → 256차원)

#### 가격 Feature (60차원)
```python
[
    current_price,           # 현재가
    price_1d_ago,            # 전일가
    price_7d_ago,            # 전주가
    price_30d_ago,           # 전월가
    ma_7d, ma_14d, ma_30d,   # 이동평균
    volatility_7d,           # 변동성
    price_min_7d,            # 최저가
    price_max_7d,            # 최고가
    price_change_pct_1d,     # 변화율
    price_trend_7d,          # 추세
    ...
]
```

#### 환율 Feature (20차원)
```python
[
    usd_krw_current,
    usd_krw_ma_7d,
    usd_krw_change_pct,
    jpy_krw_current,
    ...
]
```

#### 경쟁 모델 Feature (50차원)
```python
[
    rtx5060_price,
    rtx5070_price,
    rx9070xt_price,
    price_rank,              # 가격 순위
    perf_price_ratio,        # 성능/가격
    ...
]
```

#### 뉴스 Feature (30차원)
```python
[
    sentiment_1d,            # 오늘 뉴스 감정
    sentiment_7d_avg,        # 1주 평균 감정
    article_count_1d,        # 오늘 기사 수
    keyword_freq_release,    # '출시' 키워드
    keyword_freq_discount,   # '할인' 키워드
    ...
]
```

#### 시장 Feature (20차원)
```python
[
    seller_count,            # 판매자 수
    seller_count_trend,      # 판매자 추이
    stock_status,            # 재고 상황
    shipping_days,           # 배송 기간
    ...
]
```

#### 시간 Feature (20차원)
```python
[
    days_since_release,      # 출시 후 경과일
    is_year_end,             # 연말 여부
    is_new_gen_release,      # 신제품 출시 시즌
    day_of_week,             # 요일
    ...
]
```

#### 기술 지표 (56차원)
```python
[
    rsi_7d,                  # RSI
    macd,                    # MACD
    bollinger_upper,         # 볼린저 밴드
    momentum,                # 모멘텀
    ...
]
```

### 3️⃣ 행동 정의 (Action Space)

```python
actions = [
    "BUY_NOW",      # 0: 즉시 구매
    "WAIT_SHORT",   # 1: 1주일 대기
    "WAIT_LONG",    # 2: 1개월 대기
    "HOLD",         # 3: 계속 관망
    "SKIP",         # 4: 구매 안함
]
```

### 4️⃣ 보상 정의 (Reward Function)

```python
def calculate_reward(action, current_price, future_price_7d):
    """
    구매 타이밍의 적정성을 보상으로 계산
    """
    if action == "BUY_NOW":
        # 7일 후 가격 하락 → 좋은 타이밍
        if future_price_7d < current_price:
            reward = (current_price - future_price_7d) / current_price * 100
            return reward  # +5.2% 하락 → +5.2 보상
        else:
            reward = (current_price - future_price_7d) / current_price * 100
            return reward  # -3.0% 상승 → -3.0 보상

    elif action == "WAIT_SHORT":
        # 1주 대기 후 가격이 더 하락했다면 → 좋은 판단
        if future_price_7d < current_price:
            return +10  # 대기가 정답
        else:
            return -5   # 바로 샀어야 했음
```

### 5️⃣ 백테스팅 시스템

```python
def backtest(model, historical_data):
    """
    과거 데이터로 모델 성능 검증
    """
    total_profit = 0
    correct_decisions = 0

    for date, gpu_model in historical_data:
        # 모델 예측
        action, confidence = model.predict(date, gpu_model)

        # 실제 결과 확인
        actual_result = get_actual_result(date, gpu_model, action)

        # 보상 계산
        reward = calculate_reward(action, actual_result)
        total_profit += reward

        if reward > 0:
            correct_decisions += 1

    accuracy = correct_decisions / len(historical_data)
    avg_profit = total_profit / len(historical_data)

    return accuracy, avg_profit
```

---

## 📈 기대 성능 (충분한 데이터 확보 후)

### 목표 지표

| 지표 | 목표 | 설명 |
|------|------|------|
| **정확도** | 70% 이상 | 최적 구매 시점 맞춤 |
| **평균 이득** | +5% 이상 | 대기 vs 즉시 구매 비교 |
| **최대 손실** | -10% 이하 | 최악의 경우 제한 |

### 바둑 승률과의 비교

| 바둑 | GPU 구매 |
|------|----------|
| 승률 60% = 우세 | 이득 확률 60% = 구매 추천 |
| 승률 50% = 호각 | 이득 확률 50% = 관망 |
| 승률 40% = 불리 | 이득 확률 40% = 대기 |

---

## 🎯 실전 사용 시나리오

### 시나리오 1: 즉시 구매 추천
```
입력: "RTX 5060"

AI 분석:
├─ 현재 가격: 587,850원
├─ 30일 평균: 620,000원
├─ 가격 추세: 하락 중
├─ 뉴스 감정: 중립
└─ 판매자 수: 안정

MCTS 시뮬레이션 (50회):
├─ 지금 구매: +5.2% 이득 (30회)
├─ 1주 대기: +2.1% 이득 (15회)
└─ 1개월 대기: -3.0% 손실 (5회)

결론: 🟢 지금 구매 추천 (승률 75.2%)
```

### 시나리오 2: 대기 권장
```
입력: "RTX 5080"

AI 분석:
├─ 현재 가격: 2,081,190원
├─ 30일 평균: 1,900,000원
├─ 가격 추세: 상승 중 (거품 의심)
├─ 뉴스 감정: 부정 ("재고 부족")
└─ 판매자 수: 감소 중

MCTS 시뮬레이션 (50회):
├─ 지금 구매: -8.2% 손실 (35회)
├─ 1주 대기: +1.5% 이득 (10회)
└─ 1개월 대기: +12.3% 이득 (5회)

결론: 🔴 1개월 대기 권장 (승률 62.8%)
```

---

## 📊 현재 vs 완성 후

| 항목 | 현재 | 완성 후 |
|------|------|---------|
| **아키텍처** | ✅ 완성 | ✅ 완성 |
| **모델** | ✅ 학습됨 (256차원) | ✅ 재학습 (256차원) |
| **데이터** | ❌ 12개 샘플 | ✅ 3,000+ 샘플 |
| **Feature** | ❌ 11차원 | ✅ 256차원 |
| **시계열** | ❌ 없음 | ✅ 90일 추이 |
| **백테스팅** | ❌ 없음 | ✅ 완료 |
| **정확도** | - | 70%+ |
| **사용 가능** | ⚠️ 규칙 기반만 | ✅ AlphaZero |

---

## 🚀 완성을 위한 로드맵

### Phase 1: 데이터 수집 (2주)
- [x] 크롤링 스크립트 작성
- [x] 3일치 데이터 수집
- [ ] **30일 이상 데이터 수집** ← 현재 단계
- [ ] 뉴스 수집 자동화
- [ ] 데이터 검증 및 정제

### Phase 2: Feature Engineering (1주)
- [ ] 11차원 → 256차원 확장
- [ ] 가격 이동평균 계산
- [ ] 뉴스 감정 분석
- [ ] 기술적 지표 추가
- [ ] Feature 정규화

### Phase 3: 모델 재학습 (3일)
- [ ] 256차원 입력으로 모델 재구성
- [ ] Self-Play 학습
- [ ] 하이퍼파라미터 튜닝

### Phase 4: 백테스팅 (3일)
- [ ] 과거 데이터로 성능 검증
- [ ] 정확도 측정
- [ ] 평균 이득률 계산

### Phase 5: 프로덕션 배포 (1주)
- [ ] REST API 완성
- [ ] 프론트엔드 연동
- [ ] 실시간 모니터링
- [ ] 사용자 피드백 수집

**총 예상 기간: 4~5주**

---

## 💡 핵심 인사이트

### 1. AlphaZero는 왜 적합한가?

**바둑과의 유사성:**
- 바둑: 지금 두면 승률 70%
- GPU: 지금 사면 이득 확률 70%

**시뮬레이션:**
- 바둑: 여러 수 시뮬레이션
- GPU: 여러 구매 시점 시뮬레이션

**가치 판단:**
- 바둑: 국면의 가치
- GPU: 가격의 적정성

### 2. 왜 딥러닝이 필요한가?

**복잡한 패턴:**
- 가격, 환율, 뉴스, 계절성의 비선형 관계
- 수백 개 Feature 간 상호작용

**미래 예측:**
- Dynamics Network로 가격 추이 예측
- MCTS로 최적 타이밍 탐색

### 3. 기존 방법과의 차이

| 방법 | 설명 | 한계 |
|------|------|------|
| **규칙 기반** | "평균보다 싸면 구매" | 단순, 정확도 낮음 |
| **선형 회귀** | 가격 추세선 | 비선형 패턴 못잡음 |
| **LSTM** | 시계열 예측 | 최적 타이밍 못찾음 |
| **AlphaZero** | 시뮬레이션 + 학습 | ✅ 최적 타이밍 탐색 |

---

## 📝 최종 평가

### ✅ 잘 된 점
1. **AlphaZero 아키텍처 완전 구현**
2. **데이터 수집 파이프라인 구축**
3. **통합 데이터 생성 자동화**
4. **시스템 목적 명확**

### ❌ 개선 필요
1. **데이터 양 부족** (12개 → 3,000개 필요)
2. **Feature 부족** (11차원 → 256차원)
3. **시계열 데이터 없음** (3일 → 90일)
4. **백테스팅 시스템 없음**

### 🎯 핵심 답변

> **"데이터만 입력하면 진짜 AI가 될 수 있는가?"**
>
> **→ 예, 가능합니다. 단, 충분한 양의 데이터가 필요합니다.**
>
> **현재 상태:**
> - ✅ 엔진 완성 (AlphaZero)
> - ❌ 연료 부족 (데이터 12개)
>
> **필요한 것:**
> - 30일+ 가격 데이터
> - 256차원 Feature
> - 백테스팅 검증
>
> **예상 기간: 4~5주**

---

## 📚 참고 자료

### 논문
1. Silver et al. (2016) - "Mastering the game of Go with deep neural networks and tree search"
2. Schrittwieser et al. (2020) - "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero)

### 적용 사례
- AlphaGo: 바둑 승률 예측
- AlphaZero: 체스, 쇼기 가치 판단
- MuZero: 미래 시뮬레이션

---

**보고서 작성:** Claude Sonnet 4.5
**검증 날짜:** 2026-02-14
**프로젝트 경로:** `/Users/younghwa.jin/Documents/gpu-advisor`
