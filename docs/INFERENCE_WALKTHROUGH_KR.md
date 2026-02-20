# 추론 워크스루: 입력부터 최종 추천까지

GPU 모델명을 입력하면 어떤 과정을 거쳐 최종 추천이 나오는지 전체 파이프라인을 추적합니다.

## 전체 흐름

```
사용자 입력: "RTX 4090"
    │
    ▼
[1] 모델명 해석 (Fuzzy Matching)
    │   "RTX 4090" → "RTX 4090" (정확 매칭)
    ▼
[2] 상태 벡터 로드 (256D)
    │   training_data_2026-02-21.json에서 로드
    ▼
[3] 잠재 상태 인코딩
    │   RepresentationNetwork: 22D → 256D
    ▼
[4] MCTS 계획 탐색 (50회 시뮬레이션)
    │   행동 확률 + 루트 가치 출력
    ▼
[5] 기대 보상 계산
    │   DynamicsNetwork: 각 행동의 보상 예측
    ▼
[6] 정책 보정
    │   4개 신호 혼합 + 반붕괴 정규화
    ▼
[7] 안전장치 확인
    │   신뢰도/엔트로피 게이트
    ▼
[8] 최종 응답 생성
    │   행동 + 설명 + 에이전트 트레이스
    ▼
JSON 응답 반환
```

## 1단계: 모델명 해석

```python
# 사용자 입력
query = "RTX 4090"

# 정확 매칭 시도 (대소문자 무시)
"rtx 4090" == "RTX 4090".lower() → 매칭!

# 매칭 실패 시 퍼지 매칭:
# "4090" → 부분 문자열 검색 → "RTX 4090" 반환
# "RTX4090" → difflib.get_close_matches(cutoff=0.4) → "RTX 4090" 반환
```

**관련 코드**: `gpu_purchase_agent.py:_resolve_gpu_model()`

## 2단계: 상태 벡터 로드

최신 학습 데이터 파일에서 해당 GPU의 256차원 벡터를 로드합니다.

```python
# 파일: data/processed/dataset/training_data_2026-02-21.json
{
  "date": "2026-02-21",
  "gpu_model": "RTX 4090",
  "state_vector": [0.234, 0.228, 0.225, 0.230, -0.002, 0.001, ...]  # 256개 값
}
```

벡터 구성:

```
인덱스  0-59:   가격 특징 (정규화 가격, MA7, MA14, MA30, 변화율, 변동성, ...)
인덱스 60-79:   환율 특징 (USD/KRW, JPY/KRW, EUR/KRW 정규화)
인덱스 80-109:  뉴스 특징 (감정 점수, 기사 수, 긍정/부정 비율)
인덱스 110-129: 시장 특징 (판매자 수, 재고 상태)
인덱스 130-149: 시간 특징 (요일, 월, 연말 여부)
인덱스 150-255: 기술 지표 (RSI, MACD, 모멘텀, 패딩)
```

**관련 코드**: `gpu_purchase_agent.py:_get_state_vector()`

## 3단계: 잠재 상태 인코딩

Representation Network h(s)가 시장 상태를 잠재 공간으로 변환합니다.

```
입력:  state_vector = [0.234, 0.228, ...] (22D — 체크포인트의 입력 차원)
       ↓
       Linear(22, 256) + LayerNorm
       ↓
       PositionalEncoding
       ↓
       FeedForward × 3 (256 → 512 → 256)
       ↓
       LayerNorm + Linear(256, 256)
       ↓
출력:  latent = [0.451, -0.123, 0.872, ...] (256D)
```

**핵심**: 22차원의 원시 시장 데이터가 256차원의 풍부한 표현으로 변환됩니다. 이 잠재 벡터에는 가격 추세, 시장 심리, 거시 경제 상황이 모두 압축되어 있습니다.

**관련 코드**: `representation_network.py:RepresentationNetwork.forward()`

## 4단계: MCTS 계획 탐색

잠재 벡터를 루트로 50회 시뮬레이션을 실행합니다.

```
루트 상태: latent (256D)
시뮬레이션 횟수: 50
롤아웃 깊이: 5단계
탐험 상수: c = √2

출력:
  mcts_probs = [0.16, 0.36, 0.24, 0.14, 0.10]
                BUY   WAIT_S WAIT_L HOLD   SKIP
  root_value = 0.095
```

자세한 수치 과정은 [MCTS_WALKTHROUGH_KR.md](MCTS_WALKTHROUGH_KR.md)를 참조하세요.

**관련 코드**: `mcts_engine.py:MCTSEngine.search()`

## 5단계: 기대 보상 계산

각 행동에 대해 Dynamics Network가 예상 보상을 계산합니다.

```python
for action in [BUY_NOW, WAIT_SHORT, WAIT_LONG, HOLD, SKIP]:
    action_onehot = one_hot(action, dim=5)
    _, reward_mean, _ = dynamics_network(latent, action_onehot)

결과:
  BUY_NOW:    reward = -0.008  (약간 손해 예상)
  WAIT_SHORT: reward = +0.015  (소폭 이득 예상) ← 최대
  WAIT_LONG:  reward = +0.010  (소폭 이득)
  HOLD:       reward = +0.002  (중립)
  SKIP:       reward = -0.003  (기회비용)
```

**관련 코드**: `gpu_purchase_agent.py:decide_from_state()` 내 보상 계산 루프

## 6단계: 정책 보정

4개 신호를 혼합하고, 반붕괴 정규화를 적용합니다.

```
MCTS 정책:   [0.16, 0.36, 0.24, 0.14, 0.10]  (계획 기반)
보상 정책:   [0.12, 0.32, 0.26, 0.17, 0.13]  (보상 기반)
사전 확률:   [0.20, 0.25, 0.22, 0.18, 0.15]  (데이터 분포)
효용 편향:   [0.10, 0.28, 0.30, 0.17, 0.15]  (시장 상식)

혼합 = 0.45×[0.16,...] + 0.25×[0.12,...] + 0.15×[0.20,...] + 0.15×[0.10,...]
     = [0.143, 0.319, 0.252, 0.157, 0.117]

최소 2% 보장 → 변화 없음 (모두 > 0.02)
정규화 → [0.145, 0.323, 0.255, 0.159, 0.118]

엔트로피 = 1.49 ≥ 0.65 → 반붕괴 정규화 불필요
```

**관련 코드**: `gpu_purchase_agent.py:decide_from_state()` 내 보정 로직

## 7단계: 안전장치 확인

```
최고 확률 행동: WAIT_SHORT (0.323)
신뢰도: 0.323

신뢰도 0.323 ≥ 0.25 (최소) → 통과 ✓
엔트로피 1.49 ≤ 1.58 (최대) → 통과 ✓

안전 모드: OFF
최종 행동: WAIT_SHORT (원본 유지)
```

**관련 코드**: `gpu_purchase_agent.py:decide_from_state()` 내 안전장치 로직

## 8단계: 최종 응답 생성

```python
AgentDecision(
    gpu_model = "RTX 4090",
    action = "WAIT_SHORT",          # 최종 행동
    raw_action = "WAIT_SHORT",      # 안전장치 전 행동
    confidence = 0.323,             # 최종 신뢰도
    entropy = 1.49,                 # 정책 엔트로피
    value = 0.095,                  # MCTS 루트 가치
    action_probs = {                # 보정된 행동 확률
        "BUY_NOW": 0.145,
        "WAIT_SHORT": 0.323,
        "WAIT_LONG": 0.255,
        "HOLD": 0.159,
        "SKIP": 0.118
    },
    expected_rewards = {            # 행동별 기대 보상
        "BUY_NOW": -0.008,
        "WAIT_SHORT": 0.015,
        "WAIT_LONG": 0.010,
        "HOLD": 0.002,
        "SKIP": -0.003
    },
    date = "2026-02-21",
    simulations = 50,
    safe_mode = False,
    safe_reason = None
)
```

## API 응답 형태

```json
{
  "title": "RTX 4090",
  "summary": "AI Agent Decision: WAIT_SHORT",
  "specs": "Confidence 32.3% | Value 0.095",
  "usage": "MCTS 50회 계획 탐색 | 데이터 기준일 2026-02-21",
  "recommendation": "단기 대기가 더 유리하다고 판단했습니다. (신뢰도 32.3%, 상태가치 0.095, MCTS 50회)",
  "agent_trace": {
    "selected_action": "WAIT_SHORT",
    "raw_action": "WAIT_SHORT",
    "confidence": 0.323,
    "entropy": 1.49,
    "value": 0.095,
    "safe_mode": false,
    "safe_reason": null,
    "action_probs": {"BUY_NOW": 0.145, "WAIT_SHORT": 0.323, ...},
    "expected_rewards": {"BUY_NOW": -0.008, "WAIT_SHORT": 0.015, ...}
  }
}
```

## 처리 시간 분석

| 단계 | 소요 시간 (CPU) | 비율 |
|------|----------------|------|
| 모델명 해석 | ~1ms | 0.2% |
| 벡터 로드 | ~5ms | 1.0% |
| 잠재 인코딩 | ~10ms | 2.0% |
| **MCTS 탐색** | **~450ms** | **90%** |
| 보상 계산 | ~25ms | 5.0% |
| 보정 + 안전장치 | ~1ms | 0.2% |
| 응답 생성 | ~1ms | 0.2% |
| **총계** | **~500ms** | **100%** |

> MCTS가 전체 추론 시간의 90%를 차지합니다. 시뮬레이션 횟수를 줄이면 속도가 향상되지만 결정 품질이 저하됩니다.

---

**관련 코드**:
- 전체 흐름: `backend/simple_server.py:ask_gpu()`
- 에이전트 결정: `backend/agent/gpu_purchase_agent.py:decide()`
- MCTS 탐색: `backend/models/mcts_engine.py:MCTSEngine.search()`
