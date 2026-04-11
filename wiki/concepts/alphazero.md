# AlphaZero / MuZero 아키텍처

> 마지막 업데이트: 2026-04-11

## 개요

GPU Advisor는 DeepMind의 MuZero 아키텍처를 기반으로 GPU 구매 타이밍을 최적화합니다. 4개의 신경망이 협력하여 의사결정을 수행합니다.

## 4개 신경망

### Representation Network h(s)
- 256D 시장 상태 → 256D 잠재 상태 인코딩
- 3× FeedForward with residual connections, GELU activation
- ~6.4M 파라미터

### Dynamics Network g(s,a)
- 현재 잠재 상태 + 행동 → 다음 잠재 상태 + Gaussian 보상 (μ, log σ²)
- 4개 residual blocks
- ~6.5M 파라미터

### Prediction Network f(s)
- 잠재 상태 → 정책 로짓 (5개 행동) + 가치 스칼라
- 4개 residual blocks
- ~6.0M 파라미터

### Action Model a(z)
- 잠재 상태 → 학습된 행동 사전
- ActionEmbedding 16D + ActionPriorNetwork 256→5
- ~43K 파라미터

## 정책 캘리브레이션

최종 정책은 4개 신호의 가중합:

| 신호 | 가중치 |
|------|--------|
| MCTS 방문 수 | 60% |
| Reward signal | 20% |
| f-net prior | 10% |
| ActionModel prior | 10% |

## 5개 행동

| 행동 | 의미 |
|------|------|
| BUY_NOW | 즉시 구매 |
| WAIT_SHORT | 단기 대기 (1-7일) |
| WAIT_LONG | 장기 대기 (7-30일) |
| HOLD | 보유 유지 |
| SKIP | 구매 건너뜀 |

## 학습

- **5-loss joint optimization**:
  - Latent loss (1.0) — World Model 일관성
  - Policy loss (1.0) — 정책 학습
  - Value loss (1.0) — 가치 추정
  - Reward NLL loss (0.5) — Gaussian 보상
  - Action prior loss (0.3) — Action Model 지도학습

## 관련 페이지

- [[concepts/mcts]] — MCTS 탐색 엔진
- [[concepts/feature_engineering]] — 256D 입력 생성
- [[overview]] — 프로젝트 개요

---

[[index|← 인덱스로 돌아가기]]
