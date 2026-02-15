# GPU Advisor: AI 핵심 모델 코드 심층 분석 (인덱스)

이 문서는 프로젝트의 심장부인 `backend/models/` 폴더 내 AI 모델들의 **코드 레벨 심층 분석** 시리즈의 목차입니다.

각 문서에는 실제 코드, 데이터 흐름도, 출력 예시, 그리고 다른 모듈과의 연결 관계가 상세히 기술되어 있습니다.

---

## 📊 전체 흐름

```
[시장 데이터 22D]
      ↓
  ①  h (Representation)  →  Latent State 256D
      ↓                              ↓
  ②  f (Prediction)            ③  g (Dynamics)
      ↓                              ↓
  Policy + Value             미래 State + Reward
      ↓                              ↓
      └──────── ④ MCTS ──────────────┘
                   ↓
            GPU 구매 추천 점수
```

---

## 📚 상세 문서 목록

| # | 파일 | 문서 | 핵심 내용 |
|---|------|------|----------|
| 01 | `representation_network.py` | [01_representation_network.md](models/01_representation_network.md) | PositionalEncoding 수학 원리, FeedForward 확장-축소 구조, 22D→256D 변환 과정 |
| 02 | `dynamics_network.py` | [02_dynamics_network.md](models/02_dynamics_network.md) | Action One-hot 인코딩, GELU vs ReLU 비교, 보상 분포(μ,σ²) 해석, 연쇄 호출 예시 |
| 03 | `prediction_network.py` | [03_prediction_network.md](models/03_prediction_network.md) | Dual-Head 구조 분석, Softmax→Policy 변환, MCTS에서의 Prior/Value 역할 |
| 04 | `mcts.py` + `mcts_engine.py` | [04_mcts_engine.md](models/04_mcts_engine.md) | PUCT 공식 숫자 예시, Selection→Expansion→Backpropagation 추적, MCTSTrainer Self-play |
| 05 | `transformer_model.py` | [05_transformer_model.md](models/05_transformer_model.md) | Multi-Head Attention 연산 과정, KV Cache 속도 비교, MPS 가속, Xavier 초기화 |

---

## 💡 읽는 순서 추천

1. **01 → 02 → 03** (세 네트워크의 역할 이해)
2. **04** (세 네트워크가 MCTS에서 어떻게 협업하는지)
3. **05** (고성능 Transformer 엔진의 내부 구조)
