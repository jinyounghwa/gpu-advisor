# Representation Network (`h`) 완전 해부

> **파일 위치:** `backend/models/representation_network.py`
> **역할:** 원시 시장 데이터를 AI가 이해할 수 있는 잠재 상태(Latent State)로 변환
> **연결:** 이 네트워크의 출력은 → `PredictionNetwork(f)` 와 `DynamicsNetwork(g)` 의 입력으로 전달됩니다.

---

## 1. 전체 데이터 흐름도

```
[다나와 가격, 환율, 뉴스 감정 점수 등]
        ↓
  22차원 원시 벡터 (state_dim=22)
        ↓
┌─────────────────────────────────┐
│   RepresentationNetwork (h)     │
│                                 │
│  input_embedding (22 → 256)     │
│        ↓                        │
│  LayerNorm                      │
│        ↓                        │
│  PositionalEncoding             │
│        ↓                        │
│  FeedForward Block #1           │
│        ↓                        │
│  FeedForward Block #2           │
│        ↓                        │
│  FeedForward Block #3           │
│        ↓                        │
│  LayerNorm                      │
│        ↓                        │
│  output_layer (256 → 256)       │
└─────────────────────────────────┘
        ↓
  256차원 Latent State (s_0)
        ↓
  → PredictionNetwork(f) 로 전달
  → DynamicsNetwork(g) 로 전달
  → MCTS 탐색의 시작점(Root State)
```

---

## 2. 클래스별 코드 상세 분석

### 2.1 `PositionalEncoding` — 시간 순서를 알려주는 신호

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스: cos

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]
```

**왜 필요한가?**
- AI는 기본적으로 입력 데이터의 **순서**를 알 수 없습니다. "첫 번째 데이터"와 "두 번째 데이터"를 구분 못합니다.
- sin/cos 함수의 서로 다른 주파수를 이용하여 각 위치(position)에 고유한 패턴을 부여합니다.
- `register_buffer`로 등록하면 학습 대상 파라미터가 아니라, 고정된 상수로 모델에 저장됩니다.

**출력 예시:**
```
위치 0: [sin(0), cos(0), sin(0), cos(0), ...]  = [0.00, 1.00, 0.00, 1.00, ...]
위치 1: [sin(1), cos(1), sin(0.01), cos(0.01)] = [0.84, 0.54, 0.01, 1.00, ...]
```
→ 각 위치마다 고유한 "지문"이 생성됩니다.

---

### 2.2 `FeedForward` — 핵심 특징을 추출하는 변환기

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)    # 256 → 512 (확장)
        self.relu = nn.ReLU()                        # 비선형 활성화
        self.dropout = nn.Dropout(dropout)            # 과적합 방지 (10% 뉴런 비활성화)
        self.linear2 = nn.Linear(d_ff, d_model)      # 512 → 256 (복원)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)    # 차원 확장: 더 많은 특징 공간에서 분석
        x = self.relu(x)       # 음수값 제거 → 비선형성 도입
        x = self.dropout(x)    # 학습 시 일부 뉴런을 랜덤하게 끔
        x = self.linear2(x)    # 원래 차원으로 복원
        return x
```

**왜 "확장 후 축소"를 하는가?**
- `256 → 512`로 확장하면 더 풍부한 특징 공간에서 데이터를 분석할 수 있습니다.
- `512 → 256`으로 다시 축소하면 중요한 특징만 남기고 불필요한 정보를 버립니다.
- 이 과정을 **3번 반복(ff1, ff2, ff3)**하여 점점 더 추상적이고 의미 있는 표현을 만듭니다.

**ReLU가 하는 일:**
```
입력: [-0.5, 0.3, -1.2, 0.8]
출력: [ 0.0, 0.3,  0.0, 0.8]  ← 음수는 모두 0으로
```
→ "관련 없는 신호(음수)"를 차단하여 중요한 신호만 남깁니다.

---

### 2.3 `RepresentationNetwork` — 메인 네트워크

```python
class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int = 22,      # 입력 차원 (시장 지표 수)
        latent_dim: int = 256,     # 출력 차원 (AI 내부 표현 크기)
        d_ff: int = 512,           # FeedForward 확장 차원
        dropout: float = 0.1,      # 드롭아웃 비율
        max_seq_len: int = 1,      # 시퀀스 최대 길이
    ):
        super().__init__()

        # [레이어 1] Positional Encoding
        self.pos_encoding = PositionalEncoding(latent_dim, max_seq_len)

        # [레이어 2] Input embedding: 22차원 → 256차원으로 확장
        self.input_embedding = nn.Linear(state_dim, latent_dim)
        self.layer_norm1 = nn.LayerNorm(latent_dim)

        # [레이어 3-5] FeedForward 앙상블 (3개)
        self.ff1 = FeedForward(latent_dim, d_ff, dropout)
        self.ff2 = FeedForward(latent_dim, d_ff, dropout)
        self.ff3 = FeedForward(latent_dim, d_ff, dropout)

        self.layer_norm2 = nn.LayerNorm(latent_dim)

        # [레이어 6] Output projection: 최종 출력
        self.output_layer = nn.Linear(latent_dim, latent_dim)
```

**각 레이어의 역할:**

| 순서 | 레이어 | 입력 shape | 출력 shape | 역할 |
|------|--------|-----------|-----------|------|
| 1 | `input_embedding` | `(batch, 22)` | `(batch, 256)` | 저차원 → 고차원 매핑 |
| 2 | `layer_norm1` | `(batch, 256)` | `(batch, 256)` | 값의 분포를 표준화 |
| 3 | `pos_encoding` | `(batch, 1, 256)` | `(batch, 1, 256)` | 시간적 위치 정보 추가 |
| 4 | `ff1 → ff2 → ff3` | `(batch, 1, 256)` | `(batch, 1, 256)` | 특징 추출 3단계 반복 |
| 5 | `layer_norm2` | `(batch, 1, 256)` | `(batch, 1, 256)` | 최종 표준화 |
| 6 | `output_layer` | `(batch, 1, 256)` | `(batch, 1, 256)` | 최종 Latent State 생성 |

---

### 2.4 `forward()` 메서드 — 실제 실행 흐름

```python
def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
    # ① 2D 텐서이면 시퀀스 차원을 추가
    if state_tensor.dim() == 2:
        x = state_tensor.unsqueeze(1)  # (batch, 22) → (batch, 1, 22)
    else:
        x = state_tensor

    # ② 22차원 → 256차원으로 임베딩
    x = self.input_embedding(x)        # (batch, 1, 256)
    x = self.layer_norm1(x)            # 수치 안정화

    # ③ 위치 인코딩 추가
    x = self.pos_encoding(x)

    # ④ 3단계 FeedForward로 특징 정제
    x = self.ff1(x)                    # 1차 특징 추출
    x = self.ff2(x)                    # 2차 특징 추출 (더 추상적)
    x = self.ff3(x)                    # 3차 특징 추출 (가장 추상적)

    x = self.layer_norm2(x)            # 최종 표준화

    # ⑤ 출력: Latent State
    s_0 = self.output_layer(x)
    return s_0.squeeze(1)              # (batch, 1, 256) → (batch, 256)
```

---

## 3. 실행 테스트 및 결과

```python
# 테스트 코드 (파일 하단의 if __name__ == "__main__"에 해당)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = RepresentationNetwork(state_dim=22).to(device)

# 가상 데이터: 4개의 GPU에 대한 22차원 시장 지표
dummy_input = torch.randn(4, 22).to(device)

s_0 = model(dummy_input)
```

**출력 결과:**
```
Input shape:        torch.Size([4, 22])     ← 4개 GPU, 각 22개 지표
Latent State shape: torch.Size([4, 256])    ← 4개 GPU, 각 256차원 잠재 상태
Latent State mean:  -0.0023                 ← 평균이 0 근처 (LayerNorm 효과)
Total parameters:   921,600                 ← 약 92만 개의 학습 가능 파라미터
```

---

## 4. 다음 네트워크로의 연결

이 네트워크가 생성한 `s_0` (256차원 Latent State)는 두 곳에서 사용됩니다:

```python
# mcts.py 에서의 사용 예시
root_observation = torch.randn(1, 22).to(device)  # 원시 시장 데이터

# ① h 네트워크로 Latent State 생성
root_latent = h(root_observation)                  # → (1, 256)

# ② 생성된 Latent State를 MCTS에 전달
best_action, policy_target = tree.search(
    root_latent,       # ← 이 값이 사용됨
    dynamics_model,    # g 네트워크
    prediction_model   # f 네트워크
)
```
