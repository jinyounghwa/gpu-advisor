# Transformer Model 구조 해설

> **파일 위치:** `backend/models/transformer_model.py`
> **역할:** Multi-Head Attention과 KV Cache 옵션을 갖춘 Transformer 기반 Policy-Value 모델 구현
> **운영 기준 참고:** 현재 GPU 구매 에이전트의 주 경로는 `h/g/f + mcts_engine`이며, 이 파일은 독립 실험/보조 추론 경로에서 사용됩니다.
> **모델 규모 참고:** 기본 설정(`input_dim=11, d_model=256, num_layers=6, num_actions=2`) 기준 약 **5.27M parameters**입니다.

---

## 1. 전체 아키텍처 구조

```
[시장 시퀀스 데이터]
  (batch, seq_len, 11)    ← seq_len 일자의 시장 지표 11개
          ↓
┌─────────────────────────────────────────┐
│       PolicyValueNetwork                │
│                                         │
│  input_embedding (11 → 256)             │
│          ↓                              │
│  PositionalEncoding (위치 정보 주입)       │
│          ↓                              │
│  ┌─── TransformerBlock #1 ────┐         │
│  │  MultiHeadAttention(8-head)│         │
│  │  + Residual + LayerNorm    │         │
│  │  FeedForward (256→1024→256)│         │
│  │  + Residual + LayerNorm    │         │
│  └────────────────────────────┘         │
│          ↓                              │
│  TransformerBlock #2                    │
│          ↓                              │
│  TransformerBlock #3                    │
│          ↓                              │
│  TransformerBlock #4                    │
│          ↓                              │
│  TransformerBlock #5                    │
│          ↓                              │
│  TransformerBlock #6                    │
│          ↓                              │
│  last_token = x[:, -1, :]               │
│  (마지막 시점의 특징만 추출)                  │
│          ↓                              │
│  ┌───────┴───────┐                      │
│  ↓               ↓                      │
│ Policy Head    Value Head               │
│ (256→1024→2)   (256→1024→1)             │
│ + Softmax      + Tanh                   │
└─────────────────────────────────────────┘
       ↓               ↓
  action_probs      state_value
  (batch, 2)        (batch, 1)
```

---

## 2. 핵심 구성요소 분석

### 2.1 `MultiHeadAttention` — 데이터 간 관계를 동시에 파악

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dropout=0.1, use_kv_cache=False):
        super().__init__()
        assert d_model % num_heads == 0  # 256 % 8 = 0 ✓

        self.d_model = d_model       # 256
        self.num_heads = num_heads   # 8
        self.d_k = d_model // num_heads  # 256 / 8 = 32 (각 헤드의 차원)

        # Q, K, V 변환 행렬 (각각 256 → 256)
        self.W_q = nn.Linear(d_model, d_model)  # Query: "내가 찾고 싶은 것"
        self.W_k = nn.Linear(d_model, d_model)  # Key:   "내가 가진 라벨"
        self.W_v = nn.Linear(d_model, d_model)  # Value: "실제 내용"
        self.W_o = nn.Linear(d_model, d_model)  # 출력 합성
```

**Multi-Head의 의미 (8개 헤드):**
```
Head 1: "가격 추세와 환율의 관계"에 주목
Head 2: "뉴스 감정과 가격 변동의 관계"에 주목
Head 3: "거래량과 가격의 관계"에 주목
Head 4: "과거 1주와 현재의 관계"에 주목
Head 5~8: 각기 다른 패턴에 주목
```
→ 8개의 서로 다른 관점에서 동시에 데이터를 분석합니다.

---

### 2.2 `forward()` 내부 — Attention 연산 과정

```python
def forward(self, x, mask=None, kv_cache=None):
    batch_size, seq_len, _ = x.size()
    # x shape: (4, 10, 256) — 배치 4개, 시퀀스 10일, 특징 256개

    # ① Q, K, V 생성
    Q = self.W_q(x)  # (4, 10, 256)
    K = self.W_k(x)  # (4, 10, 256)
    V = self.W_v(x)  # (4, 10, 256)

    # ② 8개 헤드로 분리
    Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    # (4, 10, 256) → (4, 10, 8, 32) → (4, 8, 10, 32)
    #                                    ↑  ↑  ↑   ↑
    #                               배치 헤드 시간 특징

    # ③ KV Cache 적용 (이전 계산 결과 재사용)
    if self.use_kv_cache and kv_cache is not None:
        cached_k, cached_v = kv_cache
        K = torch.cat([cached_k, K], dim=2)  # 이전 K에 현재 K를 이어붙임
        V = torch.cat([cached_v, V], dim=2)
    # → 이전 9일치 결과를 재계산하지 않고 캐시에서 가져옴

    # ④ Attention Score 계산
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    # Q(4,8,10,32) × K^T(4,8,32,10) = scores(4,8,10,10)
    # scores[i][h][t1][t2] = "시점 t1이 시점 t2에 얼마나 주목하는가?"
    #
    # √32 ≈ 5.66 으로 나누는 이유:
    # 차원이 클수록 내적값이 커져서 softmax가 극단적으로 됨
    # 스케일링으로 부드러운 확률 분포를 만듦

    # ⑤ 마스킹 (선택사항)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # -10억은 softmax 후 사실상 0 → 특정 위치를 "못 보게" 함

    # ⑥ Softmax → Attention Weights
    attn_weights = F.softmax(scores, dim=-1)
    # 각 행의 합 = 1.0
    # 예: 10일째가 [0.01, 0.02, ..., 0.03, 0.70, 0.15, 0.05]
    #     → 최근 3일에 가장 집중

    attn_weights = self.dropout(attn_weights)

    # ⑦ Weighted Sum
    output = torch.matmul(attn_weights, V)
    # attn(4,8,10,10) × V(4,8,10,32) = output(4,8,10,32)
    # "가장 관련 있는 시점들의 정보를 가중 합산"

    # ⑧ 헤드 합치기
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    # (4, 8, 10, 32) → (4, 10, 8, 32) → (4, 10, 256)

    # KV Cache 업데이트
    new_kv_cache = (K, V) if self.use_kv_cache else None

    return self.W_o(output), new_kv_cache
    # W_o: 8개 헤드의 결과를 하나의 표현으로 통합
```

**Attention의 실제 동작 예시:**
```
시퀀스: [1일전, 2일전, 3일전, ..., 10일전]

10일째(오늘)의 attention weights:
  1일전: 0.01  ← 너무 옛날, 관심 없음
  5일전: 0.05
  8일전: 0.10  ← 가격 급락한 날, 좀 관심
  9일전: 0.30  ← 어제, 매우 중요
 10일전: 0.40  ← 오늘, 가장 중요
```

---

### 2.3 `TransformerBlock` — Attention + FeedForward + Residual

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=8, d_ff=1024, dropout=0.1):
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),     # 256 → 1024
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),     # 1024 → 256
        )

    def forward(self, x, mask=None, kv_cache=None):
        # ① Attention + Residual Connection
        attn_output, layer_kv_cache = self.attention(x, mask, kv_cache)
        x = self.norm1(x + self.dropout(attn_output))
        #              ↑ Residual: 원본 x를 더함
        # 왜? "Attention이 찾은 새로운 관계" + "원래 가지고 있던 정보" 모두 보존

        # ② FeedForward + Residual Connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        #              ↑ Residual: 또 원본을 더함

        return x, new_kv_cache
```

**Residual Connection의 효과:**
```
Without Residual: f(x)          → 깊은 네트워크에서 그래디언트 소실
With Residual:    x + f(x)      → 최소한 x는 보존되므로 학습 안정적
```

---

### 2.4 `PolicyValueNetwork` — 최종 네트워크 조립

```python
class PolicyValueNetwork(nn.Module):
    def __init__(
        self,
        input_dim=11,          # 입력: 11개 시장 지표
        d_model=256,           # 내부 차원
        num_heads=8,           # Attention 헤드 수
        num_layers=6,          # Transformer 블록 수
        d_ff=1024,             # FeedForward 확장 차원
        num_actions=2,         # 행동 수 (Buy / Hold)
        dropout=0.1,
        use_kv_cache=False,
    ):
        # 입력 임베딩: 11 → 256
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=1000)

        # 6개의 Transformer 블록
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, use_kv_cache)
            for _ in range(num_layers)
        ])

        # Policy Head: "사야 하나? 기다려야 하나?"
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_ff),      # 256 → 1024
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_actions),   # 1024 → 2
        )

        # Value Head: "지금 상황이 얼마나 유리한가?"
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_ff),      # 256 → 1024
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),             # 1024 → 1
        )

        # Xavier 초기화: 학습 시작점을 최적화
        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Xavier 초기화: 가중치의 분산을
        # √(2 / (fan_in + fan_out)) 으로 설정
        # → 깊은 네트워크에서도 그래디언트가 사라지지 않음
```

---

### 2.5 `forward()` — 시퀀스 → 판단

```python
def forward(self, x, kv_cache=None):
    batch_size, seq_len, _ = x.size()
    # x: (4, 10, 11) — 4개 GPU, 10일치, 11개 지표

    # ① 임베딩: 11차원 → 256차원
    x = self.input_embedding(x)         # (4, 10, 256)
    x = self.positional_encoding(x)     # 위치 정보 추가

    # ② 6개 Transformer 블록 순차 통과
    new_kv_cache = {} if self.use_kv_cache else None
    for i, block in enumerate(self.transformer_blocks):
        layer_cache = kv_cache.get(f"layer_{i}") if kv_cache else None
        x, layer_new_cache = block(x, None, layer_cache)
        if self.use_kv_cache:
            new_kv_cache[f"layer_{i}"] = layer_new_cache

    # ③ 마지막 시점의 특징만 추출
    last_token = x[:, -1, :]           # (4, 256)
    # 왜 마지막만? → 10일치 전체 정보가 attention을 통해
    #                 마지막 토큰에 요약되어 있으므로

    # ④ Policy: 행동 확률
    policy = self.policy_head(last_token)   # (4, 2)
    policy = F.softmax(policy, dim=-1)       # 확률로 변환
    # 예: [0.73, 0.27] → "구매 73%, 대기 27%"

    # ⑤ Value: 상태 가치
    value = self.value_head(last_token)     # (4, 1)
    value = torch.tanh(value)                # -1 ~ +1 범위
    # 예: 0.58 → "꽤 유리한 상황"

    return policy, value, new_kv_cache
```

---

## 3. KV Cache 메커니즘 상세

KV Cache는 **추론 속도를 획기적으로 높이는** 기법입니다.

```
=== KV Cache 없이 (매번 전체 재계산) ===
시점 1: [D1] → Q,K,V 계산 1회
시점 2: [D1,D2] → Q,K,V 계산 2회  ← D1을 또 계산!
시점 3: [D1,D2,D3] → Q,K,V 계산 3회  ← D1,D2 또 계산!
총 연산: 1+2+3 = 6회

=== KV Cache 사용 (이전 결과 재활용) ===
시점 1: [D1] → Q,K,V 계산 1회 → K₁,V₁ 캐시 저장
시점 2: [D2] → Q₂ 계산 + 캐시의 K₁,V₁ 재사용 → 1회만
시점 3: [D3] → Q₃ 계산 + 캐시의 K₁₂,V₁₂ 재사용 → 1회만
총 연산: 1+1+1 = 3회  (50% 절감!)
```

```python
# 실제 코드에서의 KV Cache 흐름
if self.use_kv_cache and kv_cache is not None:
    cached_k, cached_v = kv_cache
    K = torch.cat([cached_k, K], dim=2)  # 이전 K들 + 새 K
    V = torch.cat([cached_v, V], dim=2)  # 이전 V들 + 새 V
# → Q는 현재 시점만, K와 V는 전체 히스토리
```

---

## 4. 디바이스 가속(MPS/CPU)

```python
def create_model(config: dict) -> PolicyValueNetwork:
    model = PolicyValueNetwork(...)

    # Mac M4의 GPU 가속 자동 감지
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = model.to(device)

    print(f"Model created with {model.count_parameters():,} parameters")
    print(f"Device: {device}")
    return model
```

성능 수치는 하드웨어/배치 크기/시퀀스 길이/드라이버 버전에 따라 크게 달라지므로, 본 문서에서는 고정 TPS를 단정하지 않습니다.
실측이 필요하면 동일 입력 조건으로 `backend/inference/engine.py`의 벤치마크 경로에서 측정해야 합니다.

---

## 5. 실행 테스트

```python
config = {
    "input_dim": 11,
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 1024,
    "num_actions": 2,
}

model = create_model(config)

# 4개 GPU, 10일치 데이터, 11개 지표
x = torch.randn(4, 10, 11).to(device)
policy, value, _ = model(x)
```

**출력 결과:**
```
Model created with 5,271,043 parameters  ← 기본 설정 기준 약 527만 개
Device: mps                               ← Mac GPU 가속 활성화

Policy shape: torch.Size([4, 2])          ← 4개 샘플에 대한 2개 행동 확률
Value shape:  torch.Size([4, 1])          ← 4개 GPU의 상태 가치
```

---

## 6. 다른 모델과의 관계

```
이 파일의 PolicyValueNetwork는 독립적인 "올인원" 모델입니다.

다른 모델 구조:
  h (Representation) → g (Dynamics) → f (Prediction)
  별도의 3개 네트워크가 분업하는 MuZero 스타일

이 모델 (transformer_model.py):
  시계열 데이터를 직접 입력받아 한 번에 Policy + Value를 출력
  Attention으로 시간적 관계를 자동으로 학습
  Attention 기반 시계열 정책/가치 추론 실험 경로
```

→ 이 파일은 독립 Transformer 경로를 위한 구현이며, 운영 경로와는 별도로 검증/실험할 수 있습니다.
