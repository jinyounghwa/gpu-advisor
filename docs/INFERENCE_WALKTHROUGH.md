# Inference Walkthrough: From Input to Final Recommendation

Traces the complete pipeline from a GPU model name input to the final purchase recommendation.

## End-to-End Flow

```
User Input: "RTX 4090"
    │
    ▼
[1] Model Name Resolution (Fuzzy Matching)
    │   "RTX 4090" → "RTX 4090" (exact match)
    ▼
[2] Load State Vector (256D)
    │   From training_data_2026-02-21.json
    ▼
[3] Latent State Encoding
    │   RepresentationNetwork: 22D → 256D
    ▼
[4] MCTS Planning Search (50 simulations)
    │   Action probabilities + root value
    ▼
[5] Expected Reward Computation
    │   DynamicsNetwork: reward per action
    ▼
[6] Policy Calibration
    │   4-signal blend + anti-collapse regularization
    ▼
[7] Safety Gate Check
    │   Confidence/entropy gates
    ▼
[8] Response Generation
    │   Action + explanation + agent trace
    ▼
JSON Response
```

## Step 1: Model Name Resolution

```python
query = "RTX 4090"

# Exact match (case-insensitive)
"rtx 4090" == "RTX 4090".lower() → matched!

# On failure, fuzzy matching:
# "4090" → substring search → "RTX 4090"
# "RTX4090" → difflib.get_close_matches(cutoff=0.4) → "RTX 4090"
```

**Code**: `gpu_purchase_agent.py:_resolve_gpu_model()`

## Step 2: Load State Vector

Loads the 256D vector from the latest training dataset.

```python
# File: data/processed/dataset/training_data_2026-02-21.json
{
  "date": "2026-02-21",
  "gpu_model": "RTX 4090",
  "state_vector": [0.234, 0.228, 0.225, 0.230, -0.002, 0.001, ...]  # 256 values
}
```

Vector layout:

```
Index  0-59:    Price features (normalized price, MA7, MA14, MA30, change rates, volatility)
Index 60-79:    Exchange features (USD/KRW, JPY/KRW, EUR/KRW normalized)
Index 80-109:   News features (sentiment, article count, positive/negative ratio)
Index 110-129:  Market features (seller count, stock status)
Index 130-149:  Time features (day of week, month, year-end flag)
Index 150-255:  Technical indicators (RSI, MACD, momentum, padding)
```

**Code**: `gpu_purchase_agent.py:_get_state_vector()`

## Step 3: Latent State Encoding

Representation Network h(s) transforms market state into latent space.

```
Input:  state_vector = [0.234, 0.228, ...] (22D — checkpoint input dim)
        ↓
        Linear(22, 256) + LayerNorm
        ↓
        PositionalEncoding
        ↓
        FeedForward × 3 (256 → 512 → 256)
        ↓
        LayerNorm + Linear(256, 256)
        ↓
Output: latent = [0.451, -0.123, 0.872, ...] (256D)
```

**Key insight**: 22D raw market data is transformed into a rich 256D representation. This latent vector compresses price trends, market sentiment, and macroeconomic conditions into a single vector.

**Code**: `representation_network.py:RepresentationNetwork.forward()`

## Step 4: MCTS Planning Search

Runs 50 simulations with the latent vector as root.

```
Root state: latent (256D)
Simulations: 50
Rollout depth: 5 steps
Exploration constant: c = √2

Output:
  mcts_probs = [0.16, 0.36, 0.24, 0.14, 0.10]
                BUY   WAIT_S WAIT_L HOLD   SKIP
  root_value = 0.095
```

See [MCTS_WALKTHROUGH.md](MCTS_WALKTHROUGH.md) for detailed numerical trace.

**Code**: `mcts_engine.py:MCTSEngine.search()`

## Step 5: Expected Reward Computation

Dynamics Network computes expected reward for each action.

```python
for action in [BUY_NOW, WAIT_SHORT, WAIT_LONG, HOLD, SKIP]:
    action_onehot = one_hot(action, dim=5)
    _, reward_mean, _ = dynamics_network(latent, action_onehot)

Results:
  BUY_NOW:    reward = -0.008  (slight loss expected)
  WAIT_SHORT: reward = +0.015  (small gain expected) ← max
  WAIT_LONG:  reward = +0.010  (small gain)
  HOLD:       reward = +0.002  (neutral)
  SKIP:       reward = -0.003  (opportunity cost)
```

**Code**: `gpu_purchase_agent.py:decide_from_state()` reward computation loop

## Step 6: Policy Calibration

Blends 4 signals and applies anti-collapse regularization.

```
MCTS policy:   [0.16, 0.36, 0.24, 0.14, 0.10]  (planning)
Reward policy: [0.12, 0.32, 0.26, 0.17, 0.13]  (reward-based)
Prior:         [0.20, 0.25, 0.22, 0.18, 0.15]  (data distribution)
Utility bias:  [0.10, 0.28, 0.30, 0.17, 0.15]  (market common sense)

Blend = 0.45×[0.16,...] + 0.25×[0.12,...] + 0.15×[0.20,...] + 0.15×[0.10,...]
      = [0.143, 0.319, 0.252, 0.157, 0.117]

Minimum 2% floor → no change (all > 0.02)
Normalize → [0.145, 0.323, 0.255, 0.159, 0.118]

Entropy = 1.49 ≥ 0.65 → no anti-collapse needed
```

**Code**: `gpu_purchase_agent.py:decide_from_state()` calibration logic

## Step 7: Safety Gate Check

```
Top action: WAIT_SHORT (0.323)
Confidence: 0.323

Confidence 0.323 ≥ 0.25 (min) → PASS ✓
Entropy 1.49 ≤ 1.58 (max) → PASS ✓

Safe mode: OFF
Final action: WAIT_SHORT (unchanged)
```

**Code**: `gpu_purchase_agent.py:decide_from_state()` safety logic

## Step 8: Response Generation

```python
AgentDecision(
    gpu_model = "RTX 4090",
    action = "WAIT_SHORT",
    raw_action = "WAIT_SHORT",
    confidence = 0.323,
    entropy = 1.49,
    value = 0.095,
    action_probs = {
        "BUY_NOW": 0.145, "WAIT_SHORT": 0.323,
        "WAIT_LONG": 0.255, "HOLD": 0.159, "SKIP": 0.118
    },
    expected_rewards = {
        "BUY_NOW": -0.008, "WAIT_SHORT": 0.015,
        "WAIT_LONG": 0.010, "HOLD": 0.002, "SKIP": -0.003
    },
    date = "2026-02-21",
    simulations = 50,
    safe_mode = False,
    safe_reason = None
)
```

## API Response

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
    "action_probs": {"BUY_NOW": 0.145, "WAIT_SHORT": 0.323, "...": "..."},
    "expected_rewards": {"BUY_NOW": -0.008, "WAIT_SHORT": 0.015, "...": "..."}
  }
}
```

## Latency Breakdown

| Stage | Time (CPU) | Share |
|-------|-----------|-------|
| Name resolution | ~1ms | 0.2% |
| Vector load | ~5ms | 1.0% |
| Latent encoding | ~10ms | 2.0% |
| **MCTS search** | **~450ms** | **90%** |
| Reward computation | ~25ms | 5.0% |
| Calibration + safety | ~1ms | 0.2% |
| Response build | ~1ms | 0.2% |
| **Total** | **~500ms** | **100%** |

> MCTS accounts for 90% of inference time. Reducing simulations improves speed but degrades decision quality.

---

**Related code**:
- Full flow: `backend/simple_server.py:ask_gpu()`
- Agent decision: `backend/agent/gpu_purchase_agent.py:decide()`
- MCTS search: `backend/models/mcts_engine.py:MCTSEngine.search()`
