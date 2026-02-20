# Safety Mechanisms

This document explains the multi-layer safety system that protects the GPU Advisor agent from making dangerous decisions.

## Safety Architecture

```
MCTS Policy Output
    │
    ▼
[Layer 1] Policy Calibration
    │   Blend MCTS + Reward + Prior + Utility signals
    ▼
[Layer 2] Anti-Collapse Regularizer
    │   Enforce minimum entropy
    ▼
[Layer 3] Confidence/Entropy Gate
    │   Uncertain → force HOLD
    ▼
[Layer 4] Release Quality Gates
    │   Block deployment if overall performance fails
    ▼
Final Action Output
```

## Layer 1: Policy Calibration

### Problem

Using MCTS alone can cause **mode collapse** — visits concentrate on a single action.

### Solution

Blend 4 signals with weighted combination:

```python
calibrated = 0.45 × MCTS_policy + 0.25 × reward_policy + 0.15 × prior + 0.15 × utility_bias
```

| Signal | Weight | Role | Source |
|--------|--------|------|--------|
| MCTS policy | 0.45 | Core planning-based decision | `mcts_engine.search()` |
| Reward policy | 0.25 | Expected reward correction | `dynamics_network.forward()` |
| Prior probability | 0.15 | Training data distribution | Checkpoint `action_prior` |
| Utility bias | 0.15 | Observable market signals | Price trends, moving averages |

### Utility Bias Logic

```python
# Price above moving average → suppress BUY_NOW, promote WAIT
over_ma = max(current_price - 0.5 × (MA7 + MA14), -0.2)

# Upward trend → promote waiting
trend_up = max(change_1d + 0.5 × change_7d, 0)

# Downward trend → promote buying
trend_down = max(-(change_1d + 0.5 × change_7d), 0)
```

This "common sense" prevents the model from ignoring basic market principles.

### Minimum Probability Floor

```python
calibrated = np.maximum(calibrated, 0.02)  # All actions get at least 2%
calibrated = calibrated / calibrated.sum()  # Normalize
```

**Why 2%?** No action should ever have 0% probability. 2% ensures every action can be selected even under deterministic sampling.

## Layer 2: Anti-Collapse Regularizer

### Problem

Even after calibration, if entropy is too low (one action dominates), the agent cannot adapt to diverse market conditions.

### Solution

```python
min_entropy_target = 0.65
current_entropy = -Σ(p × log(p))

if current_entropy < min_entropy_target:
    alpha = min(0.55, (target - current) / target)
    calibrated = (1 - alpha) × calibrated + alpha × prior_policy
```

| Entropy Value | Meaning | Action |
|---------------|---------|--------|
| 0.0 | Single action at 100% (full collapse) | Strong regularization |
| 0.65 | Minimum target | Regularization trigger |
| 1.0 | Moderate diversity | Normal |
| 1.61 | Uniform over 5 actions (maximum) | Normal |

### Why 0.65?

- Maximum entropy for 5 actions: ln(5) ≈ 1.609
- 0.65 is ~40% of maximum → at least 2–3 actions must have meaningful probability
- Too high → indecisive; too low → mode collapse

## Layer 3: Confidence/Entropy Gate

### How It Works

```python
if confidence < 0.25:       # Top action probability below 25%
    action = "HOLD"          # → Force observe
    safe_mode = True
    safe_reason = "low_confidence"

elif entropy > 1.58:         # Entropy exceeds 98% of maximum
    action = "HOLD"          # → Force observe
    safe_mode = True
    safe_reason = "high_entropy"
```

### Threshold Rationale

**Minimum confidence = 0.25**

```
Uniform probability over 5 actions: 1/5 = 0.20
Minimum confidence:                 0.25

0.25 requires 25% higher probability than uniform.
"At minimum, this action should be somewhat better than others."
```

**Maximum entropy = 1.58**

```
Maximum entropy (uniform): ln(5) = 1.609
Threshold:                 1.58 = 1.609 × 98%

If entropy is ≥98% of maximum, all actions have near-equal probability → no meaningful decision.
```

### Why HOLD?

- **HOLD is the safest action**: Neither buying nor avoiding — just observing
- **Best under uncertainty**: When the agent isn't confident, doing nothing is optimal
- **Go analogy**: In Go, when the position is unclear, the best move is often a safe, neutral one

## Layer 4: Release Quality Gates

The entire agent is backtested, and deployment is blocked if overall performance fails.

### 7 Gates

| Gate | Threshold | Meaning |
|------|-----------|---------|
| `accuracy_raw` | ≥ 0.55 | Directional accuracy above 55% (statistically significant vs 50% random) |
| `reward_raw` | > 0.0 | Average reward is positive (must make money) |
| `abstain` | ≤ 0.85 | Abstain ratio below 85% (too passive = useless) |
| `safe_override` | ≤ 0.90 | Safety override below 90% (too many overrides = agent is meaningless) |
| `action_entropy_raw` | ≥ 0.25 | Minimum action diversity |
| `uplift_raw_vs_buy` | ≥ 0.0 | Must beat "always buy" strategy |
| `no_mode_collapse_raw` | True | Fail if any action exceeds 95% |

### Gate Logic

```python
all_pass = all(gates.values())

if all_pass:
    status = "pass"       # Deployment candidate
else:
    status = "blocked"    # Needs data/training/policy adjustment
```

### Baseline Comparisons

The agent must outperform 3 naive strategies:

```
1. "Always Buy" strategy: BUY_NOW every day → market average return
2. "Always Wait" strategy: WAIT_SHORT every day → inverse market return
3. "Always Hold" strategy: HOLD every day → minimal loss (fees only)

Agent reward > max(always_buy, always_wait, always_hold) → PASS
```

## Safety Flow Examples

### Case 1: Normal Operation

```
Input: RTX 5070
MCTS → Calibration → [BUY_NOW: 0.42, WAIT_SHORT: 0.28, ...]
Confidence: 0.42 ≥ 0.25 ✓
Entropy: 1.35 ≤ 1.58 ✓
→ Final: BUY_NOW (unchanged)
```

### Case 2: Low Confidence → Safe Mode

```
Input: RTX 5090
MCTS → Calibration → [BUY_NOW: 0.22, WAIT_SHORT: 0.21, WAIT_LONG: 0.20, ...]
Confidence: 0.22 < 0.25 ✗
→ Safe mode activated
→ Final: HOLD (forced override)
→ Reason: "low_confidence<0.25"
```

### Case 3: High Entropy → Safe Mode

```
Input: RX 7600
MCTS → Calibration → [0.205, 0.198, 0.202, 0.200, 0.195]
Entropy: 1.609 > 1.58 ✗ (near-uniform distribution)
→ Safe mode activated
→ Final: HOLD (forced override)
→ Reason: "high_entropy>1.58"
```

## Training-Time Safety

Safety mechanisms also operate during training:

| Mechanism | Location | Role |
|-----------|----------|------|
| Gradient clipping | `fine_tuner.py` | Prevents gradient explosion (max_norm=1.0) |
| Entropy bonus | Loss function | Maintains policy diversity (-0.001 × entropy) |
| Prior regularization | Loss function | Prevents distribution drift (0.02 × KL) |
| Class weights | `cross_entropy` | Corrects imbalanced action distribution |

---

**Related code**:
- Inference safety: `backend/agent/gpu_purchase_agent.py` — `decide_from_state()`
- Release gates: `backend/agent/release_pipeline.py` — `quality_gates()`
- Training safety: `backend/agent/fine_tuner.py` — `_train_step()`
