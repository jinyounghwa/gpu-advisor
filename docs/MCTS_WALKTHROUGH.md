# MCTS Numerical Walkthrough

This document demonstrates how the GPU Advisor's MCTS (Monte Carlo Tree Search) works with **actual numbers**.

## Overall Flow

```
Root State (RTX 4090's 256D latent vector)
    │
    ├── [Simulation 1] Select → Expand → Rollout → Backup
    ├── [Simulation 2] Select → Expand → Rollout → Backup
    ├── ...
    └── [Simulation 50] Select → Backup
    │
    └── Visit-count-based action probability computation
```

## Initial State

Assume we run MCTS for RTX 4090.

```
Root Node:
  State: s_0 = [0.073, 0.068, 0.065, ...] (256D latent vector)
  Visit count: N = 0
  Value sum: W = 0
  Children: None (unexpanded)
```

## Simulation 1: First Expansion

### Step 1: Selection

Root is unexpanded, so we proceed directly to expansion.

### Step 2: Expansion

Prediction Network f(s_0) outputs policy and value:

```
f(s_0) → Policy logits: [-0.23, 0.45, 0.12, -0.08, -0.31]
         → softmax → P = [0.128, 0.253, 0.182, 0.149, 0.118]
                         BUY   WAIT_S WAIT_L HOLD   SKIP

         Value: v = 0.15
```

Five child nodes are created:

```
Root (N=0, W=0)
├── [BUY_NOW]    Prior P=0.158, N=0, W=0
├── [WAIT_SHORT] Prior P=0.311, N=0, W=0  ← highest prior
├── [WAIT_LONG]  Prior P=0.224, N=0, W=0
├── [HOLD]       Prior P=0.183, N=0, W=0
└── [SKIP]       Prior P=0.125, N=0, W=0
```

### Step 3: Rollout (Simulation)

5-step rollout from WAIT_SHORT child (highest prior):

```
Step 1: State s_0 → f(s_0) → sample action → BUY_NOW
        g(s_0, BUY_NOW) → s_1, reward r_1 = +0.012

Step 2: State s_1 → f(s_1) → sample action → HOLD
        g(s_1, HOLD) → s_2, reward r_2 = -0.003

Step 3: State s_2 → f(s_2) → sample action → WAIT_SHORT
        g(s_2, WAIT_SHORT) → s_3, reward r_3 = +0.008

Step 4: State s_3 → f(s_3) → sample action → BUY_NOW
        g(s_3, BUY_NOW) → s_4, reward r_4 = +0.005

Step 5: State s_4 → f(s_4) → sample action → HOLD
        g(s_4, HOLD) → s_5, reward r_5 = -0.001

Terminal value: f(s_5) → v_5 = 0.12
```

Total return (discount γ = 0.99):

```
V = r_1 + γ×r_2 + γ²×r_3 + γ³×r_4 + γ⁴×r_5 + γ⁵×v_5
  = 0.012 - 0.00297 + 0.00784 + 0.00490 - 0.00096 + 0.11416
  = 0.1350
```

### Step 4: Backup

Propagate value 0.1350 up the path:

```
WAIT_SHORT node: N = 0→1, W = 0→0.1350, Q = 0.1350
Root node:       N = 0→1, W = 0→0.1350, Q = 0.1350
```

## Simulation 2: UCB-Based Selection

### Step 1: UCB Score Calculation

```
UCB(a) = Q(a) + c × P(a) × √(ln(N_parent + 1) / (1 + N_child))

Root visits N_parent = 1, c = √2 ≈ 1.414

BUY_NOW:    Q=0     + 1.414 × 0.158 × √(ln(2)/1) = 0 + 0.186 = 0.186
WAIT_SHORT: Q=0.135 + 1.414 × 0.311 × √(ln(2)/2) = 0.135 + 0.259 = 0.394
WAIT_LONG:  Q=0     + 1.414 × 0.224 × √(ln(2)/1) = 0 + 0.264 = 0.264
HOLD:       Q=0     + 1.414 × 0.183 × √(ln(2)/1) = 0 + 0.215 = 0.215
SKIP:       Q=0     + 1.414 × 0.125 × √(ln(2)/1) = 0 + 0.147 = 0.147

* Unvisited nodes (N=0) have UCB = ∞, so they are explored first
→ BUY_NOW selected (one of the unvisited nodes)
```

> **Key Point**: Unvisited nodes have UCB = ∞, so the first 5 simulations explore each action once.

### Rollout & Backup

5-step rollout from BUY_NOW → total return 0.0820

```
BUY_NOW node: N = 0→1, W = 0→0.082, Q = 0.082
Root node:    N = 1→2, W = 0.135→0.217, Q = 0.108
```

## Simulations 3–5: Exploring Remaining Actions

```
Simulation 3: WAIT_LONG → rollout → V = 0.1120 → backup
Simulation 4: HOLD      → rollout → V = 0.0450 → backup
Simulation 5: SKIP      → rollout → V = 0.0280 → backup
```

Tree after 5 simulations:

```
Root (N=5, Q=0.080)
├── [BUY_NOW]    N=1, Q=0.082
├── [WAIT_SHORT] N=1, Q=0.135  ← current best
├── [WAIT_LONG]  N=1, Q=0.112
├── [HOLD]       N=1, Q=0.045
└── [SKIP]       N=1, Q=0.028
```

## Simulation 6: Full UCB-Based Search

```
UCB scores (N_parent = 5):

BUY_NOW:    0.082 + 1.414 × 0.158 × √(ln(6)/2) = 0.082 + 0.237 = 0.319
WAIT_SHORT: 0.135 + 1.414 × 0.311 × √(ln(6)/2) = 0.135 + 0.467 = 0.602 ← max
WAIT_LONG:  0.112 + 1.414 × 0.224 × √(ln(6)/2) = 0.112 + 0.337 = 0.449
HOLD:       0.045 + 1.414 × 0.183 × √(ln(6)/2) = 0.045 + 0.275 = 0.320
SKIP:       0.028 + 1.414 × 0.125 × √(ln(6)/2) = 0.028 + 0.188 = 0.216

→ WAIT_SHORT selected (both Q-value and prior are high)
```

## After Simulation 50: Final Tree

```
Root (N=50, Q=0.095)
├── [BUY_NOW]    N=8,  Q=0.078  → prob 16%
├── [WAIT_SHORT] N=18, Q=0.142  → prob 36%  ← most visited
├── [WAIT_LONG]  N=12, Q=0.118  → prob 24%
├── [HOLD]       N=7,  Q=0.052  → prob 14%
└── [SKIP]       N=5,  Q=0.031  → prob 10%
```

## Action Probability Computation

Visit counts converted via temperature (τ):

```
π(a) = N(a)^(1/τ) / Σ N(a')^(1/τ)

τ = 1.0 (default):
π = [8, 18, 12, 7, 5] / 50 = [0.16, 0.36, 0.24, 0.14, 0.10]

τ = 0 (greedy):
π = [0, 1, 0, 0, 0]  (WAIT_SHORT only)
```

## Post-Calibration Final Decision

MCTS policy is blended with other signals:

```
MCTS policy:   [0.16, 0.36, 0.24, 0.14, 0.10]
Reward policy: [0.15, 0.30, 0.25, 0.18, 0.12]
Prior:         [0.20, 0.25, 0.22, 0.18, 0.15]
Utility bias:  [0.10, 0.28, 0.30, 0.17, 0.15]

Calibrated = 0.45×MCTS + 0.25×Reward + 0.15×Prior + 0.15×Utility
           = [0.148, 0.318, 0.249, 0.157, 0.114]

→ Final action: WAIT_SHORT (31.8%)
→ Confidence: 0.318
→ Entropy: 1.52
```

## Safety Check

```
Confidence 0.318 ≥ 0.25 (min threshold) → PASS ✓
Entropy 1.52 ≤ 1.58 (max threshold) → PASS ✓

→ Safe mode: OFF
→ Final action: WAIT_SHORT (unchanged)
```

## Key Takeaways

1. **UCB balances exploration and exploitation**: Sum of Q-value (exploitation) and visit-count bonus (exploration)
2. **Unvisited nodes have UCB = ∞**: First 5 simulations try each action once
3. **Visit count = action probability**: More visits → better action (intuition)
4. **Rollout depth 5**: Dynamics Network simulates 5 days into the future
5. **Policy calibration**: MCTS alone can collapse to one action; blending with other signals prevents this

---

**Related code**: `backend/models/mcts_engine.py` — `MCTSEngine.search()` method
