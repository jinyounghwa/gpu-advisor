# Hyperparameter Design Rationale

This document explains every key hyperparameter in the GPU Advisor project and the reasoning behind each choice.

## 1. Latent Dimension: 256D

| Item | Value |
|------|-------|
| Input dimension | 22D (market state) |
| Latent space | 256D |
| Expansion ratio | ~11.6× |

**Why 256?**

- **Information capacity**: 256D is sufficient to compress 6 feature categories (price 60D, exchange 20D, news 30D, market 20D, time 20D, technical 106D) into a single vector.
- **Power of two**: 2^8 = 256 is optimal for GPU tensor operations — memory alignment and SIMD instructions benefit from powers of two.
- **MuZero reference**: The original MuZero used 256 channels for Go (19×19 board = 361 intersections). Our input (22D) is much smaller, so 256D is more than adequate.
- **128 vs 256 vs 512**:
  - 128D: Potential bottleneck for 6 feature categories
  - 256D: ~42D allocation per category, good expressiveness
  - 512D: 2× parameters but risk of overfitting on 22D input

## 2. MCTS Simulations: 50

| Item | Value |
|------|-------|
| Simulations | 50 |
| Actions | 5 |
| Avg. visits per action | 10 |

**Why 50?**

- **Relative to action space**: AlphaGo uses 1,600 simulations for 362 actions (~4.4 per action). We use 50 for 5 actions (10 per action) — **richer exploration per action than AlphaGo**.
- **Inference speed**: 50 sims × 5 rollout steps = 250 neural network calls. ~0.5s on CPU, suitable for real-time API.
- **Diminishing returns**: With only 5 actions, accuracy gains plateau after ~30 simulations.

```
Simulations vs estimated quality:
10  → 2 per action, insufficient exploration
30  → 6 per action, basic coverage
50  → 10 per action, stable search ← chosen
100 → 20 per action, 2× latency, marginal gain
```

## 3. Action Space: 5 Actions

| Action | Meaning | Design Rationale |
|--------|---------|-----------------|
| BUY_NOW | Buy immediately | Price dip detected |
| WAIT_SHORT | Wait ~1 week | Slight decline expected |
| WAIT_LONG | Wait ~1 month | Significant decline expected |
| HOLD | Observe | High uncertainty |
| SKIP | Skip this cycle | Avoid purchase entirely |

**Why 5?**

- **3 actions (buy/wait/hold)**: Too simple. Cannot express "how long to wait"
- **5 actions**: Covers both time axis (now/short/medium) and intensity axis (buy/wait/skip)
- **7+ actions**: No practical differentiation for GPU purchase decisions
- **Go analogy**: In Go, a move is "place stone at position X". In GPU buying, a move is "act at time T". The natural temporal granularity has 5 levels.

## 4. Exploration Constant (UCB): c = √2 ≈ 1.414

```
UCB(s, a) = Q(s, a) + c × P(s, a) × √(ln(N_parent) / (1 + N_child))
```

**Why √2?**

- **UCB1 theory**: Auer et al. (2002) proved √2 is the optimal exploration constant for multi-armed bandits.
- **Exploration-exploitation tradeoff**:
  - c = 1.0: Exploitation bias, gets stuck on early good actions
  - c = √2: Theoretically optimal, sufficient exploration of all actions
  - c = 2.0: Over-exploration, slow convergence
- **AlphaGo**: Uses PUCT variant with c_puct = 1.5–2.5. Our value falls within this range.

## 5. Rollout Depth: 5 Steps

| Item | Value |
|------|-------|
| Rollout depth | 5 |
| Discount factor | 0.99 |
| Effective discount at step 5 | 0.99^5 ≈ 0.951 |

**Why 5 steps?**

- **Time interpretation**: Each step = 1 day. 5 steps = simulating 5 days ahead.
- **GPU price dynamics**: Major price patterns emerge on a weekly (5 business days) basis.
- **Error accumulation**: Dynamics Network prediction error compounds at each step. Beyond 5 steps, predictions become unreliable.
- **Compute budget**: Rollout depth × simulations = total network calls. 5 × 50 = 250, suitable for real-time inference.

## 6. Network Architecture

### Hidden Dimension: 512D

```
Input 256D → Hidden 512D (×4 blocks) → Output 256D/5D/1D
```

- **2× expansion**: 512D = 2× the 256D input provides sufficient nonlinear capacity.
- **4× FFN**: Each block internally expands 512 → 2048 → 512 (Transformer convention).
- **Parameter budget**: 18.9M total parameters distributed evenly across 3 networks (~6M each).

### Layer Count: 4 Blocks

- **Depth vs width**: 4 blocks × 512D has more expressive power than 2 blocks × 1024D.
- **Gradient flow**: 4 blocks is stable for training even without residual connections.
- **MuZero reference**: Original uses 16 ResNet blocks, but our 22D input is much simpler, so 4 blocks suffice.

## 7. Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-4 | Stable convergence range for AdamW |
| Batch size | 32 | Efficient sampling for ~168 transition samples |
| Training steps | 500 | Prevents overfitting on small dataset |
| Weight decay | 1e-5 | L2 regularization against overfitting |
| Gradient clipping | 1.0 | Training stability |
| Entropy coefficient | 0.001 | Maintains policy diversity without disrupting learning |
| Prior regularization | 0.02 | Prevents policy from drifting too far from data distribution |

### Learning Rate 1e-4

```
1e-3: Unstable training, loss oscillation
1e-4: Stable convergence within 500 steps ← chosen
1e-5: Too slow, needs 5000+ steps
```

### Batch Size 32

- Total data: ~24 GPU models × days (~168 transition samples)
- Batch 32 samples ~19% of the dataset each step
- Large batches on small data → overfitting; small batches → excessive noise

## 8. Policy Calibration Weights

```python
calibrated = 0.45 × MCTS_policy + 0.25 × reward_policy + 0.15 × prior + 0.15 × utility_bias
```

| Source | Weight | Role |
|--------|--------|------|
| MCTS policy | 0.45 | Core decision (planning-based) |
| Reward policy | 0.25 | Expected reward correction |
| Prior probability | 0.15 | Data distribution regularization |
| Utility bias | 0.15 | Observable feature-based common sense |

**Why 0.45 for MCTS?**

- MCTS alone (1.0) risks mode collapse (100% on one action)
- 0.45 gives MCTS the leading role while allowing other signals to calibrate
- Weights sum to 1.0, guaranteeing a valid probability distribution

## 9. Safety Thresholds

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| Min confidence | 0.25 | Must be higher than uniform probability (1/5 = 0.20) |
| Max entropy | 1.58 | 98% of maximum entropy ln(5) ≈ 1.609 |
| Min action entropy | 0.25 | Prevents mode collapse (0 = single action only) |
| Max abstain ratio | 0.85 | Agent becomes useless if it abstains >85% of the time |
| Min accuracy | 0.55 | Must be statistically significant above random (50%) |

See [SAFETY_MECHANISMS.md](SAFETY_MECHANISMS.md) for detailed safety mechanism documentation.

## 10. Discount Factor: γ = 0.99

- **Meaning**: Present value ratio of future rewards.
- **Why 0.99**: GPU purchase decisions require a long-term perspective. Rewards 5 days ahead retain 95.1% of their value.
- **Comparison**:
  - 0.9: Only 90% at day 1 — too myopic
  - 0.99: 95.1% at day 5, 74.0% at day 30 — appropriate long-term view
  - 0.999: 97.0% at day 30 — overvalues distant rewards

---

**References**:
- Silver et al., "Mastering the game of Go without human knowledge" (AlphaGo Zero, 2017)
- Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero, 2020)
- Auer et al., "Finite-time Analysis of the Multiarmed Bandit Problem" (UCB1, 2002)
