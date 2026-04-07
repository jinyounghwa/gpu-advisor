# Hyperparameter Design Rationale

This document explains every key hyperparameter in the GPU Advisor project and the reasoning behind each choice.

## 1. Latent Dimension: 256D

| Item | Value |
|------|-------|
| Input dimension | 256D (GPU market state vector) |
| Latent space | 256D |
| Expansion ratio | 1:1 (input = latent) |

**Input vector composition (256D)**:

| Feature group | Dim | Content |
|--------------|-----|---------|
| Price features | 60D | Normalized price, MA7/MA14/MA30, % change, volatility, etc. |
| Exchange features | 20D | USD/KRW, JPY/KRW, EUR/KRW normalized |
| News features | 30D | Sentiment score, article count, positive/negative ratio |
| Market features | 20D | Seller count, inventory status |
| Time features | 20D | Day of week, month, year-end flag |
| Technical indicators | 106D | RSI, MACD, momentum, padding |

**Why 256?**

- **Information capacity**: 256D is sufficient to compress 6 feature categories into a single vector.
- **Power of two**: 2^8 = 256 is optimal for GPU tensor operations — memory alignment and SIMD instructions.
- **MuZero reference**: The original MuZero used 256 channels for Go (19×19 board = 361 intersections).
- **128 vs 256 vs 512**:
  - 128D: Potential bottleneck for 6 feature categories
  - 256D: ~42D allocation per category, good expressiveness
  - 512D: 2× parameters but risk of overfitting

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
| BUY_NOW (0) | Buy immediately | Price dip detected |
| WAIT_SHORT (1) | Wait ~1 week | Slight decline expected |
| WAIT_LONG (2) | Wait ~1 month | Significant decline expected |
| HOLD (3) | Observe | High uncertainty |
| SKIP (4) | Skip this cycle | Avoid purchase entirely |

**Action labeling thresholds** (`_action_from_delta()`):

| Price change | Label |
|-------------|-------|
| `pct >= +2%` | BUY_NOW (0) |
| `-0.5% > pct >= -2%` | WAIT_SHORT (1) |
| `pct <= -2%` | WAIT_LONG (2) |
| Otherwise | HOLD (3) |

> **Note**: SKIP(4) is never labeled from price data because the evaluator's reward function creates a reverse-direction learning signal. For a -6% price move, WAIT_LONG reward (+0.06) > SKIP reward (-0.009), so labeling SKIP from price downturns would train the model in the wrong direction.

**Why 5?**

- **3 actions (buy/wait/hold)**: Too simple. Cannot express "how long to wait"
- **5 actions**: Covers both time axis (now/short/medium) and intensity axis (buy/wait/skip)
- **7+ actions**: No practical differentiation for GPU purchase decisions

## 4. Exploration Constant (PUCT): c = √2 ≈ 1.414

MCTS uses AlphaZero-style PUCT (Predictor + UCT):

```
UCB(s, a) = Q(s, a) + c × P(s, a) × √N_parent / (1 + N_child)
```

- `Q(s, a)`: Action value estimate (average backup reward)
- `P(s, a)`: Prior probability from prediction network f
- `N_parent`: Parent node visit count
- `N_child`: Current node visit count

**Why √2?**

- **UCB1 theory**: Auer et al. (2002) proved √2 is the optimal exploration constant for multi-armed bandits.
- **Exploration-exploitation tradeoff**:
  - c = 1.0: Exploitation bias, gets stuck on early good actions
  - c = √2: Theoretically optimal, sufficient exploration of all actions
  - c = 2.0: Over-exploration, slow convergence
- **AlphaGo**: Uses PUCT variant with c_puct = 1.5–2.5. Our value falls within this range.

## 5. Dirichlet Noise (Root Exploration Diversity)

| Item | Value |
|------|-------|
| epsilon (ε) | 0.25 |
| alpha (α) | 0.03 |
| Applied to | Root node priors only |

```python
noise = Dirichlet([α] * action_dim)
policy_probs = (1 - ε) * policy_probs + ε * noise
```

**Design rationale**:
- Same method as AlphaZero: noise applied only to the root node to increase search diversity
- ε=0.25: 75% of original prior preserved, 25% noise mixed in
- α=0.03: Low value = concentrated spike noise (one action gets most noise) → stronger exploration pressure

## 6. Rollout Depth: 5 Steps

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

## 7. Network Architecture

### World Model: 3 Networks (h, g, f)

```
Input 256D → Hidden 512D (×4 residual blocks) → Output 256D/5D/1D
```

- **Residual connections**: All blocks apply `x = x + block(x)` → training stability for deep networks
- **2× expansion**: 512D = 2× the 256D input provides sufficient nonlinear capacity.

| Network | Role | Parameters |
|---------|------|------------|
| h (Representation) | 256D input → 256D latent state | ~6M |
| g (Dynamics) | Latent state + action → next state + reward (μ, σ²) | ~6M |
| f (Prediction) | Latent state → policy + value | ~6M |
| a (Action Model) | Action embedding + prior network | ~43K |

### ActionModel (a)

```
ActionEmbeddingLayer(num_actions=5, embed_dim=16)
ActionPriorNetwork: 256 → 128 → 64 → 5
Total parameters: ~43K
```

- Predicts action prior distribution from the current latent state
- Used with 15% weight in the 4-signal calibration blend
- Replaces the hardcoded `utility_bias` heuristic with learned priors

### Dynamics Network Reward Uncertainty Head

```
g(s, a) → (next_state, reward_mean μ, reward_logvar)
σ² = exp(reward_logvar)    # variance
σ  = reward_var.sqrt()     # standard deviation
```

## 8. Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-4 | Stable convergence range for AdamW |
| Batch size | 32 | Efficient sampling for ~24 GPU models × days |
| Training steps | 500 | Prevents overfitting on small dataset |
| Weight decay | 1e-5 | L2 regularization against overfitting |
| Gradient clipping | 1.0 | `clip_grad_norm_()` return value captured and tracked |
| Optimizer | AdamW | Decoupled weight decay application |

### Learning Rate 1e-4

```
1e-3: Unstable training, loss oscillation
1e-4: Stable convergence within 500 steps ← chosen
1e-5: Too slow, needs 5000+ steps
```

## 9. Loss Function Weights

Total loss is a weighted sum of 5 components:

```python
total_loss = (
    1.0 * latent_loss          # Latent state consistency (world model core)
  + 1.0 * policy_loss          # Policy KL loss
  + 1.0 * value_loss           # Value MSE loss
  + 0.5 * reward_nll_loss      # Gaussian NLL (trains uncertainty head)
  + 0.3 * action_prior_loss    # ActionModel supervised learning
)
```

**Gaussian NLL formula**:
```
NLL = 0.5 × ((r - μ)² × exp(-logvar) + logvar)
```
- Passes real gradients to the `logvar` head, enabling reward uncertainty learning

**action_prior_loss**: CrossEntropy — trains ActionModel to match price-derived action labels

## 10. Policy Calibration Weights (4-Signal Blend)

> **2026-04-03 Update**: MCTS weight raised from 0.45 to 0.60 to prioritize planning results.

```python
# Current (v3.1, 2026-04-03+)
calibrated = 0.60 × MCTS_policy + 0.20 × reward_policy + 0.10 × f_prior + 0.10 × action_model_prior

# Previous (v3.0, ~2026-04-02)
# calibrated = 0.45 × MCTS_policy + 0.25 × reward_policy + 0.15 × f_prior + 0.15 × action_model_prior
```

| Source | Weight | Role |
|--------|--------|------|
| MCTS policy | **0.60** | Core decision (planning-based) — increased |
| Reward policy | **0.20** | Expected reward correction |
| f-net prior | **0.10** | Data distribution regularization |
| ActionModel prior | **0.10** | Learned action prior (replaces utility_bias) |

**Why raise MCTS to 0.60?**

- At 0.45, MCTS planning results were being diluted by reward/prior signals
- 0.60 makes planning the dominant signal while retaining calibration
- Weights sum to 1.0, guaranteeing a valid probability distribution

## 11. Anti-Collapse Regularizer

> **2026-04-03 Update**: Threshold lowered from 0.65 to 0.45 to allow more confident decisions.

```python
# Current (v3.1, 2026-04-03+)
min_entropy_target = 0.45  # bits (was: 0.65)

if entropy(calibrated) < min_entropy_target:
    alpha = min(0.55, (min_entropy_target - entropy_now) / max(min_entropy_target, 1e-6))
    uniform = ones(num_actions) / num_actions   # unbiased uniform
    calibrated = (1 - alpha) * calibrated + alpha * uniform
```

**Design rationale**:
- **Uniform distribution** (not empirical prior): The empirical prior (e.g., 60% HOLD) amplifies training bias. Uniform distribution is neutral.
- `alpha` cap at 0.55: Prevents over-regularization
- `min_entropy_target = 0.45` (changed): Relaxed from 0.65 so that anti-collapse does not dilute high-confidence decisions. Theoretical maximum confidence rises from ~0.587 to ~0.80.

## 12. Safety Thresholds

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| Min confidence | 0.25 | Must be higher than uniform probability (1/5 = 0.20) |
| Max entropy | 1.58 | 98% of maximum entropy ln(5) ≈ 1.609 |
| Min action entropy | 0.25 | Prevents mode collapse (0 = single action only) |
| Max abstain ratio | **0.93** | Updated 2026-04-03: 0.85→0.90→0.93 — allows conservative behavior in uncertain markets |
| Min accuracy | 0.55 | Must be statistically significant above random (50%) |

See [SAFETY_MECHANISMS.md](SAFETY_MECHANISMS.md) for detailed safety mechanism documentation.

## 13. Discount Factor: γ = 0.99

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
