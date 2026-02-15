# AlphaZero/MCTS for Crypto Trading: Feasibility & Architecture Report

## 1. Executive Summary
**Verdict:** High Feasibility for Mid-Frequency Trading (1m-15m candles). Low Feasibility for HFT.

Applying AlphaZero/MuZero principles to crypto trading transforms the problem from simple pattern recognition (Transformer) to **strategic planning** (MCTS). This allows the agent to simulate future market states and evaluate the long-term impact of its actions, rather than just reacting to immediate price movements.

**Key Insight:** You cannot use AlphaZero (perfect simulator). You must use **MuZero** (learned world model) because the "rules" of the market (how price moves in response to volume) are unknown and stochastic.

---

## 2. Feasibility Assessment

### Challenges vs. Solutions
| Challenge | AlphaZero Limitation | Proposed MuZero Solution |
| :--- | :--- | :--- |
| **No Perfect Simulator** | Requires game rules (Chess/Go) | **Learned Dynamics Model**: Neural network predicts next state & reward. |
| **Stochasticity** | Deterministic transitions | **Stochastic MuZero**: Predicts *distribution* of future prices (categorical/Gaussian). |
| **Continuous State** | Discrete board state | **Latent Representation**: Encodes raw market data into dense vectors. |
| **Unbounded Time** | Fixed game length | **N-Step Bootstrapping**: Value function estimates infinite horizon returns. |

### Performance Impact (Mac M4)
*   **Current Transformer**: ~57 TPS (17ms/step). Pure inference.
*   **MCTS (50 sims)**: Est. **4-10 TPS** (100-250ms/step).
*   **Implication**: Too slow for tick-level execution, but highly effective for 1-minute candle strategies where you have 60 seconds to "think."

---

## 3. Architectural Blueprint

### A. State Representation ($s_t$)
The "Board" is a rolling window of multi-modal data.
*   **Market Data**: OHLCV (1m, 5m, 1h), Order Book Imbalance, Recent Trades.
*   **Context**: Funding Rates, Open Interest, CPI/Interest Rates (slow features).
*   **Agent State**: Current Position (Long/Short/Flat), Unrealized PnL, Wallet Balance.

### B. Action Space ($a_t$)
Discrete action space is required for standard MCTS.
*   **7 Actions**: `NEUTRAL`, `LONG_1` (0.3x), `LONG_2` (0.6x), `LONG_3` (1.0x), `SHORT_1`, `SHORT_2`, `SHORT_3`.

### C. The World Model (MuZero)
Instead of a simulator, we train three networks:
1.  **Representation ($h$)**: `Raw Data -> Latent State`
    *   *Arch*: ResNet or Transformer Encoder.
2.  **Dynamics ($g$)**: `(Latent, Action) -> (Next Latent, Reward)`
    *   *Crucial*: Learns "Market Physics" (e.g., impact of volume on price).
3.  **Prediction ($f$)**: `Latent -> (Policy, Value)`
    *   *Value*: Expected Sharpe Ratio over horizon $T$.

### D. Search Strategy (MCTS)
*   **Simulation**: Traverse the learned latent space using the Dynamics network.
*   **Selection**: UCB (Upper Confidence Bound) balances exploration vs exploitation.
*   **Evaluation**: Prediction network estimates value of leaf nodes.
*   **Depth**: 5-10 steps (lookahead of 5-10 minutes).

---

## 4. Mac M4 Optimization Recommendations

### A. Hardware Utilization
*   **Neural Engine (ANE)**: Offload the `Representation` and `Dynamics` networks to CoreML/ANE. This is critical. The M4 ANE is extremely fast for dense matrix math.
*   **CPU**: Run the MCTS Tree Search logic (C++/Rust) on the CPU performance cores.
*   **GPU**: Use Metal for batch training/inference if ANE capacity is exceeded.

### B. Software Stack
*   **Language**: Python for training, **Rust/C++** for MCTS Runtime (bound via PyO3/pybind11).
*   **Inference**: Convert PyTorch models to **CoreML** (`coremltools`).
*   **Batching**: Run MCTS for 8-16 pairs (BTC, ETH, SOL) in parallel to saturate the ANE.

---

## 5. Comparison: Transformer vs. AlphaZero

| Feature | Current (Transformer) | Proposed (AlphaZero/MuZero) |
| :--- | :--- | :--- |
| **Paradigm** | Reactive (Pattern -> Action) | **Planning** (Simulate -> Evaluate -> Action) |
| **Adaptability** | Low (Fixed weights) | **High** (Dynamic search based on volatility) |
| **Interpretability** | Black Box | **Tree Visualization** (See "thought process") |
| **Latency** | Ultra-low (~17ms) | **Medium-High** (~200ms) |
| **Training Data** | Historical Labels | **Self-Play** (Generates own scenarios) |
| **Risk Mgmt** | Implicit | **Explicit** (Value function penalizes risk) |

---

## 6. Implementation Roadmap

### Phase 1: The "World Model" (Weeks 1-3)
*   Goal: Train a Dynamics Network that accurately predicts the next candle given current state + action.
*   Metric: Next-step prediction error (MSE/Cross-Entropy).
*   *No MCTS yet. Just supervised learning of market physics.*

### Phase 2: Value Estimation (Weeks 4-5)
*   Goal: Train the Prediction Network to estimate the value (Sharpe Ratio) of a state.
*   *Validation*: Do high-value states actually correlate with future profitability?

### Phase 3: MCTS Integration (Weeks 6-8)
*   Goal: Implement MCTS (Python first, then Rust) using the trained World Model.
*   *Test*: Compare MCTS Policy vs. Raw Network Policy. MCTS should filter out "impulsive" bad trades.

### Phase 4: Productionization (Weeks 9+)
*   Optimization: CoreML conversion.
*   Live Testing: Paper trading with 1-minute execution loop.
