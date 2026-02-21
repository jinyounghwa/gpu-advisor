# AlphaZero-for-Trading Architecture Specification (Research Reference)

> 이 문서는 **트레이딩 도메인 연구용 아키텍처 스펙**입니다.  
> 현재 `gpu-advisor`의 운영 구현(실데이터 GPU 구매 의사결정)과는 입력 피처/액션 정의가 다를 수 있습니다.  
> 운영 기준은 `README.md`, `CRAWLER_GUIDE.md`, `backend/simple_server.py`, `backend/agent/*`를 우선 참조하세요.

## 1. System Overview
Adapting AlphaZero/MuZero for crypto trading requires shifting from perfect-information board games to partial-information stochastic markets.

**Core Concept:**
Instead of a game board, the "board" is a rolling window of market history.
Instead of an opponent, the "opponent" is the market environment (Nature).
The "winning condition" is maximizing risk-adjusted returns (Sharpe/Sortino) over a fixed horizon.

## 2. Component Design

### A. State Representation ($s_t$)
The input to the neural network (Representation Network).
*   **Market Features (Tensor)**:
    *   Price Data: OHLCV (1m, 5m, 1h, 4h) normalized.
    *   Order Book: Top 10 levels (bid/ask price & size), Imbalance.
    *   Derivatives: Funding rates, Open Interest, Liquidations.
    *   On-Chain: Net exchange flows, large tx counts.
*   **Macro/Context (Vector)**:
    *   Interest Rates, CPI (slow changing, broadcasted).
    *   Time encodings (Hour of day, Day of week).
*   **Agent State (Vector)**:
    *   Current Position (Long/Short/Neutral).
    *   Unrealized PnL.
    *   Available Capital.
    *   Time since last trade.

### B. Action Space ($a_t$)
MCTS works best with discrete action spaces.
*   **Discrete Actions (size 7)**:
    *   `NEUTRAL`: Close all positions.
    *   `LONG_1`: Long 33% leverage.
    *   `LONG_2`: Long 66% leverage.
    *   `LONG_3`: Long 100% leverage.
    *   `SHORT_1`: Short 33% leverage.
    *   `SHORT_2`: Short 66% leverage.
    *   `SHORT_3`: Short 100% leverage.

### C. World Model (MuZero Style)
Since we cannot perfectly "simulate" the market forward (unlike Chess), we learn a latent dynamics model.
1.  **Representation Network ($h$)**: $s_0 = h(o_1...o_t)$
    *   Encodes raw market data into a latent state $s_0$.
    *   *Arch*: Transformer Encoder or ResNet + LSTM.
2.  **Dynamics Network ($g$)**: $r, s_{k+1} = g(s_k, a_k)$
    *   Predicts immediate reward $r$ and next latent state $s_{k+1}$ given action $a_k$.
    *   *Input*: Latent state + Action embedding.
    *   *Output*: Next latent state + Expected 1-step PnL.
3.  **Prediction Network ($f$)**: $p, v = f(s_k)$
    *   Predicts policy $p$ (best action) and value $v$ (long-term return).
    *   *Value Target*: N-step future Sharpe Ratio or Log Returns.

### D. Search Strategy (MCTS)
*   **Simulation**: Traverse the learned latent space using the Dynamics Network.
*   **Selection**: UCB (Upper Confidence Bound) to balance exploration (trying new actions) and exploitation (highest value).
*   **Evaluation**: Use Prediction Network to estimate value of leaf nodes.
*   **Backpropagation**: Update Q-values up the tree.

## 3. Training Loop (Self-Play adaptation)
We cannot do true "self-play" against a static rule set. We play against **Historical Replay** with **Data Augmentation**.

1.  **Episode Generation**:
    *   Agent plays through a historical day (e.g., BTC/USDT Jan 1 2024).
    *   At each step $t$, run MCTS to select action $\pi_t$.
    *   Store trajectory $(s_t, a_t, \pi_t, r_t)$ in Replay Buffer.
2.  **Model Training**:
    *   Sample trajectories from Replay Buffer.
    *   Train World Model to predict:
        *   Real future rewards (actual PnL).
        *   Real future values (actual final Sharpe).
        *   Policy similarity (MCTS distribution).

## 4. Handling Incomplete Information
*   **No Hidden Information State**: We assume market data is the full observable state.
*   **Stochasticity**: The Dynamics Network predicts a *deterministic* path in latent space that represents the *expected* future. To handle volatility, we can use **Stochastic MuZero** (VQ-VAE in dynamics) to sample multiple possible futures.

## 5. Mac M4 Optimization Strategy
*   **Inference**: Run Neural Networks (Rep, Dyn, Pred) on Apple Neural Engine (ANE) via CoreML or MPS (Metal Performance Shaders) in PyTorch.
*   **MCTS Tree**: Implement in C++ or Rust (w/ Python bindings) to avoid Python overhead during tree traversal.
*   **Batching**: Run MCTS for multiple asset pairs in parallel (Batch MCTS) to saturate the GPU.
