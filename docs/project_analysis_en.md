# GPU Advisor Project Analysis Report

> Date: 2026-03-16
> Subject: GPU Advisor — AlphaZero-Based GPU Purchase Decision Agent

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Strengths](#2-strengths)
3. [Weaknesses and Limitations](#3-weaknesses-and-limitations)
4. [Implications](#4-implications)
5. [Summary Evaluation](#5-summary-evaluation)

---

## 1. Project Overview

GPU Advisor is a personal AI agent system that applies **AlphaZero/MuZero architecture** to the domain of GPU market purchase decision-making.
Rather than a simple price predictor, it is optimized for **action decisions** — "should I buy now, or wait?"

**Core Components:**

| Layer | Components |
|-------|------------|
| Data Collection | Danawa GPU prices + exchange rates + GPU news (automated daily at midnight) |
| Feature Engineering | Price · exchange rate · news · market · technical indicators → 256-dimensional vector |
| AI Model | Representation h(s) + **World Model g(s,a)** + Prediction f(s) + **Action Model a(z)** (~19M parameters total) |
| Planning | MCTS (50 simulations × 5 rollout steps) |
| Pipeline | Automated retraining → quality gates → release decision → Korean report generation |
| Service | FastAPI REST API + Next.js frontend |

---

## 2. Strengths

### 2-1. Originality of Domain Application — "Decision Optimization" Not "Price Prediction"

Most stock/price prediction projects approach the problem as regression, predicting "what will the price be tomorrow?"
GPU Advisor rejects this framing and redefines it as a **reinforcement learning action optimization** problem.

- The goal is not to minimize prediction error, but to learn an **action policy that maximizes expected reward**
- Even if BUY_NOW accuracy is 60%, executing at the right timing can outperform a more accurate but poorly timed predictor
- This is the right approach for domains where "when you're right" matters more than "how often you're right"

### 2-2. Transfer of AlphaZero Techniques to the Economic Domain

The core idea that enabled AlphaGo Zero to achieve superhuman performance in Go through self-play — **model-based planning** — is transplanted into the price market.

- **Representation Network h(s)**: 256-dimensional market state → compressed latent state (6.4M parameters)
- **World Model g(s,a)**: Learns how the market changes when an action is taken (6.5M parameters)
- **Prediction Network f(s)**: Estimates the best action and value from the current state (6.0M parameters)
- **Action Model a(z)**: Learns a context-aware action prior distribution (~43K parameters)
- **MCTS**: Searches for the optimal action through 50 simulations over the learned world model

The ability to perform **future scenario search** — impossible with simple supervised learning — is particularly impressive.

### 2-8. World Model — Internalizing Market Physics

The Dynamics Network `g(s_t, a_t) → (s_{t+1}, μ, σ²)` is not merely an auxiliary network; it is the **core World Model** of this entire system.

In MuZero, the world model's role is to replace the environment simulator. In Go, perfect rules exist for a perfect simulator, but the GPU market has no official rules. This project fills that gap by **inducting the market's causal structure from historical price data** using a neural network.

```
Input:  (current market latent s_t, action to take a_t)
Output: (next market state s_{t+1}, expected reward μ, reward uncertainty σ²)
```

- 4 Transformer blocks + Post-LN residual connections: learns complex nonlinearities in action-state interactions
- **Gaussian distribution output**: outputs both a point estimate (μ) and uncertainty (σ²) simultaneously, implicitly encoding confidence intervals during MCTS search
- The more accurate this network is, the closer MCTS searches to realistic futures; the less accurate it is, the lower the overall system reliability

Since world model quality determines the performance ceiling of the agent, **the World Model is simultaneously the most vulnerable and most critical component of this project**.

### 2-9. Action Model — Replacing Hardcoded Heuristics with a Learnable Prior

The `ActionModel` is a lightweight ~43K-parameter network consisting of two submodules.

**ActionEmbeddingLayer**

```
5D one-hot action → 16D learned embedding
```

Standard MuZero treats actions as simple one-hot vectors, implying all actions are equally "different." Learned embeddings can instead **induce semantic similarity between actions** from data:
- BUY_NOW and WAIT_SHORT may learn close embeddings as "buy-side" actions
- SKIP may learn an embedding distant from others as "full abandonment"

**ActionPriorNetwork (Context-Aware Prior)**

```
256D latent state → 128D → 64D → 5D logits → π(a|s)
```

The previous code hardcoded a `utility_bias` heuristic that manually adjusted action weights based on price trends:

```python
# Previous hardcoded approach
utility_bias[0] += 1.0 * trend_down - 1.2 * over_ma  # BUY_NOW
utility_bias[1] += 0.6 * over_ma + 0.5 * trend_up    # WAIT_SHORT
```

ActionPriorNetwork **learns this heuristic end-to-end from data**. It takes the market state (latent state) as input and directly outputs a probability distribution over "which action is appropriate in this situation."

It contributes with a weight of 0.15 in the final policy calibration:
```
final_π = 0.45 × MCTS + 0.25 × Reward + 0.15 × f-net + 0.15 × ActionModel
```

By replacing hardcoded human intuition with data-driven learning, this module demonstrates that **the system has evolved toward lower domain-knowledge dependency and greater self-adaptability**.

### 2-3. Explicit Uncertainty Modeling

The reward head outputs a **Gaussian distribution (μ, σ²)** rather than just a point estimate.

- The `reward_logvar` head is trained with Gaussian NLL loss, learning prediction uncertainty simultaneously
- `safe_mode` activates under low-confidence conditions (entropy > 1.58, confidence < 0.25)
- A sound design judgment that point estimates alone are insufficient for financial/decision-making AI

### 2-4. Fully Automated Online Learning Loop

```
Daily midnight → collect → feature extraction → readiness check → (after 30 days) auto-retrain → quality gates → release decision
```

The entire flow from data collection to release decision is automated without human intervention. Notable aspects:

- Retraining every 7 days for **market drift adaptation**
- 7 quality gates run automatically before release to **block bad models from deployment**
- Results are auto-generated as Korean Markdown reports for readability

This is also a robust design from an MLOps perspective.

### 2-5. Engineering Treatment of Missing Data

Acknowledging that training data is not always complete, missing data is **encoded as a feature itself** using the `value=0.0 + missing_mask=0.0` pattern.

- Technical indicators: 106 dimensions = 53 values + 53 missing masks
- Instead of filling with arbitrary constants (e.g., the mean) when data is absent, the fact of "not knowing" is made explicit
- Provides a foundation for the model to learn different behavior based on data quality

### 2-6. Practicality of a Lightweight Architecture

With approximately 19M parameters (h + g + f + a combined), real-time inference is possible on Apple MPS.

- Minimized for the domain, not GPT-scale
- Entire model managed in just 2 checkpoint files (~144MB)
- Fully operational on a personal Mac without cloud infrastructure

### 2-7. Multi-Source Calibration to Prevent Mode Collapse

The final action probability does not rely on a single source; it blends four:

```
final_π = 0.45 × MCTS + 0.25 × Reward + 0.15 × f-net + 0.15 × ActionModel
```

- Empirically effective at preventing action concentration (mode collapse)
- ActionModel provides the actual historical action distribution as a prior

---

## 3. Weaknesses and Limitations

### 3-1. Severe Data Scarcity

| Item | Figure |
|------|--------|
| Daily samples collected | 24 GPU models × 1 day = 24 transitions |
| Total training samples at 30 days | ~720 |
| Model parameter count | ~19,000,000 |

Training data is extremely sparse relative to the number of parameters.
With a batch size of 32 over 500 steps, the same data is **reused more than 20 times**, meaning overfitting risk is persistent.
Even if quality gates are passed, it is difficult to determine whether this reflects true generalization ability or memorization of training data.

### 3-2. Single-Source Dependency — Danawa

Domestic price data depends on **a single Danawa crawl**.

- Danawa UI changes → crawler immediately non-functional
- IP blocks or `robots.txt` policy changes halt data collection
- Does not reflect the gap between overseas prices (Amazon, Newegg) and domestic prices
- Excludes official MSRP, Coupang, and other major platform prices

Robustness increases with diverse data sources, but the current structure is a SPOF (single point of failure).

### 3-3. Oversimplified Reward Function

```python
if action == "BUY_NOW":
    reward = pct_change          # good if tomorrow's price rises
if action in {"WAIT_SHORT", "WAIT_LONG"}:
    reward = -pct_change         # good if tomorrow's price falls
```

The following factors are absent from real GPU purchase decisions:

- **Fees and shipping**: actual purchase costs
- **Opportunity cost**: losses when prices spike during a wait
- **Time risk**: no clear boundary between WAIT_SHORT (days) and WAIT_LONG (weeks)
- **Stock depletion risk**: "waiting until the item sells out" scenario
- **New product launch events**: discontinuous price changes

When the reward function does not sufficiently reflect reality, the goal the agent optimizes diverges from actual user benefit.

### 3-4. macOS and Apple MPS Dependency

The entire system is effectively tied to the **macOS + Apple Silicon** environment.

- `LaunchAgent`: macOS-exclusive automation tool
- `torch.backends.mps`: Apple MPS-exclusive GPU acceleration
- `pmset`: macOS-exclusive power management command

Migrating to a Linux server or cloud environment requires converting LaunchAgent → systemd/cron and MPS → CUDA/CPU.

### 3-5. Cold Start Problem — 30-Day Wait

Model training requires **30 days of data accumulation** before it can activate.

- Inference results during the initial 30 days are practically meaningless
- Restarting the service (resetting data) resets the 30-day wait
- New GPU models entering the market trigger another 30-day wait for that model

### 3-6. Fundamental Limits of the World Model — Simulating the Future from the Past

The World Model (Dynamics Network) is the **weakest link** of this entire system.

This model is trained on 30 days of historical price data. This means the 50 simulations MCTS performs are essentially **extrapolations of patterns observed in the past**. It is therefore fundamentally unable to respond to out-of-distribution events:

- **Discontinuous price jumps** like the RTX 50 series launch: the world model cannot predict transitions absent from training data
- GPU demand surges from cryptocurrency market spikes: news features exist, but are meaningless without similar historical event data
- Global semiconductor supply chain crises: long-term trend shifts cannot be captured in a 30-day window

As an analogy, this world model is a "sunny day" simulator. It cannot predict a storm because it has never seen one.

### 3-7. Structural Weakness of MCTS — Shallow Search

```
50 simulations × 5 rollout steps = 250 dynamics calls
```

Unlike Go AlphaZero performing thousands to tens of thousands of simulations, 50 simulations explore the search space very shallowly.
As world model errors accumulate over 5 rollout steps, the reliability of search results can drop sharply.

### 3-8. Weak Training Signal for the Action Model

ActionPriorNetwork is trained on **pseudo-labels** derived inversely from historical price changes — i.e., labels are generated by a simple rule: "if the price fell the next day, BUY_NOW yesterday was a bad action."

- These labels are noisy and inaccurate (daily changes cannot judge long-term value)
- Severe class imbalance: labels skew toward HOLD/SKIP during sideways markets
- `class_weights` address balance, but they are not a fundamental fix for data bias

As a result, it is difficult to determine whether the "context awareness" ActionModel learns reflects actual market causality or merely memorizes price trend correlations.

### 3-9. Gap Between Evaluation Metrics and Real User Experience

```
Directional accuracy >= 55% → release permitted
```

Even if the model correctly predicts "price direction" 55% of the time, whether actual users generate profit by following recommendations is a separate question.
The backtest evaluator (`AgentEvaluator`) is implemented without lookahead bias, but transaction costs and execution slippage are not reflected.

### 3-10. Limitations of News Sentiment Analysis

- **English-only news collection**: Korean GPU-related news, Naver Cafe, and community sites (Ruliweb, Quasarzone, etc.) are excluded
- **Fallback scorer**: Accuracy degrades significantly when operating in rule-based mode without the Transformer model
- **Insufficient GPU domain specialization**: A general-purpose sentiment model (DistilBERT SST-2) is not optimized for GPU market terminology

---

## 4. Implications

### 4-1. Reward Function Design Is Everything When Applying RL to Real Domains

The most important lesson from this project is the **importance of the reward function**.
In Go, there is a clear, immediate reward: "did you win or lose?"
But in GPU purchasing, what constitutes "a good purchase" varies by perspective.

- Is the benchmark one week later, or one month later?
- Is buying when you need it the best decision?
- Is failing to buy after waiting a loss?

When the reward function does not sufficiently proxy reality, the agent becomes an **optimizer of the reward function rather than actual user benefit**. This is a textbook case of Goodhart's Law.

### 4-2. Rethinking Deep Learning Suitability in Data-Scarce Domains

Deep learning is dominant in domains with hundreds of thousands of training rows.
But at the scale of **24 samples per day**, **720 samples over 30 days**, the following questions must be asked:

- Was there a comparison with traditional ML methods like XGBoost or LightGBM?
- Is it actually better than time-series-specialized methods like ARIMA or Prophet?
- Can a 19M-parameter neural network learn from 720 samples?

More complex models are not always better; choosing model complexity appropriate to data volume is critical.

### 4-3. The Practical Possibility of Personal AI Automation Systems

This project implements an AI system **running entirely on a personal Mac** without cloud infrastructure or large-scale resources.

- Privacy: all data stored locally
- Cost: zero GPU cloud spend
- Control: complete ownership

This is a counterexample to the conventional wisdom that "AI = big corporations + cloud."
It demonstrates that **small domain-specialized models** — not LLMs — are sufficiently practical on personal devices.

### 4-4. Release Pipeline = The Core of MLOps

Just as software has CI/CD, ML models require **automated quality validation gates**. This project illustrates this well.

```
Training complete ≠ Deployment ready
```

Seven quality gates operate automatically to prevent bad models from entering production.
In particular, `mode_collapse` detection and `action_entropy` monitoring are important mechanisms for catching RL agent-specific regression patterns.

This approach is a miniature **ML model management system (Model Registry + automated validation)**, and the same principles apply to team-scale MLOps system design.

### 4-5. Practical Challenges of Continual Learning

This project adopts a continual learning structure with retraining every 7 days.
The fundamental challenges of continual learning this structure surfaces:

- **Catastrophic Forgetting**: retraining on new data can overwrite patterns learned from the past
- **Distribution Shift**: if the market structure itself changes (e.g., a new GPU generation launches), historical data can be actively harmful
- **Evaluation Difficulty**: it is hard to determine whether the reasons a yesterday's model was good remain valid today

The retraining interval (7 days), training data window (30 days), and gate thresholds are all empirically set values.
Without sensitivity analysis of these hyperparameters, the "reasons why it works" are unclear.

### 4-6. Absence of Explainability

When a user receives the recommendation "Buy RTX 5060 now," **why that conclusion was reached** cannot be explained.

- Which features contributed most to this decision? (No feature importance)
- What future scenarios did MCTS imagine? (Simulation paths are opaque)
- Is there any basis beyond the confidence score to trust this recommendation?

In financial and purchase decision AI, explainability is important for both building trust and debugging errors.

### 4-7. World Model Quality Is the System's Performance Ceiling

An important implication this project reveals is that **in model-based reinforcement learning, world model errors propagate**.

MCTS searches for the optimal action under the assumption that the world model is accurate. If the world model is inaccurate, MCTS is searching through "illusory futures." This error propagation structure means:

- No matter how good the Prediction Network f(s) is, poor g(s,a) makes the entire search result unreliable
- World model training is especially risky in data-scarce domains — the model may learn its own biases as "market laws"
- **World model validation** is as important as agent performance validation, but the current pipeline has no gate independently measuring g(s,a) prediction accuracy

Monitoring world model prediction error (next-state MSE, reward calibration error) as a separate quality gate would greatly contribute to system stability.

### 4-8. The Difference Between "A Working Prototype" and "A Validated System"

This project is a technically polished prototype.
However, without answering the following questions, it is difficult to call it a **validated system**:

1. Over 6 months of actual use, did the agent purchase GPUs at appropriate times?
2. Did following the recommendations actually save money compared to not following them?
3. Is there a statistically significant performance difference compared to a random strategy?

Backtests can be optimistically biased because they use historical data.
Actual value can only be proven through **live A/B testing** or **tracking real purchase records**.

---

## 5. Summary Evaluation

### Technical Achievements

| Item | Rating |
|------|--------|
| Originality of idea | ★★★★★ — RL-based purchase decision-making is a fresh approach |
| Implementation completeness | ★★★★☆ — automated pipeline, safeguards, and tests in place |
| Code quality | ★★★★☆ — significantly improved after code review |
| Practical value | ★★★☆☆ — limited by data scarcity and reward function constraints |
| Scalability | ★★★☆☆ — requires resolution of macOS dependency and multi-source expansion |

### Core Conclusions

GPU Advisor is a **bold experiment applying AlphaZero ideas to the economic domain**.
It holds **outstanding value as a learning project** in that it implements and automates — from scratch — how the MCTS + World Model + Action Model approach works in a real-world domain outside of games.

Two design choices in particular are noteworthy:
- **World Model**: learns market dynamics in latent space without an environment simulator, enabling MCTS search
- **Action Model**: replaces hardcoded heuristics with a data-driven context-aware prior, reducing domain bias

However, for this to be **trusted as an actual GPU purchase assistance tool**, the following are needed:

1. **Live validation**: 6+ months of experiments tracking actual purchase decisions
2. **Independent world model validation**: separate measurement of next-state MSE and reward calibration error
3. **Baseline comparison**: fair comparison with ARIMA, XGBoost, and simple moving average strategies
4. **Multiple data sources**: eliminating Danawa dependency
5. **Reward function improvement**: incorporating transaction costs, opportunity costs, and inventory risk
6. **Explainability**: adding SHAP or attention weight-based explanations

The true value of this project lies less in the **artifact itself** than in the **engineering experience and insights** accumulated through the entire process of applying the AlphaZero architecture to a new domain.

---

*This document was written based on code review and architectural analysis as of 2026-03-16.*
