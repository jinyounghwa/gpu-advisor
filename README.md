# GPU Advisor

**Planning-based AI Agent for GPU Purchase Timing**

A planning-based AI agent that decides optimal GPU purchase timing using AlphaZero/MuZero-style world model and Monte Carlo Tree Search (MCTS). The agent observes market data, simulates future scenarios, and recommends actions — mimicking how AlphaGo evaluates win rates in Go.

## 🎯 Overview

This system helps answer the question: **"Should I buy this GPU now or wait?"**

Just like AlphaGo calculates win probabilities in Go, this system calculates **purchase profitability scores (0-100%)** to determine the best time to buy GPUs.

### 💡 Project Motivation

As a researcher and developer, I found myself constantly facing the dilemma of GPU purchase timing. Graphics card prices fluctuate dramatically based on market conditions, new releases, and global supply chains. This project was born from a simple need: **to bring AI-powered decision intelligence to everyday purchase decisions**.

By applying the same Monte Carlo Tree Search (MCTS) principles that power AlphaGo to the GPU market, this system transforms complex market signals into actionable recommendations. This is part of my **"0.1B AI Project"** series - demonstrating that sophisticated AI applications don't always require billions of parameters, but rather smart architecture and thoughtful feature engineering.

### Key Features

- 🤖 **AlphaZero Architecture**: ~19M parameters (Representation h + Dynamics g + Prediction f + ActionModel a + MCTS)
- 📊 **Automated Data Collection**: Daily crawling of GPU prices, exchange rates, and news via macOS LaunchAgent
- 🧠 **256-Dimensional Features**: Rich feature engineering from raw market data to 256D state vectors
- 📈 **Real-time Predictions**: REST API for instant purchase timing recommendations
- ⏰ **LaunchAgent Automation**: Fully automated daily data collection at midnight (00:00)

### 🎮 Core Principle: Game Theory Meets Market Analysis

The fundamental insight behind this project is simple yet powerful: **purchasing decisions can be modeled as a sequential decision-making game**, similar to Go.

**The Analogy:**
- In Go, AlphaGo evaluates "Should I play this move?" → Win probability (0-100%)
- In GPU Market, our system evaluates "Should I buy this GPU?" → Purchase profitability score (0-100%)

**How It Works:**
1. **State Representation**: Market conditions (prices, trends, news sentiment) are encoded into a 256-dimensional latent state
2. **MCTS Simulation**: The system simulates 50 possible future scenarios (price drops, new releases, market crashes)
3. **Value Prediction**: Each scenario is evaluated for purchase timing optimality
4. **Final Recommendation**: The best action (Buy Now / Wait) is selected based on simulated outcomes

Unlike traditional price prediction models that try to forecast exact future prices (which is nearly impossible), this system focuses on **decision quality** - answering "Is now a good time?" rather than "What will the price be?"

## 📋 Architecture

```
Input: GPU Model (e.g., RTX 5060)
  ↓
AlphaZero MCTS Simulation
  ↓
Output: Purchase Score 75% → "Buy Now!"
```

### System Components

1. **Data Collection System**
   - Danawa GPU price crawler (24 models)
   - Exchange rate fetcher (USD/KRW, JPY/KRW, EUR/KRW)
   - News crawler with sentiment analysis
   - 256-dimensional feature engineering

2. **AI Engine**
   - Representation Network h(s): Encodes 256D market state → 256D latent state (3× FeedForward with residual connections)
   - Dynamics Network g(s,a): Predicts next latent state + Gaussian reward distribution (μ, log σ²)
   - Prediction Network f(s): Outputs policy logits + value (shared trunk with residual blocks)
   - Action Model a(z): Learned action prior from latent state (replaces hardcoded heuristic)
   - MCTS: AlphaZero-style PUCT search with Dirichlet noise for exploration diversity

3. **Backend Server**
   - FastAPI REST API (`backend/simple_server.py`)
   - Real-time predictions
   - Training dashboard
   - Swagger UI documentation
   - Release/readiness pipeline on real crawled data

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install -r backend/requirements.txt
pip install -e ".[dev]"  # dev dependencies (pytest, httpx)
```

### Setup

1. **Configure Automated Data Collection**

```bash
./setup_cron.sh
```

This sets up daily automatic data collection at midnight (00:00).

2. **Manual Data Collection** (for testing)

```bash
python3 crawlers/run_daily.py
```

Optional (crawl/feature only, skip release/automation stage):

```bash
python3 crawlers/run_daily.py --skip-release
```

Automation note:
- Before 30-day readiness, this runs release dry-check only.
- After 30-day readiness, this automatically runs training + release flow.
- Then it retrains again whenever the configured new-data interval is accumulated.

3. **Start Backend Server (Production Path)**

```bash
python3 backend/simple_server.py
```

Access the API at: `http://localhost:8000`
Swagger UI: `http://localhost:8000/docs`

### Docker (Alternative)

```bash
docker compose up --build
```

Backend: `http://localhost:8000` / Frontend: `http://localhost:3000`

### Running Tests

```bash
python3 -m pytest tests/ -v
```

### Production Auth & Persistence

`backend/simple_server.py` supports JWT auth (OAuth2 password flow), API key auth, hybrid mode, and pluggable DB backends.

```bash
# Auth mode: none | api_key | jwt | hybrid
export GPU_ADVISOR_AUTH_MODE=jwt
export GPU_ADVISOR_AUTH_DEFAULT_USER=admin
export GPU_ADVISOR_AUTH_DEFAULT_PASSWORD=change-me
export GPU_ADVISOR_JWT_SECRET='replace-with-long-random-secret'
export GPU_ADVISOR_JWT_EXP_SECONDS=3600

# Optional for api_key or hybrid mode
export GPU_ADVISOR_API_KEYS='key1,key2'

# Rate limit
export GPU_ADVISOR_RATE_LIMIT_PER_MINUTE=60

# Persistence backend: sqlite | postgres
export GPU_ADVISOR_DB_BACKEND=sqlite
export GPU_ADVISOR_DB_PATH='data/processed/gpu_advisor.db'
# If postgres:
# export GPU_ADVISOR_DB_BACKEND=postgres
# export GPU_ADVISOR_POSTGRES_DSN='postgresql://user:pass@localhost:5432/gpu_advisor'

# Sentiment backend: auto | transformers | rule
export GPU_ADVISOR_SENTIMENT_BACKEND=auto
export GPU_ADVISOR_SENTIMENT_MODEL='distilbert-base-uncased-finetuned-sst-2-english'
```

Token issuance endpoint:

```bash
curl -X POST http://localhost:8000/api/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=change-me"
```

### Making Predictions

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"model_name": "RTX 5060"}'
```

## 📊 Data Pipeline

```
LaunchAgent (Daily @ 00:00)
  ↓
Raw Data Collection
  ├─ GPU Prices (Danawa, 24 models)
  ├─ Exchange Rates (USD/KRW, JPY/KRW, EUR/KRW)
  └─ News + Sentiment (Google News RSS)
  ↓
Feature Engineering → 256D state vector
  ↓
Training Dataset (data/processed/dataset/)
  ↓
Auto Training (post-30d): Fine-tune h/g/f/a via AgentFineTuner
  ↓
Quality Gates → Release Check
  ↓
Purchase Predictions (via MCTS planning)
```

Operational note:
- Production data/agent path uses `crawlers/run_daily.py` + `backend/simple_server.py` + `backend/agent/*`.
- Legacy synthetic benchmark modules under `backend/main.py` and `backend/api/*` are not used for production decisions.

## 🤖 Reinforcement Learning Architecture

### Learning Paradigm

Unlike supervised learning that requires labeled answers, this project uses **Model-Based Reinforcement Learning** — the same paradigm as MuZero — to learn optimal GPU purchase decisions from market data alone.

```
Supervised Learning: "Here is the correct action" (requires oracle)
Reinforcement Learning: "Here is the reward for your action" (learns from consequences)
```

The key challenge: GPU market has no "game rules" or perfect simulator. Solution: **learn the market simulator** (World Model = Dynamics Network `g`) simultaneously with the decision policy.

### MDP Formulation

| MDP Element | GPU Advisor Mapping |
|------------|---------------------|
| State S | 256D market vector (prices, exchange rates, news, technical indicators) |
| Action A | {BUY_NOW, WAIT_SHORT, WAIT_LONG, HOLD, SKIP} |
| Transition T | Dynamics Network `g(s,a) → s'` — learned market simulator |
| Reward R | Next-day price change × action direction |
| Discount γ | 0.99 — time-value of money |
| Policy π | 4-signal blend: 0.45×MCTS + 0.25×Reward + 0.15×f-net + 0.15×ActionModel |

### Training Loop (Post-30-Day)

```
Real Market Data (30 days)
    ↓
feature_engineer.py → 256D state vectors
    ↓
AgentFineTuner (500 steps, batch=32, lr=1e-4)
    ├── h(s_t) → latent z_t
    ├── g(z_t, a_t) → ẑ_{t+1}, μ, σ²     [World Model]
    ├── f(z_t) → policy, value             [Prediction]
    └── a(z_t) → action prior              [Action Model]
    ↓ (5-loss joint optimization)
alphazero_model_agent_latest.pth
    ↓
AgentEvaluator (7 quality gates)
    ↓
Release: tag release-agent-20260322-105138
```

**5-loss joint optimization:**
```python
total_loss = (
    1.0 * latent_loss          # World Model consistency
  + 1.0 * policy_loss          # Policy learning
  + 1.0 * value_loss           # Value estimation
  + 0.5 * reward_nll_loss      # Gaussian reward (μ, σ²)
  + 0.3 * action_prior_loss    # Action Model supervision
)
```

### Post-30-Day Performance (2026-03-22)

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Directional Accuracy | **89.4%** | always_buy: 11.1% |
| Avg Reward | **+0.0064** | always_buy: +0.0023 |
| Uplift vs always_buy | **+0.0040** | baseline: 0 |
| Win Rate (training) | **75%** | random: ~50% |
| Action Entropy | **1.459** | mode collapse threshold: 0.25 |
| Quality Gates | **7/7 PASS** | — |

## 🧠 Feature Engineering (256 Dimensions)

| Feature Category | Dimensions | Description |
|-----------------|------------|-------------|
| Price Features | 60 | Normalization, volatility, moving averages |
| Exchange Features | 20 | USD/KRW, JPY/KRW, EUR/KRW trends |
| News Features | 30 | Sentiment analysis, keyword frequency |
| Market Features | 20 | Stock status, seller count |
| Time Features | 20 | Day of week, month, quarter |
| Technical Indicators | 106 | RSI, MACD, Bollinger Bands |

## 📁 Project Structure

```
gpu-advisor/
├── backend/                       # AI & API backend
│   ├── simple_server.py           # FastAPI server
│   ├── agent/                     # Agent pipeline
│   │   ├── gpu_purchase_agent.py  # MCTS planning agent
│   │   ├── fine_tuner.py          # Model fine-tuning
│   │   ├── evaluator.py           # Backtest evaluator
│   │   └── release_pipeline.py    # Release quality gates
│   └── models/                    # AlphaZero networks
│       ├── representation_network.py  # h(s): 256D state → 256D latent (residual FF)
│       ├── dynamics_network.py        # g(s,a): latent transition + Gaussian reward
│       ├── prediction_network.py      # f(s): policy logits + value (residual blocks)
│       ├── action_model.py            # a(z): learned action prior (ActionEmbedding + ActionPriorNet)
│       └── mcts_engine.py             # MCTS: PUCT search + Dirichlet noise
│
├── crawlers/                      # Data collection modules
│   ├── danawa_crawler.py          # GPU price crawler
│   ├── exchange_rate_crawler.py   # Exchange rate fetcher
│   ├── news_crawler.py            # News crawler
│   ├── feature_engineer.py        # 256D feature generation
│   └── run_daily.py               # Daily orchestration
│
├── frontend/                      # Next.js React UI
│   └── app/page.tsx               # Advisor + Training dashboard
│
├── tests/                         # Pytest test suite
│   ├── test_networks.py           # Neural network tests
│   ├── test_mcts.py               # MCTS engine tests
│   ├── test_feature_engineer.py   # Feature pipeline tests
│   └── test_api.py                # API endpoint tests
│
├── data/                          # Data storage
│   ├── raw/                       # Raw collected data
│   └── processed/                 # 256D feature vectors
│
├── docs/                          # Technical documentation
├── .github/workflows/ci.yml       # CI/CD pipeline
├── Dockerfile                     # Backend container
├── docker-compose.yml             # Multi-service orchestration
├── pyproject.toml                 # Python project config
└── .env.example                   # Environment template
```

## 📖 Documentation

### Root Documents

- [`README.md`](README.md): 프로젝트 개요, 아키텍처, 실행 방법, API 사용법
- [`종합_프로젝트_보고서.md`](종합_프로젝트_보고서.md): 전체 시스템을 한 번에 보는 한국어 종합 보고서
- [`GPU_PURCHASE_ADVISOR_REPORT.md`](GPU_PURCHASE_ADVISOR_REPORT.md): 성능/평가 결과 중심의 한국어 평가 보고서
- [`CRAWLER_GUIDE.md`](CRAWLER_GUIDE.md): 일일 데이터 수집 파이프라인 및 cron 운영 가이드
- [`architecture_spec.md`](architecture_spec.md): AlphaZero/MuZero 기반 트레이딩 아키텍처 상세 명세
- [`feasibility_report.md`](feasibility_report.md): AlphaZero/MCTS 적용 타당성 및 단계별 구현 로드맵
- [`frontend/README.md`](frontend/README.md): 프론트엔드(Next.js) 실행/개발 가이드

### `docs/` Core Guides

- [`docs/STUDY_GUIDE.md`](docs/STUDY_GUIDE.md): 학습 순서 중심의 스터디 가이드
- [`docs/IMPLEMENTATION_GUIDE.md`](docs/IMPLEMENTATION_GUIDE.md): 단계별 구현 절차 가이드
- [`docs/FILE_GUIDE.md`](docs/FILE_GUIDE.md): 파일/모듈 단위 역할 설명서
- [`docs/PROJECT_PRINCIPLES.md`](docs/PROJECT_PRINCIPLES.md): AI 설계 철학과 핵심 원리 해설
- [`docs/AI_CODE_DEEP_DIVE.md`](docs/AI_CODE_DEEP_DIVE.md): AI 핵심 코드 분석 인덱스
- [`docs/AI_COMPONENTS_AND_IMPLEMENTATION_KR.md`](docs/AI_COMPONENTS_AND_IMPLEMENTATION_KR.md): AI 구성요소와 현재 구현 상태(한국어)
- [`docs/FINAL_DEVELOPMENT_REPORT_KR.md`](docs/FINAL_DEVELOPMENT_REPORT_KR.md): 최종 개발 결과 정리(한국어)
- [`docs/POST_30D_NEXT_STEPS_KR.md`](docs/POST_30D_NEXT_STEPS_KR.md): 30일 데이터 달성 후 운영 절차(한국어)
- [`docs/POST_30D_NEXT_STEPS.md`](docs/POST_30D_NEXT_STEPS.md): Post-30-day operational runbook (English)
- [`docs/AUTO_TRAINING_WORKFLOW_KR.md`](docs/AUTO_TRAINING_WORKFLOW_KR.md): 자동 학습/재학습 및 결과물 생성 워크플로우(한국어)
- [`docs/AUTO_TRAINING_WORKFLOW.md`](docs/AUTO_TRAINING_WORKFLOW.md): Auto training/retraining and artifact workflow (English)
- [`docs/RL_TRAINING_DEEP_DIVE.md`](docs/RL_TRAINING_DEEP_DIVE.md): 강화학습 학습 루프 심층 분석 — MDP 정식화, 손실 함수, 품질 게이트 상세 해설
- [`docs/WORLD_MODEL_THEORY_AND_IMPLEMENTATION.md`](docs/WORLD_MODEL_THEORY_AND_IMPLEMENTATION.md): World Model 이론과 GPU Advisor 구현 — RL 기초, Gaussian NLL, 30d 학습 결과

### `docs/` Learning Pairs (EN/KR)

| Topic | English | Korean |
|-------|---------|--------|
| Hyperparameter Design | [docs/HYPERPARAMETER_GUIDE.md](docs/HYPERPARAMETER_GUIDE.md) | [docs/HYPERPARAMETER_GUIDE_KR.md](docs/HYPERPARAMETER_GUIDE_KR.md) |
| MCTS Numerical Walkthrough | [docs/MCTS_WALKTHROUGH.md](docs/MCTS_WALKTHROUGH.md) | [docs/MCTS_WALKTHROUGH_KR.md](docs/MCTS_WALKTHROUGH_KR.md) |
| Safety Mechanisms | [docs/SAFETY_MECHANISMS.md](docs/SAFETY_MECHANISMS.md) | [docs/SAFETY_MECHANISMS_KR.md](docs/SAFETY_MECHANISMS_KR.md) |
| Inference Walkthrough | [docs/INFERENCE_WALKTHROUGH.md](docs/INFERENCE_WALKTHROUGH.md) | [docs/INFERENCE_WALKTHROUGH_KR.md](docs/INFERENCE_WALKTHROUGH_KR.md) |
| Glossary | [docs/GLOSSARY.md](docs/GLOSSARY.md) | [docs/GLOSSARY_KR.md](docs/GLOSSARY_KR.md) |

### `docs/models/` Deep Dives

- [`docs/models/01_representation_network.md`](docs/models/01_representation_network.md): Representation Network (`h`) — 256D input, residual FeedForward, GELU
- [`docs/models/02_dynamics_network.md`](docs/models/02_dynamics_network.md): Dynamics Network (`g`) — 상태전이 + Gaussian reward (μ, log σ²)
- [`docs/models/03_prediction_network.md`](docs/models/03_prediction_network.md): Prediction Network (`f`) — 정책/가치 출력, 잔차 블록
- [`docs/models/04_mcts_engine.md`](docs/models/04_mcts_engine.md): MCTS 탐색 엔진 — PUCT, Dirichlet noise, 5-step rollout
- [`docs/models/05_transformer_model.md`](docs/models/05_transformer_model.md): Transformer 모듈 구조 및 사용 방식 분석
- [`docs/models/06_action_model.md`](docs/models/06_action_model.md): Action Model (`a`) — ActionEmbeddingLayer 16D, ActionPriorNetwork 256→5, 정책 보정 통합

### `docs/reports/` Release Reports

- [`docs/reports/YYYY-MM-DD/release_report_*.md`](docs/reports): 릴리즈 판정 자동 생성 보고서(일자별 폴더)
- [`docs/reports/YYYY-MM-DD/data_status_*.md`](docs/reports): 일일 크롤링 후 실제 데이터 파일 기준 상태 보고서(일자별 폴더)
- [`docs/reports/latest_data_status.md`](docs/reports/latest_data_status.md): 최신 일일 데이터 상태 요약(자동 갱신)
- [`docs/reports/latest_release_report.md`](docs/reports/latest_release_report.md): 최신 릴리즈 판정 요약(자동 갱신)

## 🔄 Roadmap

- **Day 1** ✅: System setup, initial data collection
- **Day 30** ✅ (2026-03-22 완료): 30-day real-data window → auto training + quality gate check → all 7 gates PASS
- **Release** ✅ (2026-03-22): Tag `release-agent-20260322-105138` pushed — directional accuracy 89.4%, win_rate 75%
- **Post-30d** (진행 중): Auto-retrain every 7 days as new data accumulates (`retrain_every_days=7`)
- **Day 60+**: Stable production-ready predictions with real-market-trained model

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **AI Framework** | PyTorch ≥ 2.0 (Apple MPS acceleration) |
| **RL Architecture** | MuZero-style (h + g + f + a + MCTS) |
| **Web Framework** | FastAPI + uvicorn |
| **Frontend** | Next.js 16 + React 19 + Tailwind CSS |
| **Data Processing** | NumPy, scikit-learn, pandas |
| **Crawling** | Requests, BeautifulSoup4 |
| **Automation** | macOS LaunchAgent (daily @ 00:00) |
| **Containerization** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions (Python 3.10/3.11/3.12, Node 20) |
| **Persistence** | SQLite (default) / PostgreSQL (optional) |
| **Auth** | JWT (OAuth2) + API Key + Hybrid mode |

## 🤖 Development Tools & AI Assistance

This project was developed with assistance from multiple AI tools, demonstrating the collaborative future of software development:

- **GLM (GLM-4.7)**: Used for initial architecture design and Korean documentation
- **Antigravity**: Assisted with code generation and feature engineering implementation
- **Claude Code**: Provided development assistance, code review, and documentation refinement
- **Codex**: Provided development assistance, code review, and documentation refinement

**Transparency Note**: As part of the 0.1B AI Project philosophy, I believe in honest disclosure of development tools. This project showcases human-AI collaboration, where AI assists in accelerating development while the core design decisions, architecture, and problem-solving approach remain human-driven.

## 📊 Model Specifications

- **Total Parameters**: ~19M
- **Representation Network h**: ~6.4M params (256D latent, 3× FeedForward with residual connections, GELU)
- **Dynamics Network g**: ~6.5M params (state transition + Gaussian reward μ/log σ², 4 residual blocks)
- **Prediction Network f**: ~6.0M params (policy + value, 4 residual blocks)
- **Action Model a**: ~43K params (ActionEmbeddingLayer 16D + ActionPriorNetwork 256→128→64→5)
- **MCTS Simulations**: 50 per decision (PUCT formula, Dirichlet noise ε=0.25 α=0.03, rollout depth=5)
- **Policy Calibration**: 4-signal blend — MCTS 45% + Reward 25% + f-net prior 15% + ActionModel 15%

## 🔢 Token Note (30-Day Trained Agent)

Even after the `Day 30` learning window is secured, the core agent in this project (`backend/agent/*`) is not a text-generating LLM.
In other words, **the inference engine itself does not generate text tokens**; it makes decisions using a 256D latent state and MCTS simulations.

To avoid confusion around the term "token," operational estimates are documented below.

| Item | Meaning | Estimated per request |
|------|------|----------------------|
| Agent Inference Token | Text tokens generated during AlphaZero/MCTS inference | **0** (no text generation) |
| Auth Token (JWT) | Authentication token string issued by `/api/auth/token` | about **220-420 chars** |
| Optional LLM Output Token | Only occurs when an external LLM is added for natural-language explanations | about **300-900 tokens** combined (input + output) |

Operational notes:
- In the 30-day-trained state, the default `/api/ask` path is non-generative inference, so cost/latency is mainly driven by model inference time and API processing time.
- Token-related overhead is limited to JWT string issuance/transfer (minimal) or optional external LLM integration.

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ask` | POST | Get agent decision for GPU model |
| `/api/training/start` | POST | Start AI training |
| `/api/training/stop` | POST | Stop training |
| `/api/training/metrics` | GET | Get training metrics |
| `/api/agent/readiness` | GET | Check 30-day data readiness |
| `/api/agent/next-steps` | GET | Get bilingual next-step checklist for pre/post 30-day workflow |
| `/api/agent/evaluate` | GET | Run backtest evaluation |
| `/api/agent/release-check` | GET | Run release quality gates |
| `/api/agent/model-info` | GET | Check loaded model metadata |
| `/api/agent/pipeline/run` | POST | Run full release pipeline (readiness→train→evaluate→report) |
| `/api/system/status` | GET | System status |
| `/docs` | GET | Swagger UI documentation |

## ✅ One-Click Release (After 30 Days)

When 30-day crawling is complete, run the full release flow from CLI:

```bash
python3 backend/run_release_ready.py
```

Options:

```bash
# Validate gates without forcing 30-day window (for dry run)
python3 backend/run_release_ready.py --allow-short-window --no-train --lookback-days 7

# On pass, create and push release tag
python3 backend/run_release_ready.py --tag --push-tag
```

## 📝 Disclaimer

1. **Educational Purpose Only**
   The codebase and architecture in this repository are a solo project and reference material for education and research, implementing an MCTS (Monte Carlo Tree Search)-based AI agent pipeline (crawling-learning-inference). It does not guarantee production-level integrity for commercial services.

2. **No Financial Liability**
   If this project is modified or applied to price prediction or automated trading of stocks, cryptocurrencies, or physical assets, all financial losses and legal liabilities are solely the responsibility of the user running the code. The original author provides no warranty and accepts no liability for any derived outcomes.

3. **Web Scraping Liability**
   The included data crawling pipeline is example code intended to demonstrate system operation. Any legal disputes arising from collecting data from third-party websites, including Terms of Service violations, IP blocking, or service disruption due to server load, are the sole responsibility of the executor. You must independently review and comply with each target website's `robots.txt` and Terms of Service.

4. **Algorithm Attribution**
   The state-transition simulation and tree-search logic in this system are independently implemented by borrowing concepts from widely published reinforcement learning architectures such as DeepMind's AlphaZero and MuZero. Rights to the underlying algorithmic concepts belong to their original authors.

## 👤 Author

**Jin Younghwa**
Email: timotolkie@gmail.com

Creator of the 0.1B AI Project series, exploring how thoughtful architecture and smart feature engineering can deliver powerful AI solutions without requiring billions of parameters.

## 🤝 Contributing

This is a personal research project. Feel free to fork and experiment!

---

**Last Updated**: 2026-03-22
**Version**: 0.4.0 (First Production Release — `release-agent-20260322-105138`)
**Project Type**: 0.1B AI Project
