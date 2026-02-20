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

- 🤖 **AlphaZero Architecture**: 18.9M parameters (Representation, Dynamics, Prediction networks + MCTS)
- 📊 **Automated Data Collection**: Daily crawling of GPU prices, exchange rates, and news
- 🧠 **256-Dimensional Features**: Rich feature engineering from 11D to 256D
- 📈 **Real-time Predictions**: REST API for instant purchase timing recommendations
- ⏰ **Cron Automation**: Fully automated daily data collection

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
   - Representation Network (h): Encodes market state → latent state
   - Dynamics Network (g): Predicts next state + reward
   - Prediction Network (f): Outputs policy + value
   - MCTS: Simulates future scenarios for optimal decisions

3. **Backend Server**
   - FastAPI REST API
   - Real-time predictions
   - Training dashboard
   - Swagger UI documentation

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

3. **Start Backend Server**

```bash
cd backend
python3 simple_server.py
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

### Making Predictions

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"model_name": "RTX 5060"}'
```

## 📊 Data Pipeline

```
Crawlers (Daily @ 00:00)
  ↓
Raw Data Collection
  ├─ GPU Prices (Danawa)
  ├─ Exchange Rates
  └─ News + Sentiment
  ↓
Feature Engineering (11D → 256D)
  ↓
Training Dataset
  ↓
AlphaZero Training
  ↓
Purchase Predictions
```

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
│       ├── representation_network.py  # h(s): State encoder
│       ├── dynamics_network.py        # g(s,a): World model
│       ├── prediction_network.py      # f(s): Policy-value
│       └── mcts_engine.py             # MCTS tree search
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

### Learning Guides (English / Korean)

| Topic | English | Korean |
|-------|---------|--------|
| Hyperparameter Design | [HYPERPARAMETER_GUIDE.md](docs/HYPERPARAMETER_GUIDE.md) | [KR](docs/HYPERPARAMETER_GUIDE_KR.md) |
| MCTS Numerical Walkthrough | [MCTS_WALKTHROUGH.md](docs/MCTS_WALKTHROUGH.md) | [KR](docs/MCTS_WALKTHROUGH_KR.md) |
| Safety Mechanisms | [SAFETY_MECHANISMS.md](docs/SAFETY_MECHANISMS.md) | [KR](docs/SAFETY_MECHANISMS_KR.md) |
| Inference Walkthrough | [INFERENCE_WALKTHROUGH.md](docs/INFERENCE_WALKTHROUGH.md) | [KR](docs/INFERENCE_WALKTHROUGH_KR.md) |
| Glossary | [GLOSSARY.md](docs/GLOSSARY.md) | [KR](docs/GLOSSARY_KR.md) |

### Project Documentation

- `종합_프로젝트_보고서.md` - Complete system guide (Korean)
- `CRAWLER_GUIDE.md` - Crawler usage guide (Korean)
- `GPU_PURCHASE_ADVISOR_REPORT.md` - System evaluation report (Korean)
- `docs/STUDY_GUIDE.md` - 4-week learning curriculum
- `docs/IMPLEMENTATION_GUIDE.md` - Step-by-step implementation
- `docs/PROJECT_PRINCIPLES.md` - Core design principles

## 🔄 Roadmap

- **Day 1** (Current): System setup, initial data collection
- **Day 30**: 720 samples collected → Begin AI training
- **Day 60+**: Production-ready predictions

## 🛠️ Technology Stack

- **AI Framework**: PyTorch (with Apple MPS acceleration)
- **Web Framework**: FastAPI
- **Data Processing**: NumPy, Pandas, scikit-learn
- **Crawling**: Requests, BeautifulSoup4
- **Automation**: Cron

## 🤖 Development Tools & AI Assistance

This project was developed with assistance from multiple AI tools, demonstrating the collaborative future of software development:

- **GLM (GLM-4.7)**: Used for initial architecture design and Korean documentation
- **Antigravity**: Assisted with code generation and feature engineering implementation
- **Claude Code**: Provided development assistance, code review, and documentation refinement
- **Codex**: Provided development assistance, code review, and documentation refinement

**Transparency Note**: As part of the 0.1B AI Project philosophy, I believe in honest disclosure of development tools. This project showcases human-AI collaboration, where AI assists in accelerating development while the core design decisions, architecture, and problem-solving approach remain human-driven.

## 📊 Model Specifications

- **Total Parameters**: 18.9M
- **Representation Network**: 6.4M params (256D latent state)
- **Dynamics Network**: 6.5M params (state transition + reward)
- **Prediction Network**: 6.0M params (policy + value)
- **MCTS Simulations**: 50 per decision

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ask` | POST | Get agent decision for GPU model |
| `/api/training/start` | POST | Start AI training |
| `/api/training/stop` | POST | Stop training |
| `/api/training/metrics` | GET | Get training metrics |
| `/api/agent/readiness` | GET | Check 30-day data readiness |
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

This project is for **educational and research purposes only**. It uses algorithms inspired by AlphaGo Zero / MuZero (DeepMind) for studying reinforcement learning applications in market analysis. Not intended for commercial use or financial advice.

## 👤 Author

**Jin Younghwa**
Email: timotolkie@gmail.com

Creator of the 0.1B AI Project series, exploring how thoughtful architecture and smart feature engineering can deliver powerful AI solutions without requiring billions of parameters.

## 🤝 Contributing

This is a personal research project. Feel free to fork and experiment!

---

**Last Updated**: 2026-02-21
**Version**: 0.2.0
**Project Type**: 0.1B AI Project
