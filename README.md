# GPU Advisor

**AI-Powered GPU Purchase Timing Prediction System**

A sophisticated AI system that predicts optimal GPU purchase timing using AlphaZero/MuZero architecture, mimicking the win-rate calculation approach used in Go.

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
# Python 3.8+
pip install -r requirements.txt
```

### Setup

1. **Configure Automated Data Collection**

```bash
cd /Users/younghwa.jin/Documents/gpu-advisor
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

### Making Predictions

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "RTX 5060", "action": "query"}'
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
├── crawlers/                      # Data collection modules
│   ├── danawa_crawler.py          # GPU price crawler
│   ├── exchange_rate_crawler.py   # Exchange rate fetcher
│   ├── news_crawler.py            # News crawler
│   ├── feature_engineer.py        # 256D feature generation
│   └── run_daily.py               # Daily orchestration script
│
├── backend/                       # AI & API backend
│   ├── simple_server.py           # FastAPI server
│   ├── models/                    # AlphaZero networks
│   │   ├── representation_network.py
│   │   ├── dynamics_network.py
│   │   ├── prediction_network.py
│   │   └── mcts.py
│   └── data/                      # Data processing
│
├── data/                          # Data storage
│   ├── raw/                       # Raw collected data
│   │   ├── danawa/
│   │   ├── exchange/
│   │   └── news/
│   └── processed/                 # Processed features
│       └── dataset/
│
├── logs/                          # System logs
│
├── setup_cron.sh                  # Cron automation setup
├── CRAWLER_GUIDE.md               # Crawler documentation (Korean)
├── GPU_PURCHASE_ADVISOR_REPORT.md # System report (Korean)
└── 종합_프로젝트_보고서.md          # Complete guide (Korean)
```

## 📖 Documentation

- **English**: This README
- **Korean**:
  - `종합_프로젝트_보고서.md` - Complete system guide
  - `CRAWLER_GUIDE.md` - Crawler usage guide
  - `GPU_PURCHASE_ADVISOR_REPORT.md` - System evaluation report

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

- **GLM (GLM-4-Plus)**: Used for initial architecture design and Korean documentation
- **Antigravity**: Assisted with code generation and feature engineering implementation
- **Claude Code**: Provided development assistance, code review, and documentation refinement

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
| `/api/predict` | POST | Get purchase timing prediction |
| `/api/training/start` | POST | Start AI training |
| `/api/training/stop` | POST | Stop training |
| `/api/training/metrics` | GET | Get training metrics stream |
| `/api/system/status` | GET | System status |
| `/docs` | GET | Swagger UI documentation |

## 📝 License

This project is for educational and research purposes.

## 👤 Author

**Jin Younghwa**
Email: timotolkie@gmail.com

Creator of the 0.1B AI Project series, exploring how thoughtful architecture and smart feature engineering can deliver powerful AI solutions without requiring billions of parameters.

## 🤝 Contributing

This is a personal research project. Feel free to fork and experiment!

---

**Last Updated**: 2026-02-15
**Version**: 1.0.0
**Project Type**: 0.1B AI Project
