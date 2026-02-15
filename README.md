# GPU Advisor

**AI-Powered GPU Purchase Timing Prediction System**

A sophisticated AI system that predicts optimal GPU purchase timing using AlphaZero/MuZero architecture, mimicking the win-rate calculation approach used in Go.

## 🎯 Overview

This system helps answer the question: **"Should I buy this GPU now or wait?"**

Just like AlphaGo calculates win probabilities in Go, this system calculates **purchase profitability scores (0-100%)** to determine the best time to buy GPUs.

### Key Features

- 🤖 **AlphaZero Architecture**: 18.9M parameters (Representation, Dynamics, Prediction networks + MCTS)
- 📊 **Automated Data Collection**: Daily crawling of GPU prices, exchange rates, and news
- 🧠 **256-Dimensional Features**: Rich feature engineering from 11D to 256D
- 📈 **Real-time Predictions**: REST API for instant purchase timing recommendations
- ⏰ **Cron Automation**: Fully automated daily data collection

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

Built with Claude Code CLI

## 🤝 Contributing

This is a personal research project. Feel free to fork and experiment!

---

**Last Updated**: 2026-02-14
**Version**: 1.0.0
