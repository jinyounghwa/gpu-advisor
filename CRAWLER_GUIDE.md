# 크롤러 시스템 사용 가이드
## GPU 구매 타이밍 AI - 자동 데이터 수집

---

## 📋 개요

이 크롤러 시스템은 **매일 자동으로** GPU 가격, 환율, 뉴스를 수집하고 **256차원 Feature**를 생성합니다.
또한 실행 직후 `docs/reports/`에 실제 수집 파일 기준 상태 보고서를 자동 생성합니다.

### 수집 데이터
1. **다나와 GPU 가격** - 24개 모델 실시간 최저가
2. **환율 정보** - USD/KRW, JPY/KRW, EUR/KRW
3. **GPU 뉴스** - 감정 분석 포함
4. **256차원 Feature** - AlphaZero 학습용

---

## 🚀 빠른 시작

### 1단계: Cron 자동화 설정 (한 번만 실행)

```bash
cd /Users/younghwa.jin/Documents/gpu-advisor
./setup_cron.sh
```

**설정 완료!** 매일 자정 00:00에 자동으로 실행됩니다.

### 2단계: 수동 실행 (즉시 테스트)

```bash
python3 crawlers/run_daily.py
```

---

## 📁 파일 구조

```
gpu-advisor/
├── crawlers/                      # 크롤러 모듈
│   ├── danawa_crawler.py          # 다나와 GPU 가격
│   ├── exchange_rate_crawler.py   # 환율
│   ├── news_crawler.py            # 뉴스
│   ├── feature_engineer.py        # Feature 생성 (256차원)
│   └── run_daily.py               # 일일 실행 스크립트
│
├── data/                          # 데이터 저장소
│   ├── raw/                       # 원시 데이터
│   │   ├── danawa/                # GPU 가격
│   │   ├── exchange/              # 환율
│   │   └── news/                  # 뉴스
│   │
│   └── processed/                 # 처리된 데이터
│       └── dataset/               # 256차원 Feature
│
├── logs/                          # 로그 파일
│   ├── cron.log                   # Cron 실행 로그
│   └── daily_crawl.log            # 상세 로그
│
├── docs/reports/                  # 자동 보고서
│   ├── data_status_*.json         # 일일 데이터 상태(시점별)
│   ├── data_status_*.md           # 일일 데이터 상태(시점별)
│   ├── latest_data_status.json    # 최신 상태(고정 파일)
│   └── latest_data_status.md      # 최신 상태(고정 파일)
│
└── setup_cron.sh                  # Cron 설정 스크립트
```

---

## 🔄 Cron 스케줄

### 기본 설정
- **실행 시간**: 매일 자정 00:00
- **실행 내용**:
  1. 다나와 GPU 가격 크롤링 (24개 모델)
  2. 환율 정보 수집
  3. GPU 뉴스 크롤링 및 감정 분석
  4. 256차원 Feature 생성

### Cron 관리 명령어

```bash
# Cron job 목록 보기
crontab -l

# Cron job 편집
crontab -e

# 특정 Cron job 제거
crontab -l | grep -v "run_daily.py" | crontab -

# 모든 Cron job 삭제
crontab -r
```

### 실행 시간 변경

```bash
# Cron 편집
crontab -e

# 예시: 매일 오전 6시 실행
0 6 * * * cd /Users/younghwa.jin/Documents/gpu-advisor && python3 crawlers/run_daily.py >> logs/cron.log 2>&1

# 예시: 매일 12시간마다 (00:00, 12:00)
0 0,12 * * * cd /Users/younghwa.jin/Documents/gpu-advisor && python3 crawlers/run_daily.py >> logs/cron.log 2>&1

# 예시: 매 6시간마다
0 */6 * * * cd /Users/younghwa.jin/Documents/gpu-advisor && python3 crawlers/run_daily.py >> logs/cron.log 2>&1
```

---

## 📊 데이터 형식

### 1. 다나와 GPU 가격 (data/raw/danawa/YYYY-MM-DD.json)

```json
{
  "date": "2026-02-14",
  "source": "danawa",
  "total_products": 24,
  "products": [
    {
      "product_name": "MSI 지포스 RTX 5060 벤투스 2X OC D7 8GB",
      "manufacturer": "MSI",
      "chipset": "RTX 5060",
      "lowest_price": 606320,
      "seller_count": 15,
      "stock_status": "in_stock",
      "product_url": "https://prod.danawa.com/info/?pcode=90956033"
    }
  ]
}
```

### 2. 환율 (data/raw/exchange/YYYY-MM-DD.json)

```json
{
  "date": "2026-02-14",
  "source": "exchange_rate_api",
  "rates": {
    "USD/KRW": 1442.7,
    "JPY/KRW": 943.28,
    "EUR/KRW": 1560.5
  }
}
```

### 3. 뉴스 (data/raw/news/YYYY-MM-DD.json)

```json
{
  "date": "2026-02-14",
  "source": "google_news_rss",
  "total_articles": 5,
  "articles": [
    {
      "title": "GPU 가격 하락 전망",
      "url": "https://news.example.com/...",
      "published_at": "2026-02-14T10:30:00",
      "sentiment": "positive",
      "sentiment_score": 0.75,
      "keywords": ["GPU price", "price drop"]
    }
  ],
  "statistics": {
    "total": 5,
    "sentiment_avg": 0.42,
    "positive_count": 3,
    "negative_count": 1,
    "neutral_count": 1
  }
}
```

### 4. 256차원 Feature (data/processed/dataset/training_data_YYYY-MM-DD.json)

```json
[
  {
    "date": "2026-02-14",
    "gpu_model": "RTX 5060",
    "state_vector": [0.0606, 0.062, 0.065, ... (256개 값)]
  }
]
```

**Feature 구성 (256차원):**
- 가격 Feature (60차원): 이동평균, 변화율, 변동성, 추세
- 환율 Feature (20차원): USD/KRW, JPY/KRW, EUR/KRW
- 뉴스 Feature (30차원): 감정 분석, 기사 수
- 시장 Feature (20차원): 판매자 수, 재고 상황
- 시간 Feature (20차원): 요일, 월, 계절성
- 기술 지표 (106차원): RSI, MACD, 모멘텀

---

## 🧪 테스트

### 즉시 실행 테스트

```bash
# 전체 파이프라인 실행
python3 crawlers/run_daily.py

# 개별 크롤러 테스트
python3 crawlers/danawa_crawler.py
python3 crawlers/exchange_rate_crawler.py
python3 crawlers/news_crawler.py
python3 crawlers/feature_engineer.py
```

### 로그 확인

```bash
# Cron 실행 로그
tail -f logs/cron.log

# 상세 로그
tail -f logs/daily_crawl.log

# 최근 100줄
tail -n 100 logs/daily_crawl.log

# 최신 자동 상태 보고서
cat docs/reports/latest_data_status.json | python3 -m json.tool
cat docs/reports/latest_data_status.md
```

### 데이터 확인

```bash
# 오늘 수집된 데이터
cat data/raw/danawa/$(date +%Y-%m-%d).json | python3 -m json.tool
cat data/raw/exchange/$(date +%Y-%m-%d).json | python3 -m json.tool
cat data/raw/news/$(date +%Y-%m-%d).json | python3 -m json.tool

# 256차원 Feature
cat data/processed/dataset/training_data_$(date +%Y-%m-%d).json | python3 -m json.tool
```

---

## ⚙️ 설정 변경

### 수집 대상 GPU 모델 추가

**파일:** `crawlers/danawa_crawler.py`

```python
self.target_gpus = [
    "RTX 5090",
    "RTX 5080",
    # ... 기존 모델들
    "RTX 3060",  # ← 추가
    "RX 6700 XT",  # ← 추가
]
```

### Feature 차원 변경

**파일:** `crawlers/feature_engineer.py`

```python
# 현재: 256차원
# 변경하려면 각 Feature 함수의 반환 차원 수정
def calculate_price_features(self, ...):
    # ...
    while len(features) < 100:  # ← 60에서 100으로
        features.append(0.0)
    return features[:100]
```

---

## 📈 데이터 축적 현황

### 최소 필요 데이터
- **30일 이상** (AlphaZero 학습용)
- **GPU 모델당 30개 이상 샘플**

### 현재 진행 상황 확인

```bash
# 수집된 날짜 수
ls -1 data/raw/danawa/*.json | wc -l

# 총 샘플 수
python3 << EOF
import json
from pathlib import Path

total = 0
for file in Path("data/raw/danawa").glob("*.json"):
    with open(file) as f:
        data = json.load(f)
        total += len(data.get("products", []))

print(f"총 샘플 수: {total}개")
print(f"필요 샘플: 3,000개 (30일 × 100개)")
print(f"진행률: {total/3000*100:.1f}%")
EOF
```

---

## 🐛 문제 해결

### 1. Cron이 실행되지 않음

```bash
# Cron 서비스 상태 확인 (Linux)
sudo systemctl status cron

# macOS에서는 자동 실행됨 (확인만)
crontab -l
```

### 2. Python 경로 오류

```bash
# Python3 경로 확인
which python3

# Cron job에서 절대 경로 사용
# setup_cron.sh 실행 시 자동으로 설정됨
```

### 3. 권한 오류

```bash
# 실행 권한 부여
chmod +x crawlers/run_daily.py
chmod +x setup_cron.sh

# 디렉토리 권한 확인
ls -la data/
ls -la logs/
```

### 4. 로그 확인

```bash
# 최근 오류 확인
tail -n 50 logs/daily_crawl.log | grep ERROR
```

---

## 🚀 30일 데이터 수집 후

### 1. 모델 재학습

```bash
python3 backend/train_alphazero_v2.py \
    --data_dir data/processed/dataset \
    --output alphazero_model_256d.pth \
    --epochs 100
```

### 2. 백테스팅

```bash
python3 backend/backtest.py \
    --model alphazero_model_256d.pth \
    --test_days 7
```

### 3. 실전 배포

```bash
python3 backend/run_server.py
```

---

## 📞 도움말

### 크롤러 작동 확인

```bash
# 전체 시스템 테스트
python3 crawlers/run_daily.py

# 성공 시 출력:
# [1/4] 다나와 GPU 가격 크롤링
# ✓ RTX 5060: 606,320원
# ...
# [2/4] 환율 정보 수집
# ✓ USD/KRW: 1442.7
# ...
# [3/4] GPU 뉴스 크롤링
# ✓ 총 5개 기사 수집
# ...
# [4/4] Feature Engineering (256차원)
# ✓ Feature 생성 완료: 24개 샘플
```

### Cron 작동 확인

```bash
# 다음날 오전 확인
cat logs/cron.log

# 정상 실행 시:
# 2026-02-15 00:00:01 - INFO - 일일 데이터 수집 시작
# 2026-02-15 00:00:15 - INFO - ✓ 일일 데이터 수집 완료!
```

---

## ✅ 체크리스트

- [ ] Cron 설정 완료 (`./setup_cron.sh`)
- [ ] 수동 실행 테스트 (`python3 crawlers/run_daily.py`)
- [ ] 데이터 파일 생성 확인 (`ls data/raw/danawa/`)
- [ ] 256차원 Feature 생성 확인 (`ls data/processed/dataset/`)
- [ ] 로그 확인 (`tail logs/daily_crawl.log`)
- [ ] 30일 대기 (자동 수집)
- [ ] 모델 재학습
- [ ] 백테스팅
- [ ] 실전 배포

---

**작성:** 2026-02-14
**버전:** 1.0
**프로젝트:** GPU Purchase Timing Advisor
