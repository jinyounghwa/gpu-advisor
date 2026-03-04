# 크롤러 시스템 사용 가이드
## GPU 구매 타이밍 AI - 자동 데이터 수집

---

## 📋 개요

이 크롤러 시스템은 **매일 자동으로** GPU 가격, 환율, 뉴스를 수집하고 **256차원 Feature**를 생성합니다.
실행 직후 `docs/reports/`에 실제 수집 파일 기준 상태 보고서도 자동 생성됩니다.

### 수집 데이터
1. **다나와 GPU 가격** - 대상 GPU 모델 최저가/판매자수/재고 상태
2. **환율 정보** - USD/KRW, JPY/KRW, EUR/KRW
3. **GPU 뉴스** - 기사 수/감정 통계 포함
4. **256차원 Feature** - 에이전트 입력 데이터(`training_data_YYYY-MM-DD.json`)

---

## 🚀 빠른 시작

### 1단계: Cron 자동화 설정 (한 번만 실행)

```bash
cd /Users/younghwa.jin/Documents/gpu-advisor
./setup_cron.sh
```

기본 스케줄은 **매일 00:00**입니다.

### 2단계: 수동 실행 (즉시 테스트)

```bash
python3 crawlers/run_daily.py
```

---

## 📁 파일 구조

```text
gpu-advisor/
├── crawlers/
│   ├── danawa_crawler.py
│   ├── exchange_rate_crawler.py
│   ├── news_crawler.py
│   ├── feature_engineer.py
│   ├── auto_training.py
│   ├── status_report.py
│   └── run_daily.py
├── data/
│   ├── raw/
│   │   ├── danawa/
│   │   ├── exchange/
│   │   └── news/
│   └── processed/
│       └── dataset/
├── logs/
│   ├── cron.log
│   └── daily_crawl.log
├── docs/reports/
│   ├── YYYY-MM-DD/
│   │   ├── data_status_*.json
│   │   └── data_status_*.md
│   ├── latest_data_status.json
│   ├── latest_data_status.md
│   ├── latest_release_report.json
│   ├── latest_release_report.md
│   ├── latest_auto_training_status.json
│   └── latest_auto_training_status.md
└── setup_cron.sh
```

---

## 🔄 Cron 스케줄

### 기본 설정
- 실행 시간: 매일 00:00
- 실행 순서:
1. 다나와 GPU 가격 크롤링
2. 환율 정보 수집
3. GPU 뉴스 크롤링
4. 256차원 Feature 생성
5. 상태 보고서 생성
6. 자동 학습 오케스트레이션 실행
  - 30일 미달: 릴리즈 드라이 체크(`backend/run_release_daily.py`)
  - 30일 도달/재학습 간격 충족: 학습+평가(`backend/run_release_ready.py`)
7. 자동화/릴리즈 결과 보고서 생성
8. `/usr/bin/time -l`로 메모리/시간 지표를 `logs/cron.log`에 기록

### Cron 관리 명령어

```bash
crontab -l
crontab -e
crontab -l | grep -v "run_daily.py" | crontab -
crontab -r
```

### 실행 시간 변경 예시

```bash
# 매일 06:00
0 6 * * * cd /Users/younghwa.jin/Documents/gpu-advisor && /usr/bin/time -l python3 crawlers/run_daily.py >> logs/cron.log 2>&1

# 매일 00:00, 12:00
0 0,12 * * * cd /Users/younghwa.jin/Documents/gpu-advisor && /usr/bin/time -l python3 crawlers/run_daily.py >> logs/cron.log 2>&1
```

---

## 📊 데이터 형식

### 1) 다나와 (`data/raw/danawa/YYYY-MM-DD.json`)

```json
{
  "date": "2026-02-21",
  "source": "danawa",
  "total_products": 24,
  "products": [
    {
      "product_name": "...",
      "manufacturer": "...",
      "chipset": "RTX 5060",
      "lowest_price": 606320,
      "seller_count": 15,
      "stock_status": "in_stock",
      "product_url": "https://prod.danawa.com/info/?pcode=..."
    }
  ],
  "metadata": {
    "crawled_at": "2026-02-21T00:00:00",
    "target_models": 24
  }
}
```

### 2) 환율 (`data/raw/exchange/YYYY-MM-DD.json`)

```json
{
  "date": "2026-02-21",
  "source": "exchange_rate_api",
  "rates": {
    "USD/KRW": 1442.7,
    "JPY/KRW": 943.28,
    "EUR/KRW": 1560.5
  }
}
```

### 3) 뉴스 (`data/raw/news/YYYY-MM-DD.json`)

```json
{
  "date": "2026-02-21",
  "source": "google_news_rss",
  "total_articles": 5,
  "statistics": {
    "total": 5,
    "sentiment_avg": 0.42,
    "positive_count": 3,
    "negative_count": 1,
    "neutral_count": 1
  }
}
```

### 4) Feature (`data/processed/dataset/training_data_YYYY-MM-DD.json`)

```json
[
  {
    "date": "2026-02-21",
    "gpu_model": "RTX 5060",
    "state_vector": [0.0606, 0.062, 0.065]
  }
]
```

`state_vector` 길이는 256입니다.

---

## 🧪 테스트 및 점검

### 전체 파이프라인 실행

```bash
python3 crawlers/run_daily.py
```

### 자동 학습 비활성화(드라이 체크만)

```bash
python3 crawlers/run_daily.py --disable-auto-train
```

### 재학습 간격 임시 오버라이드

```bash
python3 crawlers/run_daily.py --auto-retrain-days 5
```

### 로그 확인

```bash
tail -f logs/cron.log
tail -f logs/daily_crawl.log
tail -n 100 logs/daily_crawl.log
```

### 자동 상태 보고서 확인

```bash
cat docs/reports/latest_data_status.json | python3 -m json.tool
cat docs/reports/latest_data_status.md
ls -la docs/reports/$(date +%Y-%m-%d)
```

### 당일 파일 확인

```bash
cat data/raw/danawa/$(date +%Y-%m-%d).json | python3 -m json.tool
cat data/raw/exchange/$(date +%Y-%m-%d).json | python3 -m json.tool
cat data/raw/news/$(date +%Y-%m-%d).json | python3 -m json.tool
cat data/processed/dataset/training_data_$(date +%Y-%m-%d).json | python3 -m json.tool
```

---

## ⚙️ 설정 변경

### 수집 대상 GPU 모델 변경
- 파일: `crawlers/danawa_crawler.py`
- `self.target_gpus` 목록 수정

### Feature 차원 변경
- 파일: `crawlers/feature_engineer.py`
- 각 블록(가격/환율/뉴스/시장/시간/기술지표) 차원과 최종 합계가 일치하도록 함께 수정

### 자동 학습 정책 변경(환경변수)

```bash
export GPU_ADVISOR_AUTO_TRAIN_ENABLED=true
export GPU_ADVISOR_AUTO_TRAIN_TARGET_DAYS=30
export GPU_ADVISOR_AUTO_RETRAIN_EVERY_DAYS=7
export GPU_ADVISOR_AUTO_TRAIN_STEPS=500
export GPU_ADVISOR_AUTO_TRAIN_BATCH_SIZE=32
export GPU_ADVISOR_AUTO_TRAIN_LR=0.0001
export GPU_ADVISOR_AUTO_TRAIN_SEED=42
export GPU_ADVISOR_AUTO_TRAIN_LOOKBACK_DAYS=30
export GPU_ADVISOR_AUTO_TRAIN_TIMEOUT_SEC=5400
```

---

## 📈 누적 데이터 확인

```bash
# 소스별 날짜 수
ls -1 data/raw/danawa/*.json | wc -l
ls -1 data/raw/exchange/*.json | wc -l
ls -1 data/raw/news/*.json | wc -l
ls -1 data/processed/dataset/training_data_*.json | wc -l

# 최신 상태 보고서의 준비도 확인
cat docs/reports/latest_data_status.json | python3 -m json.tool
```

30일 준비 여부는 상태 보고서의 `ready_for_30d_training` 필드와
릴리즈 파이프라인의 readiness 결과(`backend/run_release_ready.py`)로 확인합니다.

---

## 🚀 30일 이후 자동 동작

`python3 crawlers/run_daily.py` 실행 시 자동으로:

1. 첫 30일 도달 시 학습+릴리즈 실행
2. 이후 `GPU_ADVISOR_AUTO_RETRAIN_EVERY_DAYS` 누적마다 재학습+릴리즈 실행
3. 누적이 부족한 날에는 드라이 체크만 실행
4. 결과물은 `latest_release_report.*`, `latest_auto_training_status.*`에 최신본 저장

---

## ✅ 체크리스트

- [ ] Cron 설정 완료 (`./setup_cron.sh`)
- [ ] 수동 실행 테스트 (`python3 crawlers/run_daily.py`)
- [ ] raw/processed 파일 생성 확인
- [ ] `docs/reports/latest_data_status.*` 갱신 확인
- [ ] `docs/reports/latest_auto_training_status.*` 확인
- [ ] `docs/reports/latest_release_report.*` 확인

---

**작성/갱신:** 2026-03-04
**프로젝트:** GPU Purchase Timing Advisor
