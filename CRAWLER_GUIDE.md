# 크롤러 시스템 사용 가이드
## GPU 구매 타이밍 AI - 자동 데이터 수집

---

## 개요

이 크롤러 시스템은 **매일 자동으로** GPU 가격, 환율, 뉴스를 수집하고 **256차원 Feature**를 생성합니다.
실행 직후 `docs/reports/`에 실제 수집 파일 기준 상태 보고서도 자동 생성됩니다.

### 수집 데이터
1. **다나와 GPU 가격** - 대상 GPU 모델 최저가/판매자수/재고 상태
2. **환율 정보** - USD/KRW, JPY/KRW, EUR/KRW
3. **GPU 뉴스** - 기사 수/감정 통계 포함
4. **256차원 Feature** - 에이전트 입력 데이터(`training_data_YYYY-MM-DD.json`)

---

## 빠른 시작

### 1단계: 수동 즉시 실행 (테스트)

```bash
cd /Users/younghwa.jin/Documents/gpu-advisor
python3 crawlers/run_daily.py
```

### 2단계: macOS LaunchAgent 자동화 (한 번만 설정)

> **주의**: macOS에서는 `cron` 대신 **LaunchAgent**를 사용해야 합니다.
> `~/Documents/` 경로는 macOS TCC 보호로 인해 launchd의 stdout/stderr 대상으로 사용할 수 없습니다.
> 로그는 반드시 `~/Library/Logs/` 하위에 기록해야 합니다.

LaunchAgent plist 경로:
```
~/Library/LaunchAgents/com.gpu-advisor.daily-crawl.plist
```

로드/재로드 명령:
```bash
# 로드
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gpu-advisor.daily-crawl.plist

# 언로드
launchctl bootout gui/$(id -u)/com.gpu-advisor.daily-crawl

# 상태 확인
launchctl print gui/$(id -u)/com.gpu-advisor.daily-crawl
```

pmset 웨이크 스케줄 (Mac이 자동으로 깨어나도록):
```bash
sudo pmset repeat wakepoweron MTWRFSU 23:58:00
```

---

## 파일 구조

```text
gpu-advisor/
├── crawlers/
│   ├── danawa_crawler.py
│   ├── exchange_rate_crawler.py
│   ├── news_crawler.py
│   ├── feature_engineer.py
│   ├── auto_training.py          # 자동 학습 오케스트레이션
│   ├── status_report.py
│   └── run_daily.py              # 일일 배치 오케스트레이터
├── data/
│   ├── raw/
│   │   ├── danawa/
│   │   ├── exchange/
│   │   └── news/
│   └── processed/
│       ├── dataset/              # training_data_YYYY-MM-DD.json
│       └── auto_training_state.json  # 자동 학습 상태
├── docs/reports/
│   ├── YYYY-MM-DD/
│   │   ├── data_status_*.json
│   │   ├── data_status_*.md
│   │   ├── release_report_*.json
│   │   ├── release_report_*.md
│   │   └── auto_training_status_*.json
│   ├── latest_data_status.json
│   ├── latest_data_status.md
│   ├── latest_release_report.json
│   ├── latest_release_report.md
│   ├── latest_auto_training_status.json
│   └── latest_auto_training_status.md
└── ~/Library/LaunchAgents/
    └── com.gpu-advisor.daily-crawl.plist
```

### 로그 경로

| 로그 | 경로 | 설명 |
|------|------|------|
| LaunchAgent stdout/stderr | `~/Library/Logs/gpu-advisor/cron.log` | launchd 실행 결과 |
| 상세 파이썬 로그 | `data/gpu-advisor/logs/daily_crawl.log` | 크롤러 상세 로그 |

> `logs/cron.log` (프로젝트 내부) 경로는 macOS TCC 보호로 launchd 사용 불가.
> 반드시 `~/Library/Logs/` 경로를 사용해야 합니다.

---

## LaunchAgent 스케줄

### 기본 설정
- 실행 시간: 매일 00:00 (`StartCalendarInterval`)
- 실행 순서:

1. 다나와 GPU 가격 크롤링
2. 환율 정보 수집
3. GPU 뉴스 크롤링
4. 256차원 Feature 생성
5. 상태 보고서 생성
6. 자동 학습 오케스트레이션 (`auto_training.py`)
   - 30일 미달: 릴리즈 드라이 체크 (`backend/run_release_daily.py`)
   - 30일 도달 또는 재학습 간격 충족: 학습+평가 (`backend/run_release_ready.py`)
7. 자동화/릴리즈 결과 보고서 생성

### LaunchAgent plist 핵심 설정

```xml
<key>StartCalendarInterval</key>
<dict>
    <key>Hour</key><integer>0</integer>
    <key>Minute</key><integer>0</integer>
</dict>
<key>StandardOutPath</key>
<string>/Users/younghwa.jin/Library/Logs/gpu-advisor/cron.log</string>
<key>StandardErrorPath</key>
<string>/Users/younghwa.jin/Library/Logs/gpu-advisor/cron.log</string>
```

### LaunchAgent 진단

```bash
# 상태 확인 (exit code 78 = EX_CONFIG = TCC 로그 경로 오류)
launchctl print gui/$(id -u)/com.gpu-advisor.daily-crawl

# 수동 트리거
launchctl kickstart -k gui/$(id -u)/com.gpu-advisor.daily-crawl

# 재로드
launchctl bootout gui/$(id -u)/com.gpu-advisor.daily-crawl
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gpu-advisor.daily-crawl.plist
```

---

## 데이터 형식

### 1) 다나와 (`data/raw/danawa/YYYY-MM-DD.json`)

```json
{
  "date": "2026-03-15",
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
    "crawled_at": "2026-03-15T00:00:00",
    "target_models": 24
  }
}
```

### 2) 환율 (`data/raw/exchange/YYYY-MM-DD.json`)

```json
{
  "date": "2026-03-15",
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
  "date": "2026-03-15",
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
    "date": "2026-03-15",
    "gpu_model": "RTX 5060",
    "state_vector": [0.0606, 0.062, ...]
  }
]
```

`state_vector` 길이는 **256**입니다. 구성:

| 그룹 | 차원 |
|------|------|
| 가격 피처 | 60D |
| 환율 피처 | 20D |
| 뉴스 피처 | 30D |
| 시장 피처 | 20D |
| 시간 피처 | 20D |
| 기술지표 | 106D |
| **합계** | **256D** |

---

## CLI 옵션

`python3 crawlers/run_daily.py` 지원 옵션:

| 옵션 | 설명 |
|------|------|
| *(없음)* | 크롤링 + 자동 학습 판정 전체 실행 |
| `--skip-release` | 크롤링만 실행, 릴리즈 파이프라인 생략 |
| `--disable-auto-train` | 자동 학습 비활성화 (드라이 체크만) |
| `--auto-retrain-days N` | 재학습 간격 N일로 오버라이드 |
| `--auto-target-days N` | 타깃 데이터 윈도우 N일로 오버라이드 |

---

## 테스트 및 점검

### 전체 파이프라인 실행

```bash
python3 crawlers/run_daily.py
```

### 크롤링만 (릴리즈 생략)

```bash
python3 crawlers/run_daily.py --skip-release
```

### 자동 학습 비활성화 (드라이 체크만)

```bash
python3 crawlers/run_daily.py --disable-auto-train
```

### 재학습 간격 임시 오버라이드

```bash
python3 crawlers/run_daily.py --auto-retrain-days 5
```

### 로그 확인

```bash
# LaunchAgent 실행 로그
tail -f ~/Library/Logs/gpu-advisor/cron.log

# 상세 파이썬 로그
tail -f data/gpu-advisor/logs/daily_crawl.log
tail -n 100 data/gpu-advisor/logs/daily_crawl.log
```

### 보고서 확인

```bash
cat docs/reports/latest_data_status.md
cat docs/reports/latest_release_report.md
cat docs/reports/latest_auto_training_status.md
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

## 설정 변경

### 수집 대상 GPU 모델 변경
- 파일: `crawlers/danawa_crawler.py`
- `self.target_gpus` 목록 수정

### Feature 차원 변경
- 파일: `crawlers/feature_engineer.py`
- 각 블록(가격/환율/뉴스/시장/시간/기술지표) 차원과 최종 합계가 일치하도록 함께 수정

### 자동 학습 정책 변경 (환경변수)

```bash
export GPU_ADVISOR_AUTO_TRAIN_ENABLED=true        # 자동 학습 on/off
export GPU_ADVISOR_AUTO_TRAIN_TARGET_DAYS=30      # 학습 기준 일수
export GPU_ADVISOR_AUTO_RETRAIN_EVERY_DAYS=7      # 재학습 간격 (신규 누적 일수)
export GPU_ADVISOR_AUTO_TRAIN_STEPS=500           # 학습 스텝 수
export GPU_ADVISOR_AUTO_TRAIN_BATCH_SIZE=32       # 배치 크기
export GPU_ADVISOR_AUTO_TRAIN_LR=0.0001           # 학습률
export GPU_ADVISOR_AUTO_TRAIN_SEED=42             # 재현성 시드
export GPU_ADVISOR_AUTO_TRAIN_LOOKBACK_DAYS=30    # 학습 데이터 범위
export GPU_ADVISOR_AUTO_TRAIN_TIMEOUT_SEC=5400    # 학습 타임아웃 (초)
export GPU_ADVISOR_AUTO_TRAIN_STATE_PATH=data/processed/auto_training_state.json
```

---

## 누적 데이터 확인

```bash
# 소스별 날짜 수
ls -1 data/raw/danawa/*.json | wc -l
ls -1 data/raw/exchange/*.json | wc -l
ls -1 data/raw/news/*.json | wc -l
ls -1 data/processed/dataset/training_data_*.json | wc -l

# 최신 상태 보고서
cat docs/reports/latest_data_status.json | python3 -m json.tool
```

30일 준비 여부는 상태 보고서의 `current_min_days` / `ready_for_target` 필드와
릴리즈 파이프라인(`python3 backend/run_release_ready.py`)으로 확인합니다.

---

## 30일 이후 자동 동작

`python3 crawlers/run_daily.py` 실행 시 `auto_training.py`가 자동 판정:

| 조건 | 동작 |
|------|------|
| 30일 미달 | 드라이 체크만 (`backend/run_release_daily.py`) |
| 30일 도달 (첫 학습) | 학습+평가+릴리즈 (`backend/run_release_ready.py`) |
| 30일 이후 + 신규 7일 누적 | 재학습+평가+릴리즈 |
| 30일 이후 + 누적 미달 | 드라이 체크만 |

> **현재 상태 (2026-04-08)**: 47일 데이터 누적. 마지막 학습 2026-04-03 (5/7일 누적). 다음 자동 재학습 ~2026-04-10. 파이프라인 BLOCKED (abstain 게이트 93.38% > 93%).

결과물:
- `docs/reports/latest_release_report.{json,md}`
- `docs/reports/latest_auto_training_status.{json,md}`
- `data/processed/auto_training_state.json` (다음 판정 기준 저장)

---

## 운영 체크리스트

- [ ] LaunchAgent 등록 (`launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gpu-advisor.daily-crawl.plist`)
- [ ] pmset 웨이크 스케줄 설정 (`sudo pmset repeat wakepoweron MTWRFSU 23:58:00`)
- [ ] 수동 실행 테스트 (`python3 crawlers/run_daily.py`)
- [ ] raw/processed 파일 생성 확인
- [ ] LaunchAgent 로그 경로 확인 (`~/Library/Logs/gpu-advisor/cron.log`)
- [ ] `docs/reports/latest_data_status.*` 갱신 확인
- [ ] `docs/reports/latest_auto_training_status.*` 확인
- [ ] `docs/reports/latest_release_report.*` 확인

---

**작성/갱신:** 2026-04-08
**프로젝트:** GPU Purchase Timing Advisor
