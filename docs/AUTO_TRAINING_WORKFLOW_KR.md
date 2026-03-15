# 자동 학습 및 결과물 생성 워크플로우 (KR)

## 1. 목표
이 문서는 다음 요구사항을 운영 경로에서 자동화하는 방법을 설명합니다.

1. 30일 데이터 윈도우 달성 시 자동 학습 실행
2. 학습 후 결과물(체크포인트/릴리즈 리포트/자동화 리포트) 자동 생성
3. 이후에도 일정 데이터 누적 시 자동 재학습 및 결과물 생성

적용 경로:
- `crawlers/run_daily.py` (일일 배치 오케스트레이터)
- `crawlers/auto_training.py` (자동 학습/재학습 판단 + 실행 + 상태 저장)

## 2. 자동화 정책

### 2.1 30일 이전
- 동작: 학습 없이 릴리즈 드라이 체크 수행
- 실행 스크립트: `backend/run_release_daily.py`
- 목적: 게이트/지표 추세 모니터링

### 2.2 30일 도달 직후
- 동작: 전체 학습+평가 파이프라인 자동 실행
- 실행 스크립트: `backend/run_release_ready.py`
- 결과물:
  - 최신 체크포인트(학습 실행 시): `alphazero_model_agent_latest.pth`
  - 릴리즈 리포트: `docs/reports/latest_release_report.{json,md}`
  - 자동화 리포트: `docs/reports/latest_auto_training_status.{json,md}`

### 2.3 30일 이후 주기적 재학습
- 기준: 마지막 학습 시점 대비 신규 데이터가 `N일` 이상 누적
- 기본값: `N=7`일 (`GPU_ADVISOR_AUTO_RETRAIN_EVERY_DAYS`)
- 조건 충족 시: `backend/run_release_ready.py` 재실행
- 조건 미충족 시: 드라이 체크(`backend/run_release_daily.py`)만 실행

## 3. 의사결정 로직

`decide_auto_training_action()` 함수가 아래 순서로 판정합니다:

| 조건 | 결정 | 이유 코드 |
|------|------|-----------|
| `auto_training_enabled = false` | `release_check` | `auto_training_disabled` |
| `current_min_days < target_days` | `release_check` | `insufficient_data_window` |
| `dataset_last_date` 없음 | `release_check` | `dataset_last_date_missing` |
| 학습 이력 없음 (`last_trained_data_date` 없음) | `train_release` | `first_training_after_target` |
| `newly_accumulated_days >= retrain_every_days` | `train_release` | `retrain_accumulation_met` |
| 나머지 | `release_check` | `retrain_accumulation_not_met` |

## 4. 상태 저장 및 재실행 판단

- 상태 파일 기본 경로: `data/processed/auto_training_state.json`
- 주요 필드:

| 필드 | 설명 |
|------|------|
| `last_trained_data_date` | 마지막 학습에 사용된 데이터 최신일 |
| `last_training_run_at` | 마지막 학습 실행 시각 (ISO 8601) |
| `last_pipeline_status` | `train_release` / `release_check` / `error` |
| `last_decision` | `AutoTrainingDecision` 전체 스냅샷 |

다음 실행에서 이 상태를 읽어 재학습 필요 여부를 판정합니다.

## 5. 환경변수 설정

아래 변수로 자동화 동작을 제어할 수 있습니다.

```bash
# 자동 학습 기능 on/off (기본: true)
export GPU_ADVISOR_AUTO_TRAIN_ENABLED=true

# 타깃 데이터 윈도우 일수 (기본: 30)
export GPU_ADVISOR_AUTO_TRAIN_TARGET_DAYS=30

# 재학습 트리거 누적 일수 (기본: 7)
export GPU_ADVISOR_AUTO_RETRAIN_EVERY_DAYS=7

# 자동 학습 실행 파라미터
export GPU_ADVISOR_AUTO_TRAIN_STEPS=500
export GPU_ADVISOR_AUTO_TRAIN_BATCH_SIZE=32
export GPU_ADVISOR_AUTO_TRAIN_LR=0.0001
export GPU_ADVISOR_AUTO_TRAIN_SEED=42
export GPU_ADVISOR_AUTO_TRAIN_LOOKBACK_DAYS=30
export GPU_ADVISOR_AUTO_TRAIN_TIMEOUT_SEC=5400

# 상태 파일 경로(선택)
export GPU_ADVISOR_AUTO_TRAIN_STATE_PATH=data/processed/auto_training_state.json
```

## 6. 실행 방법

### 기본 실행 (권장)
```bash
python3 crawlers/run_daily.py
```
- 크롤링/피처 생성 후 자동으로 학습 or 드라이체크를 선택 실행

### 자동 학습 강제 비활성화
```bash
python3 crawlers/run_daily.py --disable-auto-train
```

### 단일 실행에서 재학습 간격 오버라이드
```bash
python3 crawlers/run_daily.py --auto-retrain-days 5
```

### 타깃 데이터 윈도우 오버라이드
```bash
python3 crawlers/run_daily.py --auto-target-days 20
```

### 릴리즈 파이프라인 생략 (크롤링만 실행)
```bash
python3 crawlers/run_daily.py --skip-release
```

## 7. 생성되는 결과물

### 7.1 릴리즈 결과
- `docs/reports/latest_release_report.json`
- `docs/reports/latest_release_report.md`

### 7.2 자동화 결과
- `docs/reports/latest_auto_training_status.json`
- `docs/reports/latest_auto_training_status.md`

### 7.3 날짜별 보관
- `docs/reports/YYYY-MM-DD/auto_training_status_*.{json,md}`
- `docs/reports/YYYY-MM-DD/release_report_*.{json,md}`

### 7.4 체크포인트 구조
학습 완료 시 체크포인트(`alphazero_model_agent_latest.pth`)에 포함되는 항목:

| 키 | 설명 |
|----|------|
| `h_state_dict` | 표현 네트워크 (RepresentationNetwork) |
| `g_state_dict` | 역학 네트워크 (DynamicsNetwork) |
| `f_state_dict` | 예측 네트워크 (PredictionNetwork) |
| `a_state_dict` | 행동 모델 (ActionModel) |
| `optimizer_state_dict` | AdamW 옵티마이저 상태 |

## 8. 운영 체크리스트

1. **LaunchAgent 등록**: `~/Library/LaunchAgents/com.gpu-advisor.daily-crawl.plist`
   - macOS TCC 보호로 인해 로그는 반드시 `~/Library/Logs/` 경로 사용
2. **LaunchAgent 로그 확인**: `~/Library/Logs/gpu-advisor/cron.log`
3. **상세 파이썬 로그**: `data/gpu-advisor/logs/daily_crawl.log`에서 자동화 action 확인
4. **자동화 상태 확인**: `docs/reports/latest_auto_training_status.md`에서 재학습 트리거 여부 확인
5. **릴리즈 게이트 확인**: `docs/reports/latest_release_report.md`의 게이트 통과 여부 확인
6. `blocked`면 하이퍼파라미터/데이터 품질 조정 후 다음 누적 주기에서 재학습

### LaunchAgent 진단 명령

```bash
# 상태 확인
launchctl print gui/$(id -u)/com.gpu-advisor.daily-crawl

# 재로드
launchctl bootout gui/$(id -u)/com.gpu-advisor.daily-crawl.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gpu-advisor.daily-crawl.plist

# 수동 실행
python3 crawlers/run_daily.py
```
