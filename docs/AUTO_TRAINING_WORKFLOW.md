# Automatic Training and Artifact Workflow (EN)

## 1. Objective
This document explains how the production path now automates:

1. Automatic training once the 30-day data window is reached
2. Automatic artifact generation after training (checkpoint/release report/automation report)
3. Ongoing retraining after additional data accumulation

Applied path:
- `crawlers/run_daily.py` (daily batch orchestrator)
- `crawlers/auto_training.py` (decision + execution + state persistence)

## 2. Automation Policy

### 2.1 Before day-30 readiness
- Behavior: run release dry-check without training
- Script: `backend/run_release_daily.py`
- Purpose: monitor gate/metric trends

### 2.2 At first day-30 readiness
- Behavior: run full train+evaluate release flow automatically
- Script: `backend/run_release_ready.py`
- Artifacts:
  - latest checkpoint (when training runs): `alphazero_model_agent_latest.pth`
  - release report: `docs/reports/latest_release_report.{json,md}`
  - automation report: `docs/reports/latest_auto_training_status.{json,md}`

### 2.3 Post-30 periodic retraining
- Trigger: at least `N` newly accumulated days since last training
- Default: `N=7` (`GPU_ADVISOR_AUTO_RETRAIN_EVERY_DAYS`)
- If met: re-run `backend/run_release_ready.py`
- If not met: run dry-check only (`backend/run_release_daily.py`)

## 3. State Persistence and Triggering
- Default state file: `data/processed/auto_training_state.json`
- Key fields:
  - `last_trained_data_date`
  - `last_training_run_at`
  - `last_pipeline_status`
  - `last_decision`

The next daily run reads this state and decides whether retraining is required.

## 4. Environment Variables
Use these variables to control automation behavior:

```bash
# Enable/disable auto-training (default: true)
export GPU_ADVISOR_AUTO_TRAIN_ENABLED=true

# Target data window in days (default: 30)
export GPU_ADVISOR_AUTO_TRAIN_TARGET_DAYS=30

# Retrain trigger interval in newly accumulated days (default: 7)
export GPU_ADVISOR_AUTO_RETRAIN_EVERY_DAYS=7

# Training params for auto mode
export GPU_ADVISOR_AUTO_TRAIN_STEPS=500
export GPU_ADVISOR_AUTO_TRAIN_BATCH_SIZE=32
export GPU_ADVISOR_AUTO_TRAIN_LR=0.0001
export GPU_ADVISOR_AUTO_TRAIN_SEED=42
export GPU_ADVISOR_AUTO_TRAIN_LOOKBACK_DAYS=30
export GPU_ADVISOR_AUTO_TRAIN_TIMEOUT_SEC=5400

# Optional custom state path
export GPU_ADVISOR_AUTO_TRAIN_STATE_PATH=data/processed/auto_training_state.json
```

## 5. How to Run

### Default (recommended)
```bash
python3 crawlers/run_daily.py
```
- After crawling/feature generation, it automatically chooses train or dry-check.

### Disable auto-training for one run
```bash
python3 crawlers/run_daily.py --disable-auto-train
```

### Override retrain interval for one run
```bash
python3 crawlers/run_daily.py --auto-retrain-days 5
```

## 6. Generated Artifacts

### 6.1 Release outputs
- `docs/reports/latest_release_report.json`
- `docs/reports/latest_release_report.md`

### 6.2 Automation outputs
- `docs/reports/latest_auto_training_status.json`
- `docs/reports/latest_auto_training_status.md`

### 6.3 Date-versioned history
- `docs/reports/YYYY-MM-DD/auto_training_status_*.{json,md}`
- `docs/reports/YYYY-MM-DD/release_report_*.{json,md}`

## 7. Operator Checklist
1. Register daily cron via `setup_cron.sh`
2. Verify automation action in `logs/daily_crawl.log`
3. Check retrain trigger details in `latest_auto_training_status.md`
4. Review gate outcomes in `latest_release_report.md`
5. If `blocked`, tune hyperparameters/data quality and continue with the next accumulation cycle
