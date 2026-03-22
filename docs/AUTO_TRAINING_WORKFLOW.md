# Automatic Training and Artifact Workflow (EN)

## 1. Objective

> ✅ **2026-03-22 Completed**: 30-day data window achieved, auto train+evaluate+release pipeline ran successfully, release tag `release-agent-20260322-105138` pushed. Now in post-30d periodic retraining mode (every 7 days).

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

## 3. Decision Logic

`decide_auto_training_action()` evaluates conditions in this order:

| Condition | Decision | Reason Code |
|-----------|----------|-------------|
| `auto_training_enabled = false` | `release_check` | `auto_training_disabled` |
| `current_min_days < target_days` | `release_check` | `insufficient_data_window` |
| `dataset_last_date` missing | `release_check` | `dataset_last_date_missing` |
| No training history (`last_trained_data_date` absent) | `train_release` | `first_training_after_target` |
| `newly_accumulated_days >= retrain_every_days` | `train_release` | `retrain_accumulation_met` |
| Otherwise | `release_check` | `retrain_accumulation_not_met` |

## 4. State Persistence and Triggering

- Default state file: `data/processed/auto_training_state.json`
- Key fields:

| Field | Description |
|-------|-------------|
| `last_trained_data_date` | Latest data date used in the last training run |
| `last_training_run_at` | Timestamp of last training run (ISO 8601) |
| `last_pipeline_status` | `train_release` / `release_check` / `error` |
| `last_decision` | Full `AutoTrainingDecision` snapshot |

The next daily run reads this state and decides whether retraining is required.

## 5. Environment Variables

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

## 6. How to Run

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

### Override target data window for one run
```bash
python3 crawlers/run_daily.py --auto-target-days 20
```

### Skip release pipeline (crawling only)
```bash
python3 crawlers/run_daily.py --skip-release
```

## 7. Generated Artifacts

### 7.1 Release outputs
- `docs/reports/latest_release_report.json`
- `docs/reports/latest_release_report.md`

### 7.2 Automation outputs
- `docs/reports/latest_auto_training_status.json`
- `docs/reports/latest_auto_training_status.md`

### 7.3 Date-versioned history
- `docs/reports/YYYY-MM-DD/auto_training_status_*.{json,md}`
- `docs/reports/YYYY-MM-DD/release_report_*.{json,md}`

### 7.4 Checkpoint structure
When training completes, `alphazero_model_agent_latest.pth` contains:

| Key | Description |
|-----|-------------|
| `h_state_dict` | Representation Network |
| `g_state_dict` | Dynamics Network |
| `f_state_dict` | Prediction Network |
| `a_state_dict` | Action Model |
| `optimizer_state_dict` | AdamW optimizer state |

## 8. Operator Checklist

1. **Register LaunchAgent**: `~/Library/LaunchAgents/com.gpu-advisor.daily-crawl.plist`
   - Due to macOS TCC protection, log paths must be under `~/Library/Logs/`
2. **LaunchAgent log**: `~/Library/Logs/gpu-advisor/cron.log`
3. **Detailed Python log**: `data/gpu-advisor/logs/daily_crawl.log` — verify automation action
4. **Automation status**: `docs/reports/latest_auto_training_status.md` — check retrain trigger
5. **Release gates**: `docs/reports/latest_release_report.md` — check gate outcomes
6. If `blocked`, tune hyperparameters/data quality and continue with the next accumulation cycle

### LaunchAgent Diagnostics

```bash
# Check status
launchctl print gui/$(id -u)/com.gpu-advisor.daily-crawl

# Reload
launchctl bootout gui/$(id -u)/com.gpu-advisor.daily-crawl.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gpu-advisor.daily-crawl.plist

# Manual run
python3 crawlers/run_daily.py
```
