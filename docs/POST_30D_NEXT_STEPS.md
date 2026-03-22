# Post-30-Day Next Steps Guide (EN)

## 1. Purpose
This runbook defines what to do after GPU Advisor secures the **30-day data window** for the RL (AlphaZero/MuZero-style) production path.

**Current Status (2026-03-22):**
- 30-day window: **achieved** (30 files, 2026-02-21 → 2026-03-22)
- All 7 quality gates: **PASS** (directional accuracy 89.4%, avg reward +0.0064)
- Release tag: `release-agent-20260322-105138` pushed to remote
- Model checkpoint: `alphazero_model_agent_latest.pth` (227MB)
- Next auto-retrain: triggered when 7 more days of data accumulate

Primary goals:
- While `< 30 days`: keep daily collection + dry checks
- Once `>= 30 days`: run training → evaluation → release gates → release candidate flow ✅ **Completed**

## 2. What Is Pre-Programmed Now
The codebase now includes these pre-wired steps:

1. Auto-generated next-step checklist in daily status reports  
   - Files: `docs/reports/latest_data_status.json`, `docs/reports/latest_data_status.md`
   - Field: `next_steps`

2. Next-step API endpoint  
   - Endpoint: `GET /api/agent/next-steps`
   - Returns bilingual summary and step-by-step commands/conditions

3. 30-day release runner  
   - Command: `python3 backend/run_release_ready.py`
   - Internal flow: readiness → training → evaluation → gates → report

## 3. Operational Scenarios

### Scenario A: Not yet 30 days
1. Keep daily crawling active
```bash
python3 crawlers/run_daily.py
```
2. Run dry release checks (no training)
```bash
python3 backend/run_release_daily.py
```
3. Preflight release path without training
```bash
python3 backend/run_release_ready.py --allow-short-window --no-train --lookback-days 7
```

### Scenario B: 30-day window reached
1. Run full release pipeline
```bash
python3 backend/run_release_ready.py
```
2. Review latest report
```bash
cat docs/reports/latest_release_report.md
```

### Scenario C: Gate result is `blocked`
1. Increase/re-tune training and re-run
```bash
python3 backend/run_release_ready.py --steps 1000 --lookback-days 30
```
2. Fix failed gates (`accuracy_raw`, `reward_raw`, `uplift_raw_vs_buy`, etc.) and iterate

### Scenario D: Gate result is `pass`
1. Create/push a release candidate tag
```bash
python3 backend/run_release_ready.py --tag --push-tag
```

## 4. Daily Operator Checkpoints
- Data status: `docs/reports/latest_data_status.md`
- Release decision: `docs/reports/latest_release_report.md`
- API workflow status: `GET /api/agent/next-steps`

## 5. Notes
- The RL production path is implemented in `backend/agent/*`.
- The `30-day` threshold is a release gate policy; technically, training can run with lower minimum data windows (2+ aligned days).
