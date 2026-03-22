# GPU Advisor 자동 학습/결과물 생성 상태 보고서

- 생성시각: 2026-03-22T00:13:07.989448
- 자동화 실행모드: training_release
- 실행 스크립트: backend/run_release_ready.py
- 파이프라인 상태: pass

## 1) 자동화 결정 (Decision)
- action: train_release
- reason: first_training_after_target
- current_min_days: 30
- target_days: 30
- ready_for_target: True
- newly_accumulated_days: 0
- retrain_every_days: 7

## 2) 데이터 준비도
- current_min_days: 30
- remaining_days: 0
- ready_for_target: True

## 3) 결과물 (Artifacts)
- checkpoint: /Users/younghwa.jin/Documents/gpu-advisor/alphazero_model_agent_latest.pth
- release report (json): /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/2026-03-22/release_report_20260322_001307.json
- release report (md): /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/2026-03-22/release_report_20260322_001307.md
- latest release json: /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/latest_release_report.json
- latest release md: /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/latest_release_report.md

## 4) State
- state_path: /Users/younghwa.jin/Documents/gpu-advisor/data/processed/auto_training_state.json
- last_trained_data_date(after): 2026-03-22

## 5) Bilingual Summary
- KR: 30일 윈도우 이후 자동 학습/평가 루틴이 실행되었습니다.
- EN: Automatic post-30-day training/evaluation routine was executed.
