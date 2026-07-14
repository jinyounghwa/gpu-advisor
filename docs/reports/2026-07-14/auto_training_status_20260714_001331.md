# GPU Advisor 자동 학습/결과물 생성 상태 보고서

- 생성시각: 2026-07-14T00:13:31.159777
- 자동화 실행모드: training_release
- 실행 스크립트: backend/run_release_ready.py
- 파이프라인 상태: blocked

## 1) 자동화 결정 (Decision)
- action: train_release
- reason: retrain_interval_reached
- current_min_days: 144
- target_days: 30
- ready_for_target: True
- newly_accumulated_days: 7
- retrain_every_days: 7

## 2) 데이터 준비도
- current_min_days: 144
- remaining_days: 0
- ready_for_target: True

## 3) 결과물 (Artifacts)
- checkpoint: /Users/younghwa.jin/Documents/gpu-advisor/alphazero_model_agent_latest.pth
- release report (json): /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/2026-07-14/release_report_20260714_001330.json
- release report (md): /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/2026-07-14/release_report_20260714_001330.md
- latest release json: /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/latest_release_report.json
- latest release md: /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/latest_release_report.md

## 4) State
- state_path: /Users/younghwa.jin/Documents/gpu-advisor/data/processed/auto_training_state.json
- last_trained_data_date(after): 2026-07-14

## 5) Bilingual Summary
- KR: 30일 윈도우 이후 자동 학습/평가 루틴이 실행되었습니다.
- EN: Automatic post-30-day training/evaluation routine was executed.
