# GPU Advisor 자동 학습/결과물 생성 상태 보고서

- 생성시각: 2026-04-06T00:03:29.206331
- 자동화 실행모드: release_check
- 실행 스크립트: backend/run_release_daily.py
- 파이프라인 상태: blocked

## 1) 자동화 결정 (Decision)
- action: release_check
- reason: waiting_for_more_data_for_retrain
- current_min_days: 45
- target_days: 30
- ready_for_target: True
- newly_accumulated_days: 3
- retrain_every_days: 7

## 2) 데이터 준비도
- current_min_days: 45
- remaining_days: 0
- ready_for_target: True

## 3) 결과물 (Artifacts)
- checkpoint: N/A
- release report (json): /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/2026-04-06/release_report_20260406_000329.json
- release report (md): /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/2026-04-06/release_report_20260406_000329.md
- latest release json: /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/latest_release_report.json
- latest release md: /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/latest_release_report.md

## 4) State
- state_path: /Users/younghwa.jin/Documents/gpu-advisor/data/processed/auto_training_state.json
- last_trained_data_date(after): 2026-04-03

## 5) Bilingual Summary
- KR: 자동 학습 조건 미충족이므로 평가/리포트 루틴만 실행되었습니다.
- EN: Training condition was not met, so evaluation/report routine ran only.
