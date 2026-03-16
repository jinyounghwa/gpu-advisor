# GPU Advisor 자동 학습/결과물 생성 상태 보고서

- 생성시각: 2026-03-16T00:03:37.086585
- 자동화 실행모드: release_check
- 실행 스크립트: backend/run_release_daily.py
- 파이프라인 상태: blocked

## 1) 자동화 결정 (Decision)
- action: release_check
- reason: insufficient_data_window
- current_min_days: 24
- target_days: 30
- ready_for_target: False
- newly_accumulated_days: 0
- retrain_every_days: 7

## 2) 데이터 준비도
- current_min_days: 24
- remaining_days: 6
- ready_for_target: False

## 3) 결과물 (Artifacts)
- checkpoint: N/A
- release report (json): /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/2026-03-16/release_report_20260316_000336.json
- release report (md): /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/2026-03-16/release_report_20260316_000336.md
- latest release json: /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/latest_release_report.json
- latest release md: /Users/younghwa.jin/Documents/gpu-advisor/docs/reports/latest_release_report.md

## 4) State
- state_path: /Users/younghwa.jin/Documents/gpu-advisor/data/processed/auto_training_state.json
- last_trained_data_date(after): None

## 5) Bilingual Summary
- KR: 자동 학습 조건 미충족이므로 평가/리포트 루틴만 실행되었습니다.
- EN: Training condition was not met, so evaluation/report routine ran only.
