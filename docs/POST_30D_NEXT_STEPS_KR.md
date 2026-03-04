# 30일 데이터 달성 후 다음 진행사항 가이드 (KR)

## 1. 목적
이 문서는 GPU Advisor에서 **30일 데이터 윈도우**를 확보한 뒤, RL(AlphaZero/MuZero 스타일) 운영 경로를 어떤 순서로 실행해야 하는지 정리한 운영 안내서입니다.

핵심 목표:
- `30일 미만` 구간: 일일 수집 + 드라이 체크 유지
- `30일 이상` 구간: 학습 → 평가 → 게이트 판정 → 릴리즈 후보 확정

## 2. 현재 코드 반영 상태
다음 기능이 코드에 미리 반영되어 있습니다.

1. 일일 상태 리포트에 다음 단계 자동 생성  
   - 파일: `docs/reports/latest_data_status.json`, `docs/reports/latest_data_status.md`
   - 필드: `next_steps`

2. API로 다음 단계 조회  
   - 엔드포인트: `GET /api/agent/next-steps`
   - 응답: 한/영 요약 + 단계별 실행 명령/조건

3. 30일 기준 릴리즈 실행기  
   - 명령: `python3 backend/run_release_ready.py`
   - 내부 단계: readiness → training → evaluation → gates → report

## 3. 운영 시나리오

### 시나리오 A: 아직 30일 미만
1. 일일 수집 유지
```bash
python3 crawlers/run_daily.py
```
2. 드라이 체크로 지표 추세 점검
```bash
python3 backend/run_release_daily.py
```
3. 사전 경로 점검(학습 없이)
```bash
python3 backend/run_release_ready.py --allow-short-window --no-train --lookback-days 7
```

### 시나리오 B: 30일 도달
1. 전체 릴리즈 파이프라인 실행
```bash
python3 backend/run_release_ready.py
```
2. 최신 결과 확인
```bash
cat docs/reports/latest_release_report.md
```

### 시나리오 C: 게이트 결과가 `blocked`
1. 학습 스텝/구성 조정 후 재실행
```bash
python3 backend/run_release_ready.py --steps 1000 --lookback-days 30
```
2. 실패 게이트(`accuracy_raw`, `reward_raw`, `uplift_raw_vs_buy` 등) 원인 수정 후 반복

### 시나리오 D: 게이트 결과가 `pass`
1. 릴리즈 후보 태그 생성/푸시
```bash
python3 backend/run_release_ready.py --tag --push-tag
```

## 4. 운영자가 매일 볼 위치
- 데이터 상태: `docs/reports/latest_data_status.md`
- 릴리즈 판정: `docs/reports/latest_release_report.md`
- API 기반 진행 체크: `GET /api/agent/next-steps`

## 5. 참고
- 이 프로젝트의 RL 운영 경로는 `backend/agent/*`를 기준으로 동작합니다.
- `30일`은 운영 게이트 기준이며, 학습 자체는 데이터 최소 요건(2일+)에서도 기술적으로 실행될 수 있습니다.
