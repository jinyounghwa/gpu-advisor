# 30일 데이터 달성 후 다음 진행사항 가이드 (KR)

## 1. 목적
이 문서는 GPU Advisor에서 **30일 데이터 윈도우**를 확보한 뒤, RL(AlphaZero/MuZero 스타일) 운영 경로를 어떤 순서로 실행해야 하는지 정리한 운영 안내서입니다.

핵심 목표:
- `30일 미만` 구간: 일일 수집 + 드라이 체크 유지
- `30일 이상` 구간: 자동 학습 오케스트레이션 → 평가 → 게이트 판정 → 릴리즈 후보 확정

**현재 상태 (2026-03-22):**
- dataset: **30일** 확보 완료 (목표 달성)
- 방향정확도: **0.894** / 평균 보상: +0.0064 (양수)
- 게이트: **7개 중 7개 통과 (전체 PASS)**
- 릴리즈 태그: `release-agent-20260322-105138` 생성 및 원격 저장소 푸시 완료
- 모델 체크포인트: `alphazero_model_agent_latest.pth` (227MB)

## 2. 현재 코드 반영 상태

다음 기능이 코드에 구현되어 있습니다.

1. **자동 학습 오케스트레이션** (`crawlers/auto_training.py`)
   - `AutoTrainingConfig`: 학습 파라미터 + 환경변수 바인딩
   - `decide_auto_training_action()`: 30일 도달 / 재학습 간격 판정
   - 30일 도달 시 자동으로 `backend/run_release_ready.py` 실행

2. **일일 상태 리포트**
   - 파일: `docs/reports/latest_data_status.json`, `docs/reports/latest_data_status.md`
   - 필드: `next_steps`, `current_min_days`, `ready_for_target`

3. **API로 다음 단계 조회**
   - 엔드포인트: `GET /api/agent/next-steps`
   - 응답: 한/영 요약 + 단계별 실행 명령/조건

4. **30일 기준 릴리즈 실행기**
   - 명령: `python3 backend/run_release_ready.py`
   - 내부 단계: readiness → training → evaluation → gates → report

5. **LaunchAgent 자동화** (`~/Library/LaunchAgents/com.gpu-advisor.daily-crawl.plist`)
   - 매일 자정 `run_daily.py` 자동 실행
   - 로그: `~/Library/Logs/gpu-advisor/cron.log`

## 3. 자동 학습 판정 로직

`crawlers/auto_training.py`의 `decide_auto_training_action()`이 매일 다음 판정을 수행합니다:

| 조건 | 결정 |
|------|------|
| `auto_training_enabled = false` | `release_check` (드라이 체크만) |
| `current_min_days < target_days` | `release_check` (데이터 부족) |
| 학습 이력 없음 + 30일 도달 | `train_release` (첫 학습!) |
| 신규 데이터 ≥ 7일 누적 | `train_release` (재학습) |
| 나머지 | `release_check` |

30일 도달 시 **수동 개입 없이** 전체 학습+평가 파이프라인이 자동 실행됩니다.

## 4. 운영 시나리오

### 시나리오 A: 아직 30일 미만 (완료됨 — 2026-03-22 기준 30일 달성)
1. 일일 수집 유지 (LaunchAgent 자동 실행)
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

### 시나리오 B: 30일 도달 (자동 처리됨)
- LaunchAgent가 `run_daily.py` → `auto_training.py` → `train_release` 결정 → `run_release_ready.py` 순서로 자동 실행
- 결과 확인:
```bash
cat docs/reports/latest_release_report.md
cat docs/reports/latest_auto_training_status.md
```
- 수동으로 먼저 실행하려면:
```bash
python3 backend/run_release_ready.py
```

### 시나리오 C: 게이트 결과가 `blocked`
1. 실패 게이트 확인
```bash
cat docs/reports/latest_release_report.md
# uplift_raw_vs_buy, accuracy_raw 등 실패 항목 확인
```
2. 학습 스텝/구성 조정 후 재실행
```bash
python3 backend/run_release_ready.py --steps 1000 --lookback-days 30
```
3. 다음 자동 재학습: 7일 추가 데이터 누적 후 LaunchAgent가 자동 트리거

### 시나리오 D: 게이트 결과가 `pass`
1. 릴리즈 후보 태그 생성/푸시
```bash
python3 backend/run_release_ready.py --tag --push-tag
```

### 시나리오 E: 자동 학습 일시 비활성화
```bash
# 특정 실행에서만 비활성화
python3 crawlers/run_daily.py --disable-auto-train

# 타깃 윈도우 임시 변경
python3 crawlers/run_daily.py --auto-target-days 20

# 재학습 간격 단축
python3 crawlers/run_daily.py --auto-retrain-days 3
```

## 5. 운영자가 매일 볼 위치

| 항목 | 위치 |
|------|------|
| LaunchAgent 실행 여부 | `~/Library/Logs/gpu-advisor/cron.log` |
| 상세 파이썬 로그 | `data/gpu-advisor/logs/daily_crawl.log` |
| 데이터 상태 | `docs/reports/latest_data_status.md` |
| 자동 학습 상태 | `docs/reports/latest_auto_training_status.md` |
| 릴리즈 판정 | `docs/reports/latest_release_report.md` |
| API 기반 진행 체크 | `GET /api/agent/next-steps` |

## 6. 30일 도달 후 품질 판정 기준

| 게이트 | 의미 | 현재 상태 |
|--------|------|-----------|
| accuracy_raw | 방향정확도 > 0.55 | ✅ 0.894 |
| reward_raw | 평균 보상 > 0 | ✅ +0.0064 |
| abstain | 관망비율 < 0.85 | ✅ 0.788 |
| safe_override | 안전오버라이드 정상 | ✅ 0.384 |
| action_entropy_raw | 엔트로피 정상 | ✅ 1.459 |
| no_mode_collapse_raw | 모드 붕괴 없음 | ✅ |
| uplift_raw_vs_buy | always-buy 대비 초과수익 | ✅ +0.0040 |

## 7. 자동 재학습 주기 (30일 이후)

30일 도달 후에도 `auto_training.py`가 7일마다 자동으로 재학습을 트리거합니다:

```
마지막 학습 → 7일 데이터 누적 → retrain_accumulation_met → train_release
```

환경변수로 조정 가능:
```bash
export GPU_ADVISOR_AUTO_RETRAIN_EVERY_DAYS=7  # 기본값
```

## 8. 참고

- 이 프로젝트의 RL 운영 경로는 `backend/agent/*`를 기준으로 동작합니다.
- `30일`은 운영 게이트 기준이며, 학습 자체는 데이터 최소 요건(2일+)에서도 기술적으로 실행될 수 있습니다.
- 자동 학습 오케스트레이션 상세: `docs/AUTO_TRAINING_WORKFLOW_KR.md`
- 하이퍼파라미터 근거: `docs/HYPERPARAMETER_GUIDE_KR.md`
