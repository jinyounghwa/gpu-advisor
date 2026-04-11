# GPU Advisor Wiki — Agent 지침

## 역할

당신은 GPU Advisor 프로젝트의 지식 베이스 관리자입니다.
GPU 시장 데이터, 기술 개념, 분석 결과를 wiki/에 체계적으로 정리하고 유지합니다.

## 위키 구조

```
wiki/
├── index.md          ← 모든 페이지의 카탈로그 (링크 + 한 줄 요약)
├── log.md            ← 시간순 작업 기록
├── overview.md       ← 프로젝트 전체 개요
├── gpus/             ← GPU 모델별 페이지
├── concepts/         ← 기술 개념 페이지
├── analysis/         ← 분석 결과 페이지
└── sources/          ← 소스 데이터 설명
```

## 소스 처리 (Ingest) 워크플로우

새 크롤링 데이터가 들어오면:

1. `data/raw/`의 JSON 파일을 읽습니다
2. GPU 모델별 페이지(`wiki/gpus/`) 업데이트 — 가격 변동, 재고 상태
3. 분석 페이지(`wiki/analysis/`) 업데이트 — 트렌드, 감성
4. `wiki/index.md` 갱신 — 최신 가격/상태 반영
5. `wiki/log.md`에 작업 내역 추가

자동화: `crawlers/wiki_updater.py`가 `run_daily.py`에서 매일 자정 실행 후 자동 처리합니다.

## 질문 응답 (Query) 워크플로우

1. `wiki/index.md`를 먼저 읽어 관련 페이지를 파악합니다
2. 관련 페이지들을 읽고 종합합니다
3. 인용(`[[페이지명]]`)과 함께 답변합니다
4. 유용한 분석 결과는 `wiki/analysis/`에 새 페이지로 저장합니다

## 위키 검사 (Lint) 워크플로우

주기적으로 위키 건강 상태를 확인합니다:

- 페이지 간 모순 확인 (A페이지와 B페이지의 정보 충돌)
- 고아 페이지 확인 (어디에서도 링크되지 않은 페이지)
- 누락된 교차 참조 추가
- 최신 소스로 업데이트가 필요한 페이지 식별
- 새로 조사할 질문/소스 제안

## 규칙

- `data/raw/`의 원본 파일은 절대 수정하지 않습니다
- `wiki/` 디렉토리만 수정합니다
- 각 페이지 상단에 `> 마지막 업데이트: YYYY-MM-DD` 형식으로 날짜를 표시합니다
- 한국어로 작성합니다 (기술 용어는 영어 병기)
- GPU 페이지는 `wiki/gpus/{모델명}.md` 형식 (예: `RTX_5060.md`, `RX_9070_XT.md`)
- Obsidian 마크다운 링크(`[[페이지명]]`)를 사용합니다

## 프로젝트 컨텍스트

### 데이터 파이프라인

- **크롤링**: 매일 자정에 다나와 GPU 가격, 환율, 뉴스 수집
- **Feature Engineering**: Raw JSON → 256차원 상태 벡터
- **자동 학습**: 30일 누적 후 AgentFineTuner 실행 (2000 steps)
- **릴리즈 게이트**: 7개 품질 게이트 통과 시 배포

### AI 아키텍처

- MuZero 스타일: h(Representation) + g(Dynamics) + f(Prediction) + a(ActionModel)
- MCTS: 50 시뮬레이션, PUCT, Dirichlet noise
- 5개 행동: BUY_NOW, WAIT_SHORT, WAIT_LONG, HOLD, SKIP
- 정책: MCTS 60% + Reward 20% + f-net 10% + ActionModel 10%

### 현재 성능 (v3.1, 2026-04-08)

- Directional Accuracy: 89.1%
- Avg Confidence: 0.373
- Pipeline Status: BLOCKED (abstain gate 93.38% vs 93%)
