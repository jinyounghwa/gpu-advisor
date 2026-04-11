# LLM Wiki 패턴 — GPU Advisor 프로젝트 적용 가이드

> **목적**: `wiki.md`에 설명된 "LLM이 영구적으로 관리하는 개인 지식 베이스" 패턴을 GPU Advisor 프로젝트에 구체적으로 적용하는 방법을 안내합니다.

---

## 1. LLM Wiki 패턴이란?

### 1.1 기존 방식 (RAG)의 한계

대부분의 LLM 문서 활용은 **RAG(검색 증강 생성)** 방식입니다:
- 문서를 업로드 → 질문 시 관련 청크를 검색 → 답변 생성
- **문제**: 매 질문마다 처음부터 지식을 재조립. 축적이 없음. 5개 문서를 종합해야 하는 질문은 매번 새로 찾아서 연결해야 함

### 1.2 LLM Wiki 방식의 차이

LLM이 **점진적으로 구축하고 유지하는 영구 위키**를 만듭니다:
- 새 소스가 들어오면 → LLM이 읽고, 핵심 정보를 추출하고, 기존 위키에 **통합**
- 교차 참조는 이미 만들어져 있고, 모순은 이미 표시되어 있음
- **위키는 복리적으로 성장하는 자산**

### 1.3 핵심 아이디어

> **인간은 소싱, 방향 설정, 질문을 담당하고, LLM은 요약·교차참조·정리·관리 등 모든 "단순 반복 작업"을 처리한다.**

---

## 2. 아키텍처 3계층

| 계층 | 역할 | GPU Advisor에서의 위치 |
|------|------|----------------------|
| **Raw Sources (원본 소스)** | 불변의 원본 문서. LLM은 읽기만 하고 수정하지 않음 | `data/raw/` (danawa, exchange, news 등) |
| **The Wiki (위키)** | LLM이 생성/유지하는 마크다운 파일들. 요약, 엔티티 페이지, 개념 페이지, 비교, 종합 | **새로 만들어야 함** → `wiki/` 디렉토리 |
| **The Schema (스키마)** | LLM에게 위키 구조, 규칙, 워크플로우를 알려주는 설정 파일 | **새로 만들어야 함** → `AGENTS.md` 또는 `CLAUDE.md` |

### 현재 프로젝트 상태

```
✅ Raw Sources: 이미 존재 (data/raw/danawa/, data/raw/exchange/, data/raw/news/)
✅ 자동 수집: 이미 존재 (crawlers/run_daily.py, 매일 자정 실행)
✅ 보고서: 부분적 존재 (docs/reports/ 일일 리포트)
✅ Wiki 계층: 완성 (wiki/ 디렉토리, 34개 페이지 자동 인덱싱)
✅ Schema: CLAUDE.md + AGENTS.md 모두 생성 완료

**2026-04-11 업데이트**: Wiki 통합 시스템 완성
- `wiki/`: 34개 페이지 (concepts, sources, analysis, gpus)
- `CLAUDE.md`: Claude Code 개발 규칙
- `AGENTS.md`: LLM 에이전트 지침
- `.claude/wiki-index.json`: 자동 인덱싱 (34 pages)
- `.claude/code-wiki-map.json`: 코드-wiki 매핑 (11 files)
- `backend/wiki_context.py`: 위키 컨텍스트 유틸리티
```

---

## 3. GPU Advisor에 적용하기 — 단계별 가이드

### Step 1: 디렉토리 구조 만들기

```
gpu-advisor/
├── wiki/                          # ← 새로 생성 (LLM이 관리)
│   ├── index.md                   # 위키 인덱스 (콘텐츠 중심)
│   ├── log.md                     # 작업 로그 (시간순)
│   ├── overview.md                # 프로젝트 전체 개요
│   ├── gpus/                      # GPU 모델별 페이지
│   │   ├── RTX_5060.md
│   │   ├── RTX_5070.md
│   │   └── RTX_5090.md
│   ├── concepts/                  # 개념 페이지
│   │   ├── mcts.md                # MCTS (몬테카를로 트리 탐색) 설명
│   │   ├── alphazero.md           # AlphaZero 아키텍처 설명
│   │   ├── feature_engineering.md # 256D 특징 공학
│   │   └── market_indicators.md   # 시장 지표 해설
│   ├── analysis/                  # 분석 결과 페이지
│   │   ├── price_trends.md        # 가격 트렌드 분석
│   │   └── market_sentiment.md    # 시장 감성 분석
│   └── sources/                   # 소스 요약 페이지
│       ├── danawa_data.md         # 다나와 데이터 설명
│       └── news_sentiment.md      # 뉴스 감성 분석 설명
```

### Step 2: Schema 파일 생성 (AGENTS.md) ✅ 완료

**Status**: `AGENTS.md`와 `CLAUDE.md` 모두 생성 완료

#### AGENTS.md (LLM/AI 에이전트용)
파일: `AGENTS.md` (프로젝트 루트)

**내용**:
- 역할 정의: wiki 관리자 (data/raw/는 읽기 전용)
- 3가지 운영: Ingest, Query, Lint
- 핵심 규칙: 파일 관리, 마크다운 형식, 데이터 정합성
- log.md 작성 규칙
- 체크리스트

**사용 대상**: OpenCode, 자동화 에이전트 등

#### CLAUDE.md (Claude Code용)
파일: `CLAUDE.md` (프로젝트 루트)

**내용**:
- 프로젝트 개요 및 아키텍처
- Wiki 시스템 설명
- Code-Wiki 매핑 (`.claude/code-wiki-map.json`)
- 개발 시 wiki 활용법
- 코드 변경 규칙
- Wiki 인덱스 구조

**사용 대상**: Claude Code (현재 사용 중)

> **차이점**: CLAUDE.md는 "코드 개발자를 위해 wiki를 어떻게 활용할 것인가", AGENTS.md는 "LLM 에이전트가 wiki를 어떻게 관리할 것인가"에 초점

### Step 3: 초기 위키 페이지 작성 ✅ 완료

**Status**: 34개 페이지 완성, 자동 인덱싱 구현

**완성된 구조**:
```
wiki/ (34 pages)
├── index.md              # 모든 페이지 카탈로그
├── log.md               # 작업 로그
├── overview.md          # 프로젝트 개요
├── gpus/ (21 모델)
│   ├── RTX_5090, RTX_5080, RTX_5070_Ti, RTX_5070, ...
│   └── RX_9070_XT, RX_9060_XT, RX_7900_XTX, ...
├── concepts/ (4 페이지)
│   ├── alphazero.md
│   ├── mcts.md
│   ├── feature_engineering.md
│   └── market_indicators.md
├── analysis/ (2 페이지)
│   ├── price_trends.md
│   └── market_sentiment.md
└── sources/ (3 페이지)
    ├── danawa_data.md
    ├── exchange_data.md
    └── news_data.md
```

**자동 인덱싱 시스템**:
- `scripts/generate_wiki_index.py`: 모든 wiki 페이지 인덱싱 → `.claude/wiki-index.json`
- `.claude/code-wiki-map.json`: 11개 소스 파일 → 관련 wiki 페이지 매핑
- `backend/wiki_context.py`: CLI 유틸리티로 파일의 관련 wiki 페이지 조회

**사용 예시**:
```bash
python3 backend/wiki_context.py "crawlers/feature_engineer.py"
# 출력: 관련 wiki 페이지 (feature_engineering.md, market_indicators.md)
```

### Step 4: 일일 데이터 수집과 위키 업데이트 연동 ✅ 준비 완료

**Current Status**: wiki_updater.py가 이미 `crawlers/run_daily.py` 후에 자동 실행

**기존 자동화**:
- `crawlers/run_daily.py`: 자정(00:00)에 LaunchAgent 자동 실행
- `crawlers/wiki_updater.py`: 크롤링 완료 후 자동으로 wiki 업데이트
  - `wiki/gpus/*.md` 최신 가격 반영
  - `wiki/analysis/price_trends.md`, `wiki/analysis/market_sentiment.md` 갱신
  - `wiki/index.md` 날짜 업데이트

**수동 트리거** (필요시):
```bash
# 오늘 크롤링 결과를 wiki에 반영
python3 crawlers/run_daily.py

# 또는 AGENTS.md 지침에 따라 LLM 에이전트에게 요청:
# "data/raw/2026-04-11/ 데이터를 wiki에 반영해줘"
```

**다음 단계** (선택):
- MCTS 점수 변동 반영: `wiki/gpus/RTX_5060.md` 의 "MCTS 구매 점수" 섹션
- 주간 분석: `wiki/analysis/weekly_*.md` 신규 페이지 작성
- 시장 변화 감지: 새로운 패턴 발견 시 `wiki/analysis/` 에 새 분석 페이지

---

## 4. 세 가지 핵심 운영 (Operations)

### 4.1 Ingest (소스 처리)

새 데이터가 들어왔을 때 위키에 통합하는 과정:

```
새 소스 (예: data/raw/danawa/2026-04-11.json)
    ↓
LLM이 소스를 읽고 주요 인사이트 파악
    ↓
사용자와 인사이트 논의
    ↓
wiki/gpus/RTX_5060.md 업데이트 (가격 변동, 트렌드)
    ↓
wiki/index.md 업데이트
    ↓
wiki/log.md에 항목 추가: "## [2026-04-11] ingest | Danawa 일일 가격"
```

**하나의 소스가 10~15개 위키 페이지에 영향을 줄 수 있습니다.**

### 4.2 Query (질문)

위키를 기반으로 질문에 답변:

```
질문: "RTX 5060 지금 사는 게 좋을까?"
    ↓
LLM이 wiki/index.md를 읽어 관련 페이지 파악
    ↓
wiki/gpus/RTX_5060.md, wiki/analysis/price_trends.md 등 읽기
    ↓
종합 답변 생성 (인용 포함)
    ↓
유용한 분석이면 → wiki/analysis/에 새 페이지로 저장 (지식 축적)
```

### 4.3 Lint (위키 검사)

주기적으로 위키 건강 상태 확인:

```
LLM에게 요청: "위키를 검사해줘"
    ↓
모순 확인 (A페이지와 B페이지의 정보 충돌)
    ↓
고아 페이지 확인 (어디에서도 링크되지 않은 페이지)
    ↓
누락된 교차 참조 추가
    ↓
최신 데이터 반영이 필요한 페이지 식별
    ↓
새로 조사할 질문/소스 제안
```

---

## 5. index.md와 log.md 운영법

### 5.1 index.md — 콘텐츠 중심 인덱스

```markdown
# GPU Advisor Wiki — 인덱스

## GPU 모델
| 페이지 | 요약 | 마지막 업데이트 |
|--------|------|----------------|
| [[RTX_5060]] | 현재 최적 가성비. 가격 하락 trend | 2026-04-11 |
| [[RTX_5070]] | 출시 대기 중. 예상가 75만원 | 2026-04-10 |
| [[RTX_5090]] | 플래그십. 수급 부족 | 2026-04-08 |

## 개념
| 페이지 | 요약 | 마지막 업데이트 |
|--------|------|----------------|
| [[mcts]] | 몬테카를로 트리 탐색 — 의사결정 탐색 알고리즘 | 2026-04-01 |
| [[alphazero]] | AlphaZero 아키텍처 — h/g/f + MCTS | 2026-04-01 |

## 분석
| 페이지 | 요약 | 마지막 업데이트 |
|--------|------|----------------|
| [[price_trends]] | 전체 GPU 가격 트렌드 (30일) | 2026-04-11 |

## 소스
| 페이지 | 요약 | 마지막 업데이트 |
|--------|------|----------------|
| [[danawa_data]] | 다나와 가격 크롤러 — 24개 모델 | 2026-04-01 |
```

### 5.2 log.md — 시간순 작업 로그

```markdown
# GPU Advisor Wiki — 작업 로그

## [2026-04-11] ingest | Danawa 일일 가격
- RTX 5060: 45.2만원 (전일 대비 -0.8만원)
- RTX 5090: 수급 부족 지속
- 업데이트된 페이지: RTX_5060.md, price_trends.md, index.md

## [2026-04-10] query | RTX 5060 구매 타이밍 분석
- 질문: "RTX 5060 지금 살까?"
- 결과: MCTS 구매 점수 72% → "구매 권장"
- 새 페이지 작성: wiki/analysis/rtx5060_buy_analysis.md

## [2026-04-09] lint | 전체 위키 검사
- 모순: 없음
- 고아 페이지: market_sentiment.md (링크 추가함)
- 누락: JPY 환율 영향 페이지 필요 (TODO)
```

> **팁**: `grep "^## \[" log.md | tail -5` 명령으로 최근 5개 항목을 빠르게 볼 수 있습니다.

---

## 6. Obsidian과 함께 사용하기

이 위키 패턴은 **Obsidian**을 뷰어로 사용하면 가장 효과적입니다:

### 6.1 설정

1. **Obsidian 설치** → [obsidian.md](https://obsidian.md)
2. **Vault로 `gpu-advisor/` 디렉토리 열기**
3. 왼쪽에 파일 탐색기, 가운데 에디터, 오른쪽에 그래프 뷰 배치

### 6.2 유용한 Obsidian 기능

| 기능 | 설명 |
|------|------|
| **Graph View** | 위키 페이지 간 연결 관계를 시각화. 어떤 페이지가 허브인지, 고아인지 한눈에 확인 |
| **Backlinks** | 현재 페이지를 참조하는 다른 페이지 목록. 자동 생성 |
| **Web Clipper** | 브라우저 확장으로 웹 글을 마크다운으로 저장. GPU 관련 기사를 `data/raw/`에 바로 저장 |
| **Dataview** | YAML frontmatter 기반 동적 쿼리. GPU 가격 테이블 등을 자동 생성 |
| **Marp** | 마크다운 기반 슬라이드 덱. 위키 내용으로 바로 프레젠테이션 생성 |

### 6.3 이미지 다운로드 설정 (선택)

```
Obsidian Settings → Files and links → Attachment folder path: raw/assets/
Hotkeys → "Download attachments for current file" → Ctrl+Shift+D 바인딩
```

웹 클리퍼로 글을 저장한 후 Ctrl+Shift+D를 누르면 모든 이미지가 로컬에 다운로드됩니다.

---

## 7. LLM 에이전트 선택 ✅ 완료

위키 패턴은 다양한 LLM 에이전트에서 작동합니다:

| 에이전트 | Schema 파일 | 상태 | 용도 |
|----------|------------|------|------|
| **Claude Code** | `CLAUDE.md` ✅ | 완성 | 코드 개발, IDE 통합, wiki 활용 |
| **OpenCode** | `AGENTS.md` ✅ | 완성 | 자동화 에이전트, wiki 관리 |
| **자동화 스크립트** | `AGENTS.md` ✅ | 완성 | 일일 wiki 업데이트 자동화 |

**CLAUDE.md** (코드 개발자용):
- 코드 파일과 wiki 페이지 매핑 (`.claude/code-wiki-map.json`)
- 개발 시 참조할 wiki 페이지 찾기
- 코드 변경 시 wiki 동기화 규칙

**AGENTS.md** (LLM/자동화용):
- wiki 생성/유지/검사 규칙
- 3가지 운영 (Ingest, Query, Lint)
- log.md 작성 규칙
- 데이터 정합성 원칙

> **현재 상태**: 둘 다 완성되어 여러 에이전트가 동시에 활용 가능

---

## 8. GPU Advisor 특화 워크플로우

### 8.1 일일 루틴

```
매일 자정: crawlers/run_daily.py 자동 실행
    ↓ (크롤링 완료 후)
LLM 에이전트에게: "오늘 데이터를 위키에 반영해줘"
    ↓
LLM이 data/raw/의 새 파일을 읽고 wiki/ 업데이트
    ↓
Obsidian에서 업데이트된 페이지 확인
```

### 8.2 주간 분석

```
LLM 에이전트에게: "이번 주 가격 트렌드를 분석해줘"
    ↓
LLM이 wiki/의 관련 페이지들을 종합
    ↓
wiki/analysis/weekly_2026_W15.md 작성
    ↓
index.md에 새 분석 페이지 등록
```

### 8.3 구매 결정 시

```
LLM 에이전트에게: "RTX 5060 구매 타이밍 분석해줘"
    ↓
LLM이 위키 + 백엔드 API(/api/ask) 결과를 종합
    ↓
wiki/analysis/rtx5060_purchase_timing.md 작성
    ↓
관련 GPU 페이지에 분석 링크 추가
```

---

## 9. 시작하기 — 체크리스트 ✅ 완료

### 기초 작업 ✅ 완료

- [x] `wiki/` 디렉토리 생성 (34 pages)
- [x] `wiki/index.md` 작성
- [x] `wiki/log.md` 작성
- [x] `CLAUDE.md` Schema 파일 생성
- [x] `AGENTS.md` Schema 파일 생성
- [x] 초기 위키 구축 완료

### 자동화 시스템 ✅ 완료

- [x] Wiki 인덱싱: `scripts/generate_wiki_index.py` + `.claude/wiki-index.json`
- [x] Code-Wiki 매핑: `.claude/code-wiki-map.json` + `backend/wiki_context.py`
- [x] 일일 wiki 업데이트: `crawlers/wiki_updater.py` (자동 실행)
- [x] Claude Code 통합: `CLAUDE.md` 개발 규칙서

### 다음 단계 (선택)

- [ ] Obsidian으로 `gpu-advisor/` Vault 열기 (Graph View, Backlinks 활용)
- [ ] MCTS 점수 변동 자동 반영 (위키 업데이트 강화)
- [ ] 주간/월간 분석 페이지 자동 생성
- [ ] Obsidian Web Clipper → GPU 관련 기사 자동 수집
- [ ] Dataview 플러그인 → 동적 GPU 가격 테이블

### 심화 작업 (필요 시)

- [ ] 위키가 100+ 페이지일 때: `qmd` 검색 엔진 도입
- [ ] Marp로 시장 분석 슬라이드 자동 생성
- [ ] 위키 변화 자동 감시 (변화 감지 → 자동 알림)

---

## 10. FAQ

### Q: 기존 docs/와 wiki/의 차이는?
**docs/**는 개발자를 위한 기술 문서 (아키텍처, API, 튜토리얼). 수정 빈도가 낮음.
**wiki/**는 운영 지식 베이스 (가격 트렌드, GPU 분석, 시장 인사이트). 매일 업데이트됨. LLM이 관리.

### Q: docs/reports/와 wiki/log.md는 어떻게 다른가?
`docs/reports/`는 시스템이 자동 생성하는 일일 상태 보고서.
`wiki/log.md`는 LLM이 관리하는 위키 변경 이력. 인간이 읽기 쉬운 형태.

### Q: 위키가 너무 커지면?
- `index.md` 기반 탐색이 100페이지 이상에서는 부족해질 수 있음
- 그때 `qmd` 같은 로컬 검색 엔진 도입을 고려
- 또는 카테고리별 하위 인덱스 분리 (예: `wiki/gpus/index.md`)

### Q: LLM 에이전트가 위키를 망가뜨리면?
- 위키는 git으로 버전 관리 → `git diff`로 변경 확인, `git checkout`으로 복구 가능
- Schema 파일에 엄격한 규칙을 정의하여 방지
- 중요한 변경 전에 사용자 승인을 요구하도록 설정

---

## 11. 요약 ✅ 완성

| 항목 | 내용 | 상태 |
|------|------|------|
| **패턴** | LLM이 영구적으로 관리하는 개인 지식 베이스 | ✅ |
| **3계층** | Raw Sources → Wiki → Schema | ✅ |
| **Wiki 페이지** | 34개 (concepts, sources, analysis, gpus) | ✅ |
| **Schema 파일** | CLAUDE.md (개발), AGENTS.md (LLM) | ✅ |
| **자동화 인덱싱** | wiki-index.json (34 pages), code-wiki-map.json (11 files) | ✅ |
| **Context 유틸** | backend/wiki_context.py (CLI 조회) | ✅ |
| **3가지 운영** | Ingest, Query, Lint (AGENTS.md 정의) | ✅ |
| **2개 핵심 파일** | index.md (콘텐츠), log.md (작업 로그) | ✅ |
| **뷰어** | Obsidian (Graph View, Backlinks, Web Clipper) | 추천 |
| **자동 시스템** | crawlers/wiki_updater.py (매일 자동) | ✅ |

### 생성된 파일 목록

```
gpu-advisor/
├── CLAUDE.md                          # Claude Code 개발 규칙 ✅
├── AGENTS.md                          # LLM 에이전트 지침 ✅
├── .claude/
│   ├── wiki-index.json                # 34개 페이지 인덱스 ✅
│   └── code-wiki-map.json             # 11개 파일 매핑 ✅
├── backend/
│   └── wiki_context.py                # Wiki 컨텍스트 유틸 ✅
├── scripts/
│   └── generate_wiki_index.py         # 인덱스 생성 스크립트 ✅
├── wiki/                              # 34개 페이지 ✅
│   ├── index.md, log.md, overview.md
│   ├── gpus/ (21 모델)
│   ├── concepts/ (4개)
│   ├── analysis/ (2개)
│   └── sources/ (3개)
└── docs/
    └── LLM_WIKI_적용_가이드.md         # 이 가이드 문서 ✅
```

> **핵심**: 인간은 **소싱, 방향 설정, 올바른 질문**을 담당. LLM은 **요약, 교차참조, 정리, 관리** 등 모든 단순 반복 작업을 처리. 위키는 시간이 지나면서 더욱 강력해지는 복리적 지식 자산입니다.

### 다음 행동

1. **wiki_updater.py 검증**: 실제로 매일 wiki가 업데이트되는지 모니터링
2. **MCTS 점수 반영**: GPU 구매 분석에 MCTS 예측값 추가
3. **Obsidian 활용**: Graph View로 wiki 네트워크 시각화
4. **주간 리뷰**: log.md를 읽고 인사이트 도출
