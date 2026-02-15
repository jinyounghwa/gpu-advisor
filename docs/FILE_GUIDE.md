# GPU Advisor: 파일별 상세 안내서 (File-by-File Guide)

이 문서는 프로젝트 내의 모든 주요 파일을 하나하나 설명하여, 여러분이 이 프로젝트를 직접 구현하거나 수정할 때 지침으로 삼을 수 있도록 돕습니다.

---

## 📁 1. 데이터 수집 시스템 (`crawlers/`)
데이터를 매일 수집하고 AI 학습용 특징(Feature)으로 변환하는 모듈입니다.

- **`danawa_crawler.py`**: BeautifulSoup을 사용하여 다나와 웹사이트에서 24개 GPU 모델의 최저가, 판매처 수, 재고 현황을 파싱합니다.
- **`exchange_rate_crawler.py`**: 주요 환율(USD, JPY, EUR)을 수집합니다. 수입 부품인 GPU 가격 결정의 핵심 요인을 파악합니다.
- **`news_crawler.py`**: GPU 관련 뉴스를 수집하고, 간단한 텍스트 분석을 통해 시장의 긍정/부정 감정(Sentiment)을 점수화합니다.
- **`feature_engineer.py`**: 수집된 원본 데이터를 256차원의 AI 입력용 벡터로 변환합니다. 이동평균, RSI, 변동성 등 기술적 지표 계산 로직이 들어있습니다.
- **`run_daily.py`**: 위의 4개 모듈을 순차적으로 가동시키는 오케스트레이터입니다. `cron`에 의해 매일 호출됩니다.

---

## 📁 2. 백엔드 및 서비스 (`backend/`)
AI 모델을 가동하고, 프론트엔드와 통신하며, 학습 프로세스를 관리하는 서버입니다.

- **`simple_server.py`**: FastAPI 기반의 메인 서버 파일입니다. 검색 API(`/api/ask`)와 대시보드용 학습 시뮬레이션 API들을 정의합니다.
- **`train_alphazero_v2.py`**: MuZero 스타일의 자가 학습(Self-play) 과정을 구현한 파일입니다. MCTS를 통해 생성된 데이터를 바탕으로 신경망을 업데이트합니다.
- **`requirements.txt`**: PyTorch, FastAPI, uvicorn 등 백엔드 실행에 필요한 파이썬 패키지 목록입니다.

---

## 📁 3. AI 핵심 모델 (`backend/models/`)
AlphaZero/MuZero 아키텍처를 구성하는 신경망들입니다.

- **`representation_network.py` (h)**: 원시 입력을 256차원의 잠재 상태(Latent State)로 변환합니다. 시장 상황을 AI만의 언어로 요약하는 단계입니다.
- **`dynamics_network.py` (g)**: '가상 세계 모델'입니다. 특정 행동 후의 미래 상태와 예상 보상을 상상합니다.
- **`prediction_network.py` (f)**: 정책(어떤 행동을 할지)과 가치(현재 얼마나 유리한지)를 예측합니다.
- **`mcts.py`**: Monte Carlo Tree Search의 구현체입니다. 위의 세 신경망을 조합하여 수십 번의 시나리오를 시뮬레이션합니다.
- **`transformer_model.py`**: 신경망 내부에서 시퀀스 데이터를 처리하기 위한 Transformer 블록 구현이 포함되어 있습니다.

---

## 📁 4. 프론트엔드 (`frontend/`)
사용자가 GPU를 검색하고 학습 상황을 모니터링하는 웹 인터페이스입니다.

- **`app/page.tsx`**: 메인 페이지입니다. 검색창, 탭 메뉴(Advisor/Training), 결과 표시 레이아웃이 포함되어 있습니다.
- **`app/components/Dashboard.tsx`**: 지표 카드와 차트를 배치하여 실시간 대시보드의 전반적인 구조를 잡습니다.
- **`app/components/Chart.tsx`**: `recharts` 라이브러리를 사용하여 TPS, Loss, VRAM 등을 시각화하는 차트 컴포넌트입니다.
- **`app/components/TrainingDashboard.tsx`**: 백엔드와 Server-Sent Events(SSE)로 연결되어 실시간 학습 데이터를 받아오는 로직이 들어있습니다.

---

## 📁 5. 데이터 및 로그 (`data/`, `logs/`)
- **`data/raw/`**: 크롤링된 날것의 JSON 파일들이 날짜별로 저장됩니다.
- **`data/processed/`**: AI 학습에 즉시 사용 가능한 가공된 데이터셋이 저장됩니다.
- **`logs/`**: 시스템 운영 및 학습 로그가 기록되어 문제 발생 시 추적에 사용됩니다.

---

## 🛠️ 6. 루트 스크립트 및 문서
- **`setup_cron.sh`**: 매일 자정에 `run_daily.py`가 실행되도록 시스템 스케줄러를 자동 설정해주는 쉘 스크립트입니다.
- **`run_all.sh`**: 서비스의 백엔드와 프론트엔드를 한꺼번에 실행시키는 편의 도구입니다.
- **`README.md`**: 프로젝트 설치 및 퀵스타트 안내가 포함된 대문 문서입니다.

---

## 💡 구현 시 주의사항 (Implementation Tips)

1.  **경로 설정**: `backend.log`나 데이터 저장 경로가 하드코딩된 부분이 있을 수 있으니, 각 파일 상단의 경로 설정 부분을 확인하세요.
2.  **가속 하드웨어**: `representation_network.py` 등 모델 파일 내에서 `mps`(Apple Silicon) 또는 `cuda`(NVIDIA) 사용 여부를 환경에 맞게 자동 선택하도록 구현되어 있습니다.
3.  **데이터 무결성**: 크롤링 도중 웹 사이트 구조 변경으로 파싱이 실패할 수 있으니 `danawa_crawler.py`의 CSS 선택자를 주기적으로 확인하는 것이 좋습니다.

이 가이드를 바탕으로 각 코드를 읽어보시면, 전체 시스템이 어떻게 조립되는지 명확히 이해하실 수 있을 것입니다.
