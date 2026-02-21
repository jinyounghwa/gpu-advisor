# GPU Advisor: 개발 가이드 (Implementation Step-by-Step)

이 문서는 아무것도 없는 상태에서 이 프로젝트를 처음부터 구현하고자 하는 개발자를 위한 단계별 가이드입니다. 문서의 순서를 따라가면 프로젝트의 뼈대부터 완성까지 경험할 수 있습니다.

---

## 🏗️ 1단계: 데이터 파이프라인 구축 (The Foundation)
AI의 성능은 데이터의 질에 결정됩니다. 먼저 데이터를 수집하는 환경을 만듭니다.

1.  **크롤러 구현 (`crawlers/`)**:
    *   `danawa_crawler.py`를 작성하여 가격 정보를 가져옵니다. 이때 사이트 차단을 방지하기 위해 `User-Agent` 설정에 유의하세요.
    *   `exchange_rate_crawler.py`를 구현해 환율 정보를 수집합니다.
    *   `news_crawler.py`로 시장 분위기를 점수화(Sentiment Analysis)하는 로직을 추가합니다.
2.  **데이터 가공 (`feature_engineer.py`)**:
    *   수집한 CSV/JSON 데이터를 읽어 시계열 데이터(Moving Average, RSI)로 변환하는 클래스를 만듭니다.
    *   값이 너무 큰 가격 정보 등을 0~1 사이의 값으로 **정규화(Normalization)** 하는 과정이 필수적입니다.
3.  **자동화 스케줄링**:
    *   `setup_cron.sh`를 참고해 리눅스/맥 시스템의 `crontab`에 수집 스크립트(`run_daily.py`)를 등록합니다.

---

## 🧠 2단계: AI 두뇌 설계 (The Brain)
수집된 데이터를 처리할 MuZero/AlphaZero 아키텍처를 설계합니다.

1.  **네트워크 정의 (`backend/models/`)**:
    *   `representation_network.py`: 입력을 AI가 이해하는 벡터로 변환.
    *   `dynamics_network.py`: 가상 세계의 규칙(물리학)을 학습.
    *   `prediction_network.py`: 최선의 선택과 그 가치를 판단.
2.  **MCTS 알고리즘 (`mcts.py`)**:
    *   신경망 혼자 결정하는 것이 아니라, 수십 번의 미래를 시뮬레이션해보고 가장 방문 횟수가 많은(검증된) 길을 선택하는 로직을 구현합니다.
3.  **학습 로직 (`train_alphazero_v2.py`)**:
    *   수집된 데이터셋을 불러와 모델을 훈련시키는 루프를 작성합니다.
    *   PyTorch의 `Adam` 옵티마이저와 `CrossEntropy`, `MSE` 손실 함수를 조합합니다.

---

## 🔌 3단계: 백엔드 API 서버 (The Connection)
AI 모델과 외부 세상을 연결하는 문을 만듭니다.

1.  **FastAPI 서버 (`backend/simple_server.py`)**:
    *   `/api/ask`: 사용자가 모델명을 넘기면 AI가 예측값을 계산해 반환하는 엔드포인트를 만듭니다.
    *   `/api/training/metrics/stream`: 학습 진행 상황을 실시간으로 프론트엔드에 쏴주는 SSE(Server-Sent Events) 스트림을 구축합니다.
2.  **CORS 설정**: 프론트엔드(포트 3000)에서 백엔드(포트 8000)로의 접근을 허용하도록 미들웨어를 설정합니다.

---

## 🎨 4단계: 프론트엔드 대시보드 (The Interface)
사용자가 AI의 판단을 눈으로 확인하는 공간입니다.

1.  **UI 레이아웃 (`frontend/app/page.tsx`)**:
    *   Next.js와 Tailwind CSS를 사용하여 현대적이고 깔끔한 검색 인터페이스를 구현합니다.
2.  **실시간 모니터링 (`TrainingDashboard.tsx`)**:
    *   백엔드의 SSE 스트림을 구독(Subscribe)하여 학습 지표(Loss, TPS 등)가 들어올 때마다 상태를 업데이트합니다.
3.  **시각화 (`Chart.tsx`)**:
    *   `recharts` 라이브러리를 사용해 데이터의 흐름을 그래프로 그려냅니다.

---

## ✅ 5단계: 통합 및 검증 (Final Check)

1.  **통합 실행**: `run_all.sh`와 같이 백엔드와 프론트엔드를 동시에 띄우는 스크립트로 전체 시스템의 조화를 확인합니다.
2.  **예측 테스트**: 브라우저에서 `RTX 5090` 등을 검색해보고, API 응답이 정상적으로 돌아오는지 Network 탭에서 모니터링합니다.

---

## 💡 개발자 팁
*   **초기 검증용 합성 데이터(운영 제외)**: 실제 시장 데이터가 쌓이는 데는 최소 30일이 걸립니다. 초기 기능 검증 단계에서는 합성 데이터를 사용할 수 있지만, 운영 학습/판정은 실데이터 윈도우를 기준으로 진행해야 합니다.
*   **에러 핸들링**: 크롤링 서버가 죽거나 백엔드가 응답하지 않을 때 프론트엔드에서 사용자에게 적절한 안내(Loading spinner, Error message)를 제공하는 것이 중요합니다.
