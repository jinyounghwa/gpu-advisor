# GPU Advisor 학습 가이드 (Study Guide)

## 프로젝트 개요
**GPU Advisor**는 단순한 가격 예측 시스템이 아닙니다. 바둑 AI인 **AlphaZero/MuZero**의 아키텍처를 응용하여, "지금 GPU를 사는 것이 나중보다 얼마나 유리한가(승률/이득 확률)"를 계산하는 전략적 의사결정 보조 시스템입니다.

이 가이드는 프로젝트의 원리부터 코드 구현, 그리고 전체 시스템이 어떻게 연결되는지 단계별로 설명합니다.

---

## 학습 로드맵 (4주 과정)

### [1주차] 데이터 수집 시스템 (Crawlers)
원천 데이터가 없으면 AI도 없습니다. 매일 신선한 데이터를 수집하는 과정을 배웁니다.

- **핵심 모듈:** `crawlers/danawa_crawler.py`, `crawlers/exchange_rate_crawler.py`, `crawlers/news_crawler.py`
- **학습 포인트:**
    - `BeautifulSoup`을 이용한 다나와 가격 파싱 원리
    - 외부 API를 통한 환율 데이터 수집
    - 뉴스 데이터의 텍스트 분석 및 감정 점수(`sentiment`) 산출
- **실습:**
    - `python3 crawlers/run_daily.py --skip-release`를 실행하여 `data/raw/`에 JSON 파일이 생성되는지 확인하세요.
    - macOS에서는 `cron` 대신 **LaunchAgent**로 자동화합니다. `~/Library/LaunchAgents/com.gpu-advisor.daily-crawl.plist`를 열어 자동화 설정을 공부하세요.
    - 로그는 `~/Library/Logs/gpu-advisor/cron.log`에서 확인합니다 (macOS TCC 보호로 `~/Documents/` 경로 사용 불가).

### [2주차] Feature Engineering (The Bridge)
수집된 날것의 데이터를 AI가 이해할 수 있는 256차원의 벡터로 변환하는 과정입니다.

- **핵심 모듈:** `crawlers/feature_engineer.py`
- **학습 포인트:**
    - **256D 피처 구성**: 가격(60D), 환율(20D), 뉴스(30D), 시장(20D), 시간(20D), 기술지표(106D) — 합계 256D
    - 이동평균(MA7/MA14/MA30), RSI, MACD 같은 기술적 지표를 생성하는 이유
    - **정규화(Normalization):** 서로 다른 단위(원, 달러, 점수)를 AI가 처리하기 좋은 형태로 맞추는 과정
- **실습:**
    - `data/processed/dataset/` 폴더의 JSON 파일을 열어보세요. 원본 데이터가 어떻게 256개의 숫자로 변했는지 관찰하세요.

### [3주차] AlphaZero/MuZero AI 엔진 (The Brain)
이 프로젝트의 핵심인 강화학습 모델 구조를 심층 학습합니다.

- **핵심 모듈:** `backend/models/` (representation, dynamics, prediction, action_model, mcts_engine)
- **학습 포인트:**
    - **Representation Network (h):** 256D 시장 상태를 잠재 상태(latent state)로 압축. `Linear(256, 256)` + 잔차 블록 3개
    - **Dynamics Network (g):** 특정 행동을 했을 때 미래 상태와 보상(μ, σ²)을 예측하는 '월드 모델'. 잔차 블록 4개 + Gaussian NLL 불확실성 헤드
    - **Prediction Network (f):** 현재 상태에서 어떤 행동이 좋은지(Policy)와 예상 이득(Value)을 출력. 잔차 블록 4개
    - **Action Model (a):** 잠재 상태에서 행동 사전 확률을 학습. `ActionEmbeddingLayer(16D)` + `ActionPriorNetwork(256→128→64→5)` (~43K 파라미터)
    - **MCTS (PUCT 방식):** `Q(s,a) + c·P(s,a)·√N_parent/(1+N_child)` (c=√2). 루트 노드에 디리클레 노이즈(ε=0.25, α=0.03)로 탐색 다양성 확보
    - **4-신호 블렌드:** 추론 시 `0.45×MCTS + 0.25×보상 + 0.15×f-prior + 0.15×ActionModel` 가중 합산
- **실습:**
    - `backend/run_release_ready.py`를 읽으며 학습→평가→게이트 판정→리포트 생성 전체 파이프라인을 추적하세요.
    - `backend/agent/fine_tuner.py`의 `train_step()` 함수에서 5개 손실 가중치(`latent×1.0 + policy×1.0 + value×1.0 + NLL×0.5 + action_prior×0.3`)를 분석하세요.

### [4주차] 풀스택 통합 (Backend & Frontend)
AI 모델을 API 서버로 서빙하고, 사용자가 볼 수 있는 웹 화면에 연결합니다.

- **핵심 모듈:** `backend/simple_server.py`, `frontend/app/page.tsx`
- **학습 포인트:**
    - **FastAPI:** AI 모델(`pth` 파일)을 로드하여 REST API 엔드포인트(`POST /api/ask`)를 만드는 방법
    - **Next.js Connection:** 프론트엔드에서 `fetch`를 사용하여 백엔드 데이터를 가져오고 상태(State)로 관리하는 흐름
    - **자동 학습 오케스트레이션:** `crawlers/auto_training.py`의 `decide_auto_training_action()` 함수가 30일 도달 여부를 판단해 학습을 자동 트리거하는 방식
- **실습:**
    - 백엔드 서버를 띄우고(`python3 backend/simple_server.py`), 프론트엔드(`npm run dev`)에서 검색했을 때 데이터가 어떻게 오가는지 Network 탭에서 확인하세요.

---

## 백엔드 연결 상세 가이드

가장 많이 묻는 질문인 **"프론트엔드와 백엔드가 어떻게 연결되나요?"**에 대한 기술적 설명입니다.

### 1. 연결 원리
- **통신 방식:** HTTP REST API (주로 POST 방식)
- **백엔드 주소:** `http://localhost:8000`
- **프론트엔드 주소:** `http://localhost:3000`

### 2. 주요 코드 구현
#### [백엔드: FastAPI]
`backend/simple_server.py`에서 API를 정의합니다.
```python
@app.post("/api/ask")
async def ask_gpu_advisor(request: GPURequest):
    # 1. 사용자가 입력한 모델명 수신
    # 2. AI 모델을 통한 분석 수행 (MCTS + 4-신호 블렌드)
    # 3. 결과 JSON 반환
    return {
        "title": "RTX 4090",
        "recommendation": "지금 구매하는 것을 추천합니다.",
        "action_probs": {...},
        "expected_reward": 0.0027,
    }
```

#### [프론트엔드: React/Next.js]
`frontend/app/page.tsx`에서 백엔드에 요청을 보냅니다.
```typescript
const searchGPU = async (query: string) => {
  setLoading(true);
  try {
    const res = await fetch("http://localhost:8000/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: query }),
    });
    const data = await res.json();
    setResult(data); // 받은 데이터를 화면에 표시하기 위해 상태 저장
  } catch (error) {
    setError("연결 실패");
  } finally {
    setLoading(false);
  }
};
```

---

## 주요 원리 요약

### 왜 가격만 예측하지 않고 AlphaZero를 쓰나요?
가격은 주식처럼 변동성이 큽니다. 단순히 "내일은 5,000원 쌀 거다"라는 예측보다는, "지금 사면 나중에 더 싼 기회를 놓칠 확률이 30%이고, 지금 안 사면 재고가 떨어질 위험이 20%다"라는 식의 **종합적인 승률**을 계산하는 것이 구매자에게 더 큰 가치를 주기 때문입니다.

### MCTS가 하는 일은 무엇인가요?
바둑 AI가 수십 번의 가상 대국을 시뮬레이션해서 최선의 수를 찾듯, MCTS는 50번의 미래 시나리오(5일 앞)를 시뮬레이션합니다. PUCT 공식(`Q + c·P·√N/N_child`)으로 유망한 행동을 우선 탐색하고, 디리클레 노이즈로 탐색 다양성을 확보합니다.

### 256차원 Feature의 의미
AI에게 "가격" 한 가지만 알려주는 것은 장님에게 코끼리 다리만 만지게 하는 것과 같습니다.
- **7일/30일 이동평균:** 가격의 추세를 알려줍니다.
- **RSI/MACD:** 현재 가격이 거품인지 아닌지 기술적으로 알려줍니다.
- **뉴스 감정 점수:** 시장의 심리적 분위기를 알려줍니다.
이 모든 정보가 합쳐져 256개의 숫자가 되었을 때, AI는 비로소 전체 시장의 상황을 입체적으로 이해하게 됩니다.

### ActionModel은 왜 별도로 있나요?
기존에는 `utility_bias`라는 하드코딩된 휴리스틱으로 행동 경향을 보정했습니다. ActionModel(a)은 이를 **데이터에서 학습된 행동 사전 확률**로 대체합니다. "지금 시장 상태라면 어떤 행동이 선험적으로 적절한가?"를 약 43K 파라미터로 학습합니다.

---

## 프로젝트 시작하기 (Step-by-Step)

1. **가상환경 설정 및 패키지 설치**
   ```bash
   cd gpu-advisor
   pip install -r backend/requirements.txt
   ```

2. **데이터 수집 체험**
   ```bash
   python3 crawlers/run_daily.py --skip-release
   ```

3. **릴리즈 드라이 체크 (학습 없이 게이트 확인)**
   ```bash
   python3 backend/run_release_daily.py
   ```

4. **백엔드 서버 가동**
   ```bash
   python3 backend/simple_server.py
   ```

5. **프론트엔드 웹 실행**
   ```bash
   cd frontend
   npm install  # 첫 실행시
   npm run dev
   ```

이제 브라우저에서 `http://localhost:3000`에 접속하여 여러분이 만든 AI GPU 어드바이저를 경험해보세요!

---

## 주요 문서 참조

| 문서 | 내용 |
|------|------|
| `docs/HYPERPARAMETER_GUIDE_KR.md` | 모든 하이퍼파라미터 설계 근거 |
| `docs/AUTO_TRAINING_WORKFLOW_KR.md` | 30일 자동 학습 오케스트레이션 |
| `docs/MCTS_WALKTHROUGH.md` | MCTS 탐색 상세 워크스루 |
| `docs/INFERENCE_WALKTHROUGH_KR.md` | 추론 파이프라인 단계별 설명 |
| `docs/SAFETY_MECHANISMS_KR.md` | 안전장치 상세 설명 |
| `docs/POST_30D_NEXT_STEPS_KR.md` | 30일 도달 후 실행 가이드 |
