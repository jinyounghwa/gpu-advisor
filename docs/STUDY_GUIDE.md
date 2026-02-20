# GPU Advisor 학습 가이드 (Study Guide)

## 🎓 프로젝트 개요
**GPU Advisor**는 단순한 가격 예측 시스템이 아닙니다. 바둑 AI인 **AlphaZero/MuZero**의 아키텍처를 응용하여, "지금 GPU를 사는 것이 나중보다 얼마나 유리한가(승률/이득 확률)"를 계산하는 전략적 의사결정 보조 시스템입니다.

이 가이드는 프로젝트의 원리부터 코드 구현, 그리고 전체 시스템이 어떻게 연결되는지 단계별로 설명합니다.

---

## 📅 학습 로드맵 (4주 과정)

### [1주차] 데이터 수집 시스템 (Crawlers)
원천 데이터가 없으면 AI도 없습니다. 매일 신선한 데이터를 수집하는 과정을 배웁니다.

- **핵심 모듈:** `crawlers/danawa_crawler.py`, `crawlers/exchange_rate_crawler.py`, `crawlers/news_crawler.py`
- **학습 포인트:**
    - `BeautifulSoup`을 이용한 다나와 가격 파싱 원리
    - 외부 API를 통한 환율 데이터 수집
    - 뉴스 데이터의 텍스트 분석 및 감정 점수(`sentiment`) 산출
- **실습:** 
    - `python3 crawlers/run_daily.py`를 실행하여 `data/raw/`에 JSON 파일이 생성되는지 확인하세요.
    - `setup_cron.sh`를 통해 리눅스/맥의 `cron`에 어떻게 자동화 스케줄을 등록하는지 공부하세요.

### [2주차] Feature Engineering (The Bridge)
수집된 날것의 데이터를 AI가 이해할 수 있는 256차원의 벡터로 변환하는 과정입니다.

- **핵심 모듈:** `crawlers/feature_engineer.py`
- **학습 포인트:**
    - **11D → 256D 확장:** 가격, 환율, 뉴스 점수를 넘어서 이동평균(MA), RSI, MACD 같은 기술적 지표를 생성하는 이유
    - **정규화(Normalization):** 서로 다른 단위(원, 달러, 점수)를 0~1 사이의 값으로 맞추는 과정
- **실습:**
    - `data/processed/dataset/` 폴더의 JSON 파일을 열어보세요. 원본 데이터가 어떻게 256개의 숫자로 변했는지 관찰하세요.

### [3주차] AlphaZero/MuZero AI 엔진 (The Brain)
이 프로젝트의 핵심인 강화학습 모델 구조를 심층 학습합니다.

- **핵심 모듈:** `backend/models/` (representation, dynamics, prediction network)
- **학습 포인트:**
    - **Representation Network (h):** 현재 시장 상태를 내부 표상(Latent State)으로 압축
    - **Dynamics Network (g):** 특정 행동(예: 1주 대기)을 했을 때 미래 상태와 보상을 예측하는 '월드 모델'
    - **Prediction Network (f):** 현재 상태에서 어떤 행동이 좋은지(Policy)와 예상 이득(Value)을 출력
    - **MCTS (Monte Carlo Tree Search):** 수십 번의 미래 시뮬레이션을 통해 최적의 수를 찾는 탐색 알고리즘
- **실습:**
    - `backend/train_alphazero_v2.py` 코드를 읽으며 모델이 어떻게 가중치를 업데이트하는지(Loss 함수) 분석하세요.

### [4주차] 풀스택 통합 (Backend & Frontend)
AI 모델을 API 서버로 서빙하고, 사용자가 볼 수 있는 웹 화면에 연결합니다.

- **핵심 모듈:** `backend/simple_server.py`, `frontend/app/page.tsx`
- **학습 포인트:**
    - **FastAPI:** AI 모델(`pth` 파일)을 로드하여 REST API 엔드포인트(`POST /api/ask`)를 만드는 방법
    - **Next.js Connection:** 프론트엔드에서 `fetch`를 사용하여 백엔드 데이터를 가져오고 상태(State)로 관리하는 흐름
    - **실시간 지표:** 학습 진행 상황을 스트리밍하는 원리
- **실습:**
    - 백엔드 서버를 띄우고(`python3 simple_server.py`), 프론트엔드(`npm run dev`)에서 검색했을 때 데이터가 어떻게 오가는지 Network 탭에서 확인하세요.

---

## 🛠️ 백엔드 연결 상세 가이드

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
    # 2. AI 모델을 통한 분석 수행
    # 3. 결과 JSON 반환
    return {
        "title": "RTX 4090",
        "recommendation": "지금 구매하는 것을 추천합니다.",
        "specs": "24GB VRAM, DLSS 3.0...",
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

## 💡 주요 원리 요약

### 왜 가격만 예측하지 않고 AlphaZero를 쓰나요?
가격은 주식처럼 변동성이 큽니다. 단순히 "내일은 5,000원 쌀 거다"라는 예측보다는, "지금 사면 나중에 더 싼 기회를 놓칠 확률이 30%이고, 지금 안 사면 재고가 떨어질 위험이 20%다"라는 식의 **종합적인 승률**을 계산하는 것이 구매자에게 더 큰 가치를 주기 때문입니다.

### 256차원 Feature의 의미
AI에게 "가격" 한 가지만 알려주는 것은 장님에게 코끼리 다리만 만지게 하는 것과 같습니다.
- **7일/30일 이동평균:** 가격의 추세를 알려줍니다.
- **RSI/MACD:** 현재 가격이 거품인지 아닌지 기술적으로 알려줍니다.
- **뉴스 감정 점수:** 시장의 심리적 분위기를 알려줍니다.
이 모든 정보가 합쳐져 256개의 숫자가 되었을 때, AI는 비로소 전체 시장의 상황을 입체적으로 이해하게 됩니다.

---

## 🚀 프로젝트 시작하기 (Step-by-Step)

1. **가상환경 설정 및 패키지 설치**
   ```bash
   cd gpu-advisor
   pip install -r requirements.txt
   ```

2. **데이터 수집 체험**
   ```bash
   python3 crawlers/run_daily.py
   ```

3. **백엔드 서버 가동**
   ```bash
   cd backend
   python3 simple_server.py
   ```

4. **프론트엔드 웹 실행**
   ```bash
   cd frontend
   npm install  # 첫 실행시
   npm run dev
   ```

이제 브라우저에서 `http://localhost:3000`에 접속하여 여러분이 만든 AI GPU 어드바이저를 경험해보세요!
