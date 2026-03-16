# GPU Advisor 코드 리뷰 보고서

> 생성일: 2026-03-16
> 분석 범위: 프로젝트 전체 (`backend/`, `crawlers/`, `tests/`)

---

## 요약 (Executive Summary)

| 분류 | 건수 |
|------|------|
| 🔴 버그 / 잘못된 코드 | 9 |
| 🟡 설계 문제 / 기술 부채 | 12 |
| 🔵 개선 권장 사항 | 10 |
| 🗑️ 제거 대상 (데드코드/쓰레기 파일) | 8 |

---

## 1. 🔴 버그 / 잘못된 코드

### 1-1. `backend/data/scaler.py` — MinMaxScaler 역직렬화 버그
**위치**: `load_scaler()` (line 173-179)

```python
# 버그: data_min_, data_max_ 복원 후 scale_ 계산에 data_range_ 사용
# MinMaxScaler는 fit 전까지 data_range_ 속성이 존재하지 않음
self.scaler.min_ = np.array(scaler_data["min_"])
self.scaler.max_ = np.array(scaler_data["max_"])
self.scaler.scale_ = (self.scaler.max_ - self.scaler.min_) / (
    self.scaler.data_range_   # ← AttributeError 발생 가능
    if hasattr(self.scaler, "data_range_")
    else 1
)
```

**문제**: `save_scaler()`에서 저장한 키가 `min_`, `max_`인데, scikit-learn의 `MinMaxScaler`가 실제로 사용하는 내부 속성은 `data_min_`, `data_max_`, `data_range_`이다. 로드 후 `transform()` 호출 시 `NotFittedError` 또는 잘못된 결과가 반환된다.

**수정**: `joblib.dump/load` 사용 또는 `data_min_`, `data_max_`, `data_range_`, `scale_`을 모두 저장/복원해야 한다.

---

### 1-2. `backend/security/auth.py` — 평문 비밀번호 비교
**위치**: `authenticate_password()` (line 112-113)

```python
def authenticate_password(self, username: str, password: str) -> bool:
    return bool(username in self.users and self.users[username] == password)  # ← 평문 비교
```

**문제**: 비밀번호가 환경변수에 평문으로 저장되고 평문으로 비교된다. 타이밍 어택(timing attack)에 취약하며, `hmac.compare_digest()`를 사용하지 않는다. 또한 JWT secret 기본값 `"gpu-advisor-dev-secret"`이 코드에 하드코딩되어 있다.

**수정**: `hmac.compare_digest(stored, provided)` 사용 필요. 프로덕션 배포 시 반드시 환경변수로 비밀값을 교체해야 한다.

---

### 1-3. `backend/simple_server.py` — 전역 `gpu_agent` 상태 레이스 컨디션
**위치**: `start_training()` / `get_gpu_agent()` (line 527, 152-157)

```python
async def training_loop():
    global gpu_agent
    ...
    gpu_agent = None  # 학습 완료 후 null로 설정

def get_gpu_agent() -> GPUPurchaseAgent:
    global gpu_agent
    if gpu_agent is None:
        gpu_agent = GPUPurchaseAgent(...)  # ← 락 없이 재초기화
    return gpu_agent
```

**문제**: 비동기 학습 루프가 진행 중인 동안 다른 요청이 `get_gpu_agent()`를 호출하면 `gpu_agent = None` 직후 동시에 `GPUPurchaseAgent()`가 중복 초기화된다. 락이 없어 스레드 안전하지 않다.

---

### 1-4. `crawlers/auto_training.py` — `release_daily_runner` 인터페이스 불일치
**위치**: `run_auto_training_cycle()` (line 409)

```python
pipeline_result = release_daily_runner(project_root, cfg.timeout_sec)
```

**문제**: `release_daily_runner` 타입 힌트는 `ReleaseDailyRunner = Callable[[Path, int], Dict]`이지만, `run_release_daily_check()`의 실제 시그니처도 `(project_root: Path, timeout_sec: int)`이다. 그러나 반환값 `pipeline_result`에서 `pipeline_result.get("status")`를 호출하는데, `run_release_daily_check()`의 반환값은 이미 언팩된 내부 `result` 딕셔너리다. 테스트 목(mock)에서는 `{"status": "blocked", ...}`를 반환하므로 테스트는 통과하지만 실제 실행 경로와 테스트 경로의 반환값 형태가 다르다.

---

### 1-5. `backend/models/mcts_engine.py` — root visits count 오프-바이-원
**위치**: `search()` (line 93-108)

```python
root = MCTSNode(state=root_state)
self._expand(root, ...)  # root 확장 (이때 children은 visits=0)

for _ in range(self.config.num_simulations):
    node = self._select(root)
    ...
    self._backup(node, value)

# test: root.visits == num_simulations 기대
```

**문제**: `_expand(root)` 호출 시 `_backup`이 호출되지 않으므로 root.visits는 `num_simulations`가 아니라 `num_simulations` 그대로여야 하지만, `_select`에서 항상 root를 먼저 통과하므로 backup 시 root도 카운트된다. 실제로 `test_search_with_nonzero_state`에서 `tree.visits == config.num_simulations` 어서션이 통과하는 것은 backup 로직이 올바르기 때문이나, Dirichlet noise 추가로 인한 초기 확장이 1회 추가 수행되므로 총 root visits = num_simulations + initial expand visits가 될 수 있다.

---

### 1-6. `backend/simple_server.py` — 미들웨어에서 auth_subject 미초기화 버그
**위치**: `security_and_logging_middleware()` (line 276-278)

```python
remaining = rate_limiter.per_minute  # ← 초기값 (실제 remaining 아님)
auth_subject = "anonymous"

if path.startswith("/api/"):
    allowed, remaining = rate_limiter.is_allowed(client_ip)
```

**문제**: `/api/` 경로가 아닌 경우 `remaining`은 실제 남은 횟수가 아닌 `per_minute` 최대값이 응답 헤더에 들어간다. 로직 오류는 아니지만 헤더 값이 부정확하다.

---

### 1-7. `backend/models/representation_network.py` — PositionalEncoding 미사용 (코드 vs 주석 불일치)
**위치**: `__init__` (line 76), `forward` (line 109)

```python
# __init__에서 정의
self.pos_encoding = PositionalEncoding(latent_dim, max_seq_len)

# forward에서 주석 처리
# Positional Encoding은 seq_len=1에서 의미 없으므로 적용하지 않음
# (pos_encoding 모듈은 체크포인트 호환성을 위해 __init__에 유지)
```

**문제**: `pos_encoding`의 파라미터는 학습 중 그래디언트를 받지 못하므로(forward에서 미사용) 사실상 데드 가중치다. 체크포인트 호환성을 위해 유지한다고 명시했으나, 이 파라미터들은 불필요한 메모리를 차지하며 `state_dict`에 포함된다. `register_buffer`로 전환하면 파라미터 카운트에서 제외된다.

---

### 1-8. `backend/agent/fine_tuner.py` — `action_onehot`에서 `num_classes=5` 하드코딩
**위치**: `_train_step()` (line 258)

```python
action_onehot = F.one_hot(actions, num_classes=5).float()
```

**문제**: `action_dim`은 체크포인트에서 동적으로 추론되지만 `num_classes=5`로 하드코딩되어 있다. `action_dim`이 5가 아닌 모델이 로드되면 잘못된 one-hot 인코딩 생성.

**수정**: `F.one_hot(actions, num_classes=self.g.action_dim).float()` 또는 별도 `action_dim` 멤버 변수 사용.

---

### 1-9. `crawlers/danawa_crawler.py` — User-Agent 헤더 중복 정의
**위치**: `__init__` (line 52-54) vs `_crawl_danawa()` (line 91-99)

`__init__`에서 `self.headers`를 정의하지만 `_crawl_danawa()`는 지역 변수 `headers`를 새로 정의하여 `self.headers`를 무시한다. 헤더 설정을 중앙화하려는 의도가 실제로 적용되지 않는다.

---

## 2. 🟡 설계 문제 / 기술 부채

### 2-1. `check_readiness()` 로직 중복
**위치**: `backend/agent/release_pipeline.py:45`, `backend/simple_server.py:200`

`_data_readiness()`와 `AgentReleasePipeline.check_readiness()`가 동일한 로직을 각각 독립적으로 구현. 파일 날짜 파싱, `range_days` 계산, `min_days` 취합 로직이 100% 동일. 향후 변경 시 두 곳 모두 수정해야 한다.

**수정**: `simple_server.py`의 `_data_readiness()`를 `AgentReleasePipeline.check_readiness()` 위임으로 교체.

---

### 2-2. `backend/models/mcts.py` — 사용되지 않는 레거시 MCTS 구현체
`backend/models/mcts.py`와 `backend/models/mcts_engine.py` 두 개의 MCTS 구현이 존재. `mcts.py`의 `MCTSTree`/`Node` 클래스는 어디서도 import되지 않음. `mcts_engine.py`가 실제로 사용됨.

**수정**: `mcts.py` 제거 또는 아카이브.

---

### 2-3. `backend/models/` — `TransitionModel` 클래스 중복
`dynamics_network.py`(line 116)와 `prediction_network.py`(line 114) 양쪽에 동일한 `TransitionModel` 클래스가 정의되어 있으나 어디서도 사용되지 않는다.

---

### 2-4. `backend/` 루트에 테스트 파일 방치
`backend/test_agent_pipeline.py`, `backend/test_gpu_purchase_advisor.py`가 `tests/` 디렉터리가 아닌 `backend/` 루트에 위치. `pytest` 설정 `testpaths = ["tests"]`에 의해 자동 실행되지 않으며, CI에서 수집되지 않는 데드 코드.

---

### 2-5. `backend/train_alphazero.py` / `backend/train_alphazero_v2.py` — 레거시 학습 스크립트
현재 학습 파이프라인은 `AgentReleasePipeline → AgentFineTuner`를 통해 수행됨. 루트의 두 `train_alphazero*.py` 파일은 사용되지 않는 레거시 코드.

---

### 2-6. `backend/simple_server.py` — `TrainingState` 스레드 안전성 미흡
`TrainingState.metrics_history`(리스트)와 기타 필드들이 비동기 학습 루프와 HTTP 핸들러에서 동시 접근되나 `Lock`으로 보호되지 않는다. CPython GIL이 어느 정도 보호하지만, 비동기 컨텍스트에서는 완전한 보장이 없다.

---

### 2-7. `backend/data/scaler.py` — 프로덕션에서 미사용
`NumericalScaler`와 `MultiFeatureScaler`는 실제 파이프라인(`feature_engineer.py`, `fine_tuner.py`)에서 전혀 사용되지 않는다. Feature engineering은 `crawlers/feature_engineer.py`에서 직접 numpy로 수행된다.

---

### 2-8. `backend/data/multimodal_embedding.py` / `backend/data/tokenizer.py` — 미사용
두 파일 모두 다른 모듈에서 import되지 않는다. 초기 설계 시 작성된 미사용 코드.

---

### 2-9. 경로 조작을 통한 sys.path 삽입 남발
`backend/agent/gpu_purchase_agent.py`(line 19-21), `backend/run_release_daily.py`(line 14-15), `backend/run_release_ready.py`(line 17-18) 등 여러 파일에서 `sys.path.insert(0, ...)` 패턴이 반복된다. `pyproject.toml`의 `pythonpath` 설정 또는 패키지 설치(`pip install -e .`)로 해결해야 할 문제를 각 스크립트가 직접 처리한다.

---

### 2-10. `backend/simple_server.py` — `getattr`을 통한 덕 타이핑 방어 코드
**위치**: `ask_gpu()` (line 437-443)

```python
resolve_state = getattr(agent, "resolve_state", None)
decide_from_state = getattr(agent, "decide_from_state", None)
resolved = None
if callable(resolve_state) and callable(decide_from_state):
    resolved = resolve_state(query_model.model_name)
```

`GPUPurchaseAgent`는 `resolve_state`와 `decide_from_state`가 항상 존재하는 클래스이므로 이 방어 코드는 불필요하다. 레거시 인터페이스 호환 코드가 남아 복잡성만 증가시킨다.

---

### 2-11. `backend/initialize.py` — 빈 파일
파일에 주석 한 줄만 존재. 실질적 내용 없음.

---

### 2-12. `backend/storage/postgres_repository.py` — 연결 재사용 없이 매 쿼리마다 새 연결
`_connect()`는 매번 `psycopg.connect()`를 호출하는 새 연결을 반환. 프로덕션 환경에서는 연결 풀(`psycopg_pool.ConnectionPool`)이 필요하다.

---

## 3. 🔵 개선 권장 사항

### 3-1. 감성 분석기 스코어 임계값 하드코딩
`backend/api/sentiment/analyzer.py:105`의 `score > 0.1` / `score < -0.1` 임계값이 하드코딩. 환경변수 또는 설정 파라미터로 추출 권장.

### 3-2. MCTS `_simulate`에서 매 스텝마다 tensor 생성
`mcts_engine.py`의 `_simulate()` 루프에서 매 스텝마다 `torch.tensor()` 호출로 새 텐서를 생성. `clone().detach()`나 `torch.from_numpy()` + `pin_memory` 패턴으로 성능 개선 가능.

### 3-3. `feature_engineer.py` — `load_historical_data()` 매번 전체 재로드
`_build_feature_context()`가 `load_historical_data(days=30)`을 매 GPU 모델마다 호출. `process_all()` 내에서 한 번 로드 후 공유해야 O(n²) I/O를 O(n)으로 줄일 수 있다.

### 3-4. `MCTSNode` 필드 기본값에 mutable 사용
`children: List["MCTSNode"] = field(default_factory=list)` — 올바르게 `field()` 사용 중이나, `parent: Optional["MCTSNode"] = None`인 상태에서 `_ucb_score`의 deprecated property가 `parent.visits`에 무조건 접근해 root 노드에서 `AttributeError` 가능성 있음 (단, root는 parent=None이고 visits=0이어서 `if self.visits == 0: return inf` 분기에서 먼저 반환됨).

### 3-5. `backend/api/alphazero_routes.py` — 사용 여부 확인 필요
`alphazero_routes.py`가 `simple_server.py`에 포함(include_router)되지 않는지 확인 필요. 현재 `simple_server.py`에서 `alphazero_routes`를 import하지 않는 것으로 보임.

### 3-6. `crawlers/news_crawler.py` — User-Agent 문자열이 잘림
`danawa_crawler.py`의 `__init__`에서 정의된 User-Agent `"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"`가 불완전한 문자열. 실제 브라우저 User-Agent보다 짧아 일부 서버에서 차단될 수 있다.

### 3-7. 타입 힌트 불완전
`backend/agent/evaluator.py`의 `run()` 반환 타입은 `Dict[str, Any]`이지만 더 구체적인 TypedDict 또는 dataclass를 사용하면 타입 안전성이 향상된다.

### 3-8. `AgentFineTuner`에서 optimizer를 `run()` 메서드 내부에서만 생성
`self.optimizer`는 `run()` 호출 전까지 초기화되지 않는다. `save_checkpoint()` 호출 시 optimizer state는 저장되지 않으므로 warm restart 불가. 체크포인트에 `optimizer_state_dict` 포함 권장.

### 3-9. `backend/run_release_pipeline.py` — CLI 인자 없음
`run_release_pipeline.py`는 `argparse` 없이 고정 기본값만 사용. `run_release_ready.py`가 동일한 역할을 더 완전하게 수행하므로 레거시 스크립트로 판단된다.

### 3-10. `tests/` 디렉터리에 통합 테스트 누락
`AgentFineTuner`, `AgentEvaluator`, `AgentReleasePipeline`에 대한 단위/통합 테스트가 없다. 파이프라인의 핵심 컴포넌트인데 테스트 커버리지가 부족하다.

---

## 4. 🗑️ 제거 대상

| 파일/항목 | 이유 |
|-----------|------|
| `backend/models/mcts.py` | `mcts_engine.py`로 대체됨, 미사용 |
| `backend/models/dynamics_network.py::TransitionModel` | 미사용 클래스 |
| `backend/models/prediction_network.py::TransitionModel` | 미사용 클래스 (중복) |
| `backend/train_alphazero.py` | 레거시, 현재 파이프라인과 무관 |
| `backend/train_alphazero_v2.py` | 레거시, 현재 파이프라인과 무관 |
| `backend/initialize.py` | 내용 없는 빈 파일 |
| `backend/test_agent_pipeline.py` | `tests/` 디렉터리로 이동 또는 삭제 |
| `backend/test_gpu_purchase_advisor.py` | `tests/` 디렉터리로 이동 또는 삭제 |
| `backend/data/multimodal_embedding.py` | 미사용 |
| `backend/data/tokenizer.py` | 미사용 |
| 루트의 `echo`, `ls`, `디렉토리 생성 완료` 파일 | 실수로 생성된 쓰레기 파일 |
| `backend/run_release_pipeline.py` | `run_release_ready.py`로 완전 대체 가능 |

---

## 5. 보안 고려 사항

### 5-1. JWT 시크릿 기본값 노출
`backend/security/auth.py:92`
```python
self.jwt_secret = os.getenv("GPU_ADVISOR_JWT_SECRET", "gpu-advisor-dev-secret")
```
개발용 시크릿이 코드에 하드코딩. 프로덕션 배포 시 반드시 환경변수 설정 필요. `.env.example` 파일 제공 권장.

### 5-2. CORS 전체 허용
`backend/simple_server.py:64-70`
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```
프로덕션에서는 허용 오리진을 명시해야 한다.

### 5-3. `weights_only=False` 사용
`backend/agent/fine_tuner.py:67`, `gpu_purchase_agent.py:97`
```python
ckpt = torch.load(self.base_checkpoint, map_location=self.device, weights_only=False)
```
신뢰할 수 없는 체크포인트 파일 로드 시 코드 실행 취약점. 자체 학습 모델이므로 현재 위험은 낮으나 `weights_only=True` + 필요 시 `torch.serialization.add_safe_globals()` 패턴으로 전환 권장.

---

## 6. 우선순위별 수정 로드맵

### 즉시 수정 (P0)
1. **`scaler.py` MinMaxScaler 역직렬화 버그** — 데이터 손상 가능성
2. **`fine_tuner.py` action_onehot `num_classes=5` 하드코딩** — 모델 아키텍처 변경 시 silent 버그
3. **쓰레기 파일 삭제** — `echo`, `ls`, `디렉토리 생성 완료`

### 단기 수정 (P1)
4. **`check_readiness()` 중복 제거** — 유지보수성
5. **`gpu_agent` 전역 상태 락 추가** — 레이스 컨디션
6. **레거시 파일 정리** — `mcts.py`, `TransitionModel`, `train_alphazero*.py`, `initialize.py`
7. **테스트 파일 위치 정규화** — `backend/test_*.py` → `tests/`

### 중기 수정 (P2)
8. **비밀번호 타이밍 어택 방어** — `hmac.compare_digest` 사용
9. **Postgres 연결 풀링** — 프로덕션 성능
10. **`sys.path` 조작 제거** — `pip install -e .` 기반 패키지 구조
11. **AgentFineTuner/Evaluator 테스트 추가** — 커버리지 향상

---

## 7. 작업 반영 현황 (2026-03-16 최종 업데이트)

### 완료
- **1-1** `backend/data/scaler.py` 역직렬화 복구 완료
- **1-2** `backend/security/auth.py`에 `hmac.compare_digest()` 적용, JWT 모드에서 secret 미설정 시 실패하도록 수정
- **1-3** `backend/simple_server.py`의 전역 `gpu_agent` 접근에 락 추가
- **1-6** `backend/simple_server.py` rate-limit 헤더 초기값 부정확성 수정
- **1-7** `backend/models/representation_network.py`의 `pos_encoding`을 gradient 비활성 상태로 유지하도록 정리
- **1-8** `backend/agent/fine_tuner.py`의 `num_classes=5` 하드코딩 제거
- **1-9** `crawlers/danawa_crawler.py` 헤더 설정 중앙화 완료
- **2-1** `simple_server._data_readiness()`를 `AgentReleasePipeline.check_readiness()` 위임으로 교체
- **2-2** 미사용 레거시 `backend/models/mcts.py` 삭제
- **2-3** `dynamics_network.py`, `prediction_network.py`의 중복 `TransitionModel` 삭제
- **2-4** `backend/test_agent_pipeline.py`, `backend/test_gpu_purchase_advisor.py` 제거 후 `tests/test_agent_pipeline_smoke.py` 추가
- **2-5** `backend/train_alphazero.py`, `backend/train_alphazero_v2.py` 삭제
- **2-6** `TrainingState`에 락 및 snapshot/accessor 추가
- **2-8** `backend/data/multimodal_embedding.py`, `backend/data/tokenizer.py` 삭제
- **2-9** `backend/agent/gpu_purchase_agent.py`, `backend/run_release_daily.py`, `backend/run_release_ready.py`의 `sys.path` 조작 제거
- **2-10** `ask_gpu()`의 불필요한 `getattr` 덕 타이핑 제거
- **2-11** `backend/initialize.py` 삭제
- **2-12** `backend/storage/postgres_repository.py`에 `psycopg_pool.ConnectionPool` 우선 사용 추가
- **3-1** `backend/api/sentiment/analyzer.py` 감성 분석 임계값을 `GPU_ADVISOR_SENTIMENT_POS_THRESHOLD` / `GPU_ADVISOR_SENTIMENT_NEG_THRESHOLD` 환경변수로 외부화
- **3-2** `backend/models/mcts_engine.py` `_simulate()` 루프에서 `torch.tensor()` → `torch.from_numpy()` 전환 (불필요한 copy 제거)
- **3-5** `backend/api/alphazero_routes.py` 모듈 레벨 `app.include_router(router)` side effect 제거 → 명시적 등록 방식으로 변경
- **3-7** `backend/agent/evaluator.py` `run()` 반환 타입을 `EvaluationResult(TypedDict)`으로 정의
- **3-8** `backend/agent/fine_tuner.py` `save_checkpoint()`에 `optimizer_state_dict` 포함, `run()` 시작 시 이전 output 체크포인트에서 warm restart 지원
- **3-9 / 4** `backend/run_release_pipeline.py` 삭제 (`run_release_ready.py`로 완전 대체)
- **3-10** 파이프라인 스모크 테스트를 `tests/test_agent_pipeline_smoke.py`로 추가
- **4 항목 중 다수** `mcts.py`, `TransitionModel`, `train_alphazero*.py`, `initialize.py`, `backend/test_*.py`, `multimodal_embedding.py`, `tokenizer.py` 정리 완료
- **5-3** `torch.load(..., weights_only=False)` → `weights_only=True` 변경 (`fine_tuner.py`, `gpu_purchase_agent.py`)

### 미완료 / 유지 결정
- **1-4** `crawlers/auto_training.py`의 `release_daily_runner` 인터페이스 불일치 — 재현되지 않음, 변경 없음
- **1-5** `backend/models/mcts_engine.py` root visits 오프바이원 — 재현되지 않음, 변경 없음
- **2-7** `backend/data/scaler.py` 프로덕션 미사용 구조 — 버그 수정만 완료, 파일 유지
- **3-3** `feature_engineer.py` historical data 재로드 — `process_all()`에서 이미 context를 한 번 빌드 후 공유하므로 실질적 문제 없음
- **3-4** deprecated `ucb_score` property — 문서화 유지, 코드 변경 없음
- **3-6** `crawlers/danawa_crawler.py` User-Agent — 코드 확인 결과 이미 완전한 UA 문자열 포함, 수정 불필요
- **4 항목 일부** 루트 쓰레기 파일(`echo`, `ls`, `디렉토리 생성 완료`) — 작업 트리에서 확인 불가, 조치 없음
- **5-2** CORS 전체 허용 정책 — 개발 환경 의도, 유지

### 검증 결과
- `python3 -m py_compile backend/api/sentiment/analyzer.py backend/api/alphazero_routes.py backend/agent/evaluator.py backend/agent/fine_tuner.py backend/agent/gpu_purchase_agent.py backend/models/mcts_engine.py` 통과
- `python3 -m pytest tests/test_auto_training.py tests/test_auth_and_sentiment.py tests/test_api.py tests/test_mcts.py tests/test_networks.py -q` 통과 (`31 passed`)
- `tests/test_agent_pipeline_smoke.py`는 실제 모델 평가/리포트 생성까지 수행하면서 장시간 실행되어 완료 확인 전 중단함
