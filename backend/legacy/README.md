# Legacy Modules

> **2026-05-13 리팩토링 시 이동됨**

이 디렉토리의 파일은 초기 합성 벤치마크용으로 사용되었으나,
현재 운영 경로(`backend/simple_server.py` + `backend/agent/*`)에서는 사용되지 않습니다.

## 포함 파일

| 파일 | 원래 경로 | 설명 |
|-----|----------|------|
| `main.py` | `backend/main.py` | 합성 환경 벤치마크 진입점 |
| `routes.py` | `backend/api/routes.py` | 레거시 FastAPI 라우트 |
| `training.py` | `backend/api/training.py` | 레거시 학습 루프 |
| `alphazero_routes.py` | `backend/api/alphazero_routes.py` | 레거시 AlphaZero 라우트 |
| `gym_env.py` | `backend/environment/gym_env.py` | 합성 마켓 환경 |
| `run_server.py` | `backend/run_server.py` | 레거시 서버 런처 |

삭제해도 운영에 영향 없습니다.
