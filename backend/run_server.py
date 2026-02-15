import sys
import os
import uvicorn

# 현재 디렉토리(backend)를 시스템 경로에 추가하여 모듈 인식 문제 해결
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Starting AlphaZero Backend Server...")
    print("   - API: http://localhost:8000/api/alphazero")
    print("   - Swagger UI: http://localhost:8000/docs")

    # 모듈 경로를 명시적으로 지정하여 실행
    uvicorn.run("api.alphazero_routes:app", host="0.0.0.0", port=8000, reload=True)
