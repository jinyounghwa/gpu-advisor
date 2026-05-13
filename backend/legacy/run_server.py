import sys
import os
import uvicorn

# 현재 디렉토리(backend)를 시스템 경로에 추가하여 모듈 인식 문제 해결
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Starting GPU Advisor Backend Server...")
    print("   - API: http://localhost:8000/api")
    print("   - Swagger UI: http://localhost:8000/docs")
    uvicorn.run("simple_server:app", host="0.0.0.0", port=8000, reload=True)
