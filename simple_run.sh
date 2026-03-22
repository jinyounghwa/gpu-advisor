#!/bin/bash

# 1. 기존 프로세스 강제 종료 (포트 8000, 3000 점유 해제)
echo "🧹 Cleaning up ports..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

echo "🚀 Starting GPU Advisor System (Simple Version)..."

# 2. 백엔드 실행 (Simple Server)
echo "🔹 Launching Simple Backend..."
cd backend
# uvicorn 실행 (백그라운드)
python3 simple_server.py > backend_simple.log 2>&1 &
BACKEND_PID=$!
echo "   ✅ Backend running (PID: $BACKEND_PID) at http://localhost:8000"

# 3. 프론트엔드 실행
echo "🔹 Launching Frontend..."
cd ../frontend
# Next.js 실행 (백그라운드)
npm run dev > frontend_simple.log 2>&1 &
FRONTEND_PID=$!
echo "   ✅ Frontend running (PID: $FRONTEND_PID) at http://localhost:3000"

echo ""
echo "✨ System Ready!"
echo "👉 Open: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop."

# 종료 시 프로세스 정리
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT

wait
