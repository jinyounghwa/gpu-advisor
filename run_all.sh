#!/bin/bash

# Kill ports
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

echo "🚀 Starting AlphaZero Trading System..."

# Start Backend
echo "🔹 Launching Backend Server..."
cd backend
python3 run_server.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "   ✅ Backend running (PID: $BACKEND_PID)"

# Start Frontend
echo "🔹 Launching Frontend Dashboard..."
cd ../frontend
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   ✅ Frontend running (PID: $FRONTEND_PID)"

echo ""
echo "✨ System is ready!"
echo "👉 Dashboard: http://localhost:3000/alphazero"
echo "👉 API Docs:  http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers."

# Trap Ctrl+C to kill processes
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT

wait
