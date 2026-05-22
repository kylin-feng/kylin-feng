#!/bin/bash

echo "Starting SmartMeet AI..."

# Kill any existing processes on our ports
echo "Cleaning up existing processes..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:5001 | xargs kill -9 2>/dev/null || true

# Start backend
echo "Starting backend server..."
cd /Users/shixianping/smart-meet/backend
npm run dev &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "Starting frontend server..."
cd /Users/shixianping/smart-meet/frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "🚀 SmartMeet AI is starting..."
echo "📊 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user to stop
wait