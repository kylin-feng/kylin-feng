#!/bin/bash

echo "🛑 停止幸运色生成器服务"
echo "========================"

# 停止后端服务
echo "🛑 停止后端服务..."
pkill -f "python3 backend.py" || true
if [ -f backend.pid ]; then
    kill $(cat backend.pid) 2>/dev/null || true
    rm -f backend.pid
fi

# 停止前端服务
echo "🛑 停止前端服务..."
pkill -f "python3 -m http.server 8080" || true
if [ -f frontend.pid ]; then
    kill $(cat frontend.pid) 2>/dev/null || true
    rm -f frontend.pid
fi

# 检查端口是否已释放
echo "🔍 检查端口状态..."
sleep 2

if ! lsof -i :5001 &> /dev/null; then
    echo "✅ 端口5001已释放"
else
    echo "⚠️  端口5001仍被占用"
fi

if ! lsof -i :8080 &> /dev/null; then
    echo "✅ 端口8080已释放"
else
    echo "⚠️  端口8080仍被占用"
fi

echo ""
echo "✅ 所有服务已停止" 