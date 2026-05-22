#!/bin/bash

echo "=== 启动之江智慧会议系统 ==="

# 临时禁用代理环境变量
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

cd "$(dirname "$0")/backend"

echo "停止现有服务..."
pkill -f "python3 main.py" 2>/dev/null || true
pkill -f "python3 main_simple.py" 2>/dev/null || true

sleep 2

echo "启动后端服务..."
python3 main.py &
BACKEND_PID=$!

sleep 5

echo "测试后端连接..."
if curl -s http://127.0.0.1:8000/ > /dev/null; then
    echo "✅ 后端服务启动成功"
else
    echo "❌ 后端服务启动失败"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎉 系统启动完成！"
echo ""
echo "访问地址："
echo "  前端: http://localhost:11130"
echo "  后端: http://127.0.0.1:8000"
echo "  API文档: http://127.0.0.1:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务"

# 等待用户中断
trap 'echo "正在停止服务..."; kill $BACKEND_PID 2>/dev/null; exit 0' INT

wait