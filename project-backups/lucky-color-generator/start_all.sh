#!/bin/bash

echo "🎯 幸运色生成器 - 一键启动脚本"
echo "=================================="

# 检查Python3是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3，请先安装 Python 3"
    exit 1
fi

# 检查Flask是否安装
if ! python3 -c "import flask" &> /dev/null; then
    echo "❌ 错误: 未找到 Flask，请先安装: pip3 install flask"
    exit 1
fi

echo "✅ 环境检查通过"
echo ""

# 显示配置信息
echo "📋 当前配置:"
echo "后端服务: http://localhost:5001"
echo "前端服务: http://localhost:8080"
echo "硬件设备: $(python3 -c "import config; print(config.HARDWARE_IP)")"
echo ""

# 检查端口是否被占用
echo "🔍 检查端口状态..."
if lsof -i :5001 &> /dev/null; then
    echo "⚠️  端口5001已被占用，正在停止现有服务..."
    pkill -f "python3 backend.py" || true
    sleep 2
fi

if lsof -i :8080 &> /dev/null; then
    echo "⚠️  端口8080已被占用，正在停止现有服务..."
    pkill -f "python3 -m http.server 8080" || true
    sleep 2
fi

echo "✅ 端口检查完成"
echo ""

# 启动后端服务
echo "🚀 启动后端服务..."
python3 backend.py &
BACKEND_PID=$!
echo "✅ 后端服务已启动 (PID: $BACKEND_PID)"

# 等待后端服务启动
echo "⏳ 等待后端服务启动..."
sleep 3

# 测试后端服务
if curl -s http://localhost:5001/health &> /dev/null; then
    echo "✅ 后端服务启动成功"
else
    echo "❌ 后端服务启动失败"
    exit 1
fi

# 启动前端服务
echo "🚀 启动前端服务..."
python3 -m http.server 8080 &
FRONTEND_PID=$!
echo "✅ 前端服务已启动 (PID: $FRONTEND_PID)"

# 等待前端服务启动
sleep 2

# 测试前端服务
if curl -s http://localhost:8080/index.html &> /dev/null; then
    echo "✅ 前端服务启动成功"
else
    echo "❌ 前端服务启动失败"
    exit 1
fi

echo ""
echo "🎉 所有服务启动成功！"
echo ""
echo "📱 访问地址:"
echo "  主页: http://localhost:8080/index.html"
echo "  颜色生成: http://localhost:8080/color-generator.html"
echo "  测试页面: http://localhost:8080/test_complete.html"
echo ""
echo "🔧 API接口:"
echo "  健康检查: http://localhost:5001/health"
echo "  幸运色生成: http://localhost:5001/api/lucky-color"
echo "  硬件发送: http://localhost:5001/api/send-to-hardware"
echo ""
echo "💡 使用说明:"
echo "  1. 访问颜色生成页面"
echo "  2. 点击'生成幸运色'按钮"
echo "  3. 系统自动发送到硬件设备"
echo "  4. 或点击'发送到硬件'按钮手动发送"
echo ""
echo "⏹️  停止服务: 按 Ctrl+C"

# 保存PID到文件
echo $BACKEND_PID > backend.pid
echo $FRONTEND_PID > frontend.pid

# 等待用户中断
trap 'echo ""; echo "🛑 正在停止服务..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; rm -f backend.pid frontend.pid; echo "✅ 服务已停止"; exit 0' INT

# 保持脚本运行
while true; do
    sleep 1
done 