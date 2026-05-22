#!/bin/bash

echo "📊 幸运色生成器服务状态"
echo "========================"

# 检查后端服务
echo "🔍 检查后端服务..."
if lsof -i :5001 &> /dev/null; then
    echo "✅ 后端服务运行中 (端口5001)"
    if curl -s http://localhost:5001/health &> /dev/null; then
        echo "✅ 后端API响应正常"
    else
        echo "⚠️  后端API无响应"
    fi
else
    echo "❌ 后端服务未运行"
fi

echo ""

# 检查前端服务
echo "🔍 检查前端服务..."
if lsof -i :8080 &> /dev/null; then
    echo "✅ 前端服务运行中 (端口8080)"
    if curl -s http://localhost:8080/index.html &> /dev/null; then
        echo "✅ 前端页面可访问"
    else
        echo "⚠️  前端页面无法访问"
    fi
else
    echo "❌ 前端服务未运行"
fi

echo ""

# 检查硬件连接
echo "🔍 检查硬件设备连接..."
HARDWARE_IP=$(python3 -c "import config; print(config.HARDWARE_IP)" 2>/dev/null || echo "未知")
echo "硬件设备IP: $HARDWARE_IP"

# 测试硬件连接
if python3 test_hardware.py &> /dev/null; then
    echo "✅ 硬件设备连接正常"
else
    echo "⚠️  硬件设备连接异常"
fi

echo ""

# 显示访问地址
echo "📱 访问地址:"
echo "  主页: http://localhost:8080/index.html"
echo "  颜色生成: http://localhost:8080/color-generator.html"
echo "  测试页面: http://localhost:8080/test_complete.html"
echo ""

echo "🔧 API接口:"
echo "  健康检查: http://localhost:5001/health"
echo "  幸运色生成: http://localhost:5001/api/lucky-color"
echo "  硬件发送: http://localhost:5001/api/send-to-hardware" 