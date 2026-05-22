#!/bin/bash

echo "🎯 幸运色生成器 - 功能演示"
echo "=========================="
echo ""

# 检查服务状态
echo "📊 检查服务状态..."
if ! ./status.sh &> /dev/null; then
    echo "❌ 服务状态检查失败，请先启动服务"
    exit 1
fi

echo "✅ 所有服务正常运行"
echo ""

# 演示1: 基础API测试
echo "🎨 演示1: 基础API测试"
echo "----------------------"
echo "生成幸运色并发送到硬件设备..."

RESPONSE=$(curl -s http://localhost:5001/api/lucky-color)
echo "API响应: $RESPONSE"

# 解析响应
COLOR=$(echo $RESPONSE | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('description', '未知'))")
HARDWARE_SENT=$(echo $RESPONSE | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('hardware_sent', False))")

echo "幸运色: $COLOR"
if [ "$HARDWARE_SENT" = "True" ]; then
    echo "✅ 硬件发送: 成功"
else
    echo "❌ 硬件发送: 失败"
fi

echo ""
sleep 2

# 演示2: 专用硬件发送API
echo "🎯 演示2: 专用硬件发送API"
echo "-------------------------"
echo "使用专用API发送幸运色到硬件设备..."

RESPONSE2=$(curl -s -X POST http://localhost:5001/api/send-to-hardware)
echo "API响应: $RESPONSE2"

# 解析响应
MESSAGE=$(echo $RESPONSE2 | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('message', '未知'))")
HARDWARE_SENT2=$(echo $RESPONSE2 | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('hardware_sent', False))")

echo "消息: $MESSAGE"
if [ "$HARDWARE_SENT2" = "True" ]; then
    echo "✅ 硬件发送: 成功"
else
    echo "❌ 硬件发送: 失败"
fi

echo ""
sleep 2

# 演示3: 前端页面测试
echo "🌐 演示3: 前端页面测试"
echo "----------------------"
echo "检查前端页面可访问性..."

if curl -s http://localhost:8080/index.html &> /dev/null; then
    echo "✅ 主页可访问"
else
    echo "❌ 主页无法访问"
fi

if curl -s http://localhost:8080/color-generator.html &> /dev/null; then
    echo "✅ 颜色生成页面可访问"
else
    echo "❌ 颜色生成页面无法访问"
fi

if curl -s http://localhost:8080/test_complete.html &> /dev/null; then
    echo "✅ 测试页面可访问"
else
    echo "❌ 测试页面无法访问"
fi

echo ""
sleep 2

# 演示4: 硬件连接测试
echo "🔌 演示4: 硬件连接测试"
echo "----------------------"
echo "测试硬件设备连接..."

if python3 test_hardware.py &> /dev/null; then
    echo "✅ 硬件设备连接正常"
    echo "✅ UDP发送功能正常"
    echo "✅ 控制命令发送正常"
    echo "✅ 数据消息发送正常"
else
    echo "❌ 硬件设备连接异常"
fi

echo ""
sleep 2

# 总结
echo "🎉 演示完成！"
echo "============="
echo ""
echo "📱 您现在可以:"
echo "  1. 访问 http://localhost:8080/color-generator.html"
echo "  2. 点击'生成幸运色'按钮"
echo "  3. 系统自动发送到硬件设备"
echo "  4. 或点击'发送到硬件'按钮手动发送"
echo ""
echo "🧪 测试功能:"
echo "  访问 http://localhost:8080/test_complete.html"
echo "  进行完整的功能测试"
echo ""
echo "📊 监控状态:"
echo "  运行 ./status.sh 查看服务状态"
echo ""
echo "🛑 停止服务:"
echo "  运行 ./stop_all.sh 停止所有服务" 