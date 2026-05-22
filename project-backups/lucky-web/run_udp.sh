#!/bin/bash

echo "🎯 UDP发送器启动脚本"
echo "===================="

# 检查Python3是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3，请先安装 Python 3"
    exit 1
fi

echo "✅ Python3 已安装"

# 检查配置文件
if [ ! -f "config.py" ]; then
    echo "❌ 错误: 未找到 config.py 配置文件"
    exit 1
fi

echo "✅ 配置文件已找到"

# 显示当前配置
echo ""
echo "📋 当前配置:"
echo "目标IP: $(python3 -c "import config; print(config.TARGET_IP)")"
echo "目标端口: $(python3 -c "import config; print(config.TARGET_PORT)")"
echo "发送间隔: $(python3 -c "import config; print(config.SEND_INTERVAL)")秒"
echo ""

# 选择运行模式
echo "请选择运行模式:"
echo "1. 简化版发送器 (推荐)"
echo "2. 完整版发送器"
echo "3. 退出"

read -p "请输入选择 (1-3): " choice

case $choice in
    1)
        echo "🚀 启动简化版UDP发送器..."
        python3 simple_udp_sender.py
        ;;
    2)
        echo "🚀 启动完整版UDP发送器..."
        python3 udp_sender.py
        ;;
    3)
        echo "👋 退出"
        exit 0
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac 