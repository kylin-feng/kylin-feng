#!/bin/bash

# GitHub自动化工具启动脚本

echo "🚀 启动GitHub自动化工具..."

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3，请先安装Python"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建Python虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "📥 安装依赖包..."
pip install -r requirements.txt

# 启动应用
echo "✅ 启动Web应用..."
echo "🌐 访问地址: http://localhost:58899"
echo "⏰ 定时任务已设置为每天晚上8点执行"
echo "📖 按Ctrl+C停止应用"
echo ""

python app.py
