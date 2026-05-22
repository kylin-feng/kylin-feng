#!/bin/bash

echo "🎨 启动幸运色生成器后端服务..."
echo "服务将在 http://10.10.100.19:5001 启动"
echo ""

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

# 启动后端服务
echo "🚀 启动后端服务..."
python3 backend.py 