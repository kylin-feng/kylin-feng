#!/bin/bash

echo "=== 之江智慧 AI会议记录工具 ==="
echo "启动脚本"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    exit 1
fi

# 检查Node.js
if ! command -v npm &> /dev/null; then
    echo "错误: 未找到 Node.js/npm"
    exit 1
fi

echo "正在安装后端依赖..."
cd backend
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo "正在安装前端依赖..."
cd ../frontend
npm install --registry=https://registry.npmmirror.com

echo ""
echo "依赖安装完成！"
echo ""
echo "启动说明:"
echo "1. 后端: cd backend && python3 main.py"
echo "2. 前端: cd frontend && npm run dev"
echo ""
echo "访问地址: http://localhost:11130"
echo "后端API: http://localhost:11131"
echo ""
echo "注意: 需要配置通义千问API密钥到环境变量 DASHSCOPE_API_KEY"