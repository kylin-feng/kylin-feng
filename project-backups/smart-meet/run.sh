#!/bin/bash

# SmartMeet AI - 一键启动脚本
# 命令行风格，简单直接

echo "╔══════════════════════════════════════╗"
echo "║        SmartMeet AI 启动器           ║"
echo "║        命令行风格，专注功能          ║"
echo "╚══════════════════════════════════════╝"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装Python3"
    exit 1
fi

# 检查Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装，请先安装Node.js"
    exit 1
fi

# 检查npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm 未安装，请先安装npm"
    exit 1
fi

echo "✅ 环境检查通过"
echo ""

# 安装Python依赖
echo "📦 安装Python依赖..."
pip3 install -r requirements.txt

# 检查后端依赖
if [ ! -d "backend/node_modules" ]; then
    echo "📦 安装后端依赖..."
    cd backend
    npm install
    cd ..
fi

# 启动程序
echo "🚀 启动SmartMeet AI..."
python3 start_gradio.py