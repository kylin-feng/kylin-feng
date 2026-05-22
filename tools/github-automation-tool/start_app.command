#!/bin/bash
cd "$(dirname "$0")"
echo "🚀 启动GitHub自动化工具..."
echo "📂 工作目录: $(pwd)"
source venv/bin/activate
echo "✅ 虚拟环境已激活"
echo "🌐 启动Web应用..."
echo "📱 访问地址: http://localhost:58899"
echo "⏰ 定时任务设置为每天晚上8点执行"
echo "📖 按Ctrl+C停止应用"
echo "----------------------------------------"
python app.py