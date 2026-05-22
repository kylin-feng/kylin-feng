#!/bin/bash
cd "$(dirname "$0")"
echo "🔄 重启GitHub自动化工具..."
echo "📂 工作目录: $(pwd)"

# 停止旧进程
echo "🛑 停止旧进程..."
pkill -f "python app.py" 2>/dev/null || true
lsof -ti:58899 | xargs kill -9 2>/dev/null || true

# 等待端口释放
sleep 2

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 启动应用
echo "✅ 启动新的应用实例..."
echo "🌐 访问地址: http://localhost:58899"
echo "⚡ 更新功能: 现在支持自动推送到GitHub!"
echo "⏰ 定时任务: 每天晚上8点自动执行"
echo "📖 按Ctrl+C停止应用"
echo "========================================="
python app.py