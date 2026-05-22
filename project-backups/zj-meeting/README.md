# 之江智慧 - Agent会议记录工具

简洁画风的AI会议记录工具，提供音频转文本、智能分析、任务提取功能。

## 功能特性
- 音频实时录制和转文本
- AI智能分析会议内容
- 自动提取任务和总结
- 用户注册登录系统

## 技术栈
- 前端: Next.js + React + TailwindCSS
- 后端: Python + FastAPI
- 数据库: SQLite
- AI模型: 通义千问API

## 端口配置
- 前端: http://localhost:11130
- 后端: http://localhost:11131

## 快速启动

### 后端启动
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### 前端启动
```bash
cd frontend
npm install
npm run dev
```