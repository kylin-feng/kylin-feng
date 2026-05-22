#!/usr/bin/env python3
"""
之江智慧 AI会议记录工具 - 简化版后端
"""
import json
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# 创建FastAPI应用
app = FastAPI(
    title="之江智慧 Meeting API",
    description="AI会议记录工具后端API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 简单的内存存储
users_db = {}
meetings_db = {}
meeting_counter = 1

@app.get("/")
async def health_check():
    """健康检查端点"""
    return {
        "message": "之江智慧 AI会议记录工具 - API 运行正常",
        "status": "ok",
        "version": "1.0.0"
    }

@app.post("/auth/register")
async def register(username: str, email: str, password: str):
    """用户注册"""
    if username in users_db:
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    users_db[username] = {
        "email": email,
        "password": password  # 实际应用中应该加密
    }
    return {"message": "注册成功"}

@app.post("/auth/login")
async def login(username: str, password: str):
    """用户登录"""
    if username not in users_db:
        raise HTTPException(status_code=400, detail="用户名不存在")
    
    if users_db[username]["password"] != password:
        raise HTTPException(status_code=400, detail="密码错误")
    
    return {
        "access_token": f"token_{username}",
        "token_type": "bearer"
    }

@app.post("/meetings/create")
async def create_meeting(title: str):
    """创建会议"""
    global meeting_counter
    meeting_id = meeting_counter
    meeting_counter += 1
    
    meetings_db[meeting_id] = {
        "id": meeting_id,
        "title": title,
        "transcript": None,
        "analysis": None,
        "created_at": "2024-11-13T10:00:00"
    }
    
    return {"id": meeting_id, "title": title}

@app.get("/meetings")
async def get_meetings():
    """获取会议列表"""
    return [
        {
            "id": meeting["id"],
            "title": meeting["title"],
            "created_at": meeting["created_at"]
        }
        for meeting in meetings_db.values()
    ]

@app.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: int):
    """获取会议详情"""
    if meeting_id not in meetings_db:
        raise HTTPException(status_code=404, detail="会议不存在")
    
    return meetings_db[meeting_id]

@app.post("/meetings/{meeting_id}/upload-audio")
async def upload_audio(meeting_id: int, file: UploadFile = File(...)):
    """上传音频文件"""
    if meeting_id not in meetings_db:
        raise HTTPException(status_code=404, detail="会议不存在")
    
    # 模拟音频转录
    transcript = f"这是会议《{meetings_db[meeting_id]['title']}》的模拟转录内容。实际应用中会调用语音识别API处理上传的文件：{file.filename}"
    
    meetings_db[meeting_id]["transcript"] = transcript
    
    return {"transcript": transcript}

@app.post("/meetings/{meeting_id}/analyze")
async def analyze_meeting(meeting_id: int):
    """AI分析会议"""
    if meeting_id not in meetings_db:
        raise HTTPException(status_code=404, detail="会议不存在")
    
    meeting = meetings_db[meeting_id]
    if not meeting["transcript"]:
        raise HTTPException(status_code=400, detail="请先上传音频文件")
    
    # 模拟AI分析结果（实际应用中会调用通义千问API）
    analysis = {
        "summary": f"这是AI对《{meeting['title']}》会议的智能分析总结。会议讨论了项目进展、任务分配和时间安排等重要议题。",
        "key_points": [
            "讨论了项目整体进展情况",
            "明确了各团队成员的职责分工",
            "确定了下一阶段的工作重点",
            "制定了详细的时间计划表"
        ],
        "tasks": [
            {
                "task": "完成前端界面设计",
                "assignee": "UI设计师",
                "deadline": "本周五"
            },
            {
                "task": "后端API开发",
                "assignee": "开发工程师",
                "deadline": "下周三"
            },
            {
                "task": "系统集成测试",
                "assignee": "测试工程师",
                "deadline": "下周五"
            }
        ],
        "action_items": [
            "准备下次会议的进度汇报材料",
            "跟进客户需求反馈",
            "更新项目文档",
            "安排团队技术分享会"
        ]
    }
    
    meetings_db[meeting_id]["analysis"] = analysis
    
    return {"analysis": analysis}

if __name__ == "__main__":
    print("🚀 启动之江智慧 AI会议记录工具后端服务...")
    print("📡 API文档地址: http://127.0.0.1:8000/docs")
    print("✅ 服务器启动成功!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3001,
        log_level="info"
    )