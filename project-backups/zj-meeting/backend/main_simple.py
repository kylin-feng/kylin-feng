from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="之江智慧 Meeting API")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"message": "之江智慧 AI会议记录工具 - API 运行正常", "status": "ok"}

@app.post("/auth/register")
async def register(username: str, email: str, password: str):
    return {"message": "注册成功"}

@app.post("/auth/login")
async def login(username: str, password: str):
    return {"access_token": "test-token", "token_type": "bearer"}

@app.post("/meetings/create")
async def create_meeting(title: str):
    return {"id": 1, "title": title}

@app.get("/meetings")
async def get_meetings():
    return []

@app.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: int):
    return {
        "id": meeting_id,
        "title": "示例会议",
        "transcript": None,
        "analysis": None,
        "created_at": "2024-11-13T10:00:00"
    }

@app.post("/meetings/{meeting_id}/upload-audio")
async def upload_audio(meeting_id: int, file: UploadFile = File(...)):
    return {"transcript": "这是示例转录内容"}

@app.post("/meetings/{meeting_id}/analyze")
async def analyze_meeting(meeting_id: int):
    return {
        "analysis": {
            "summary": "这是AI分析的会议总结示例",
            "key_points": ["要点1", "要点2"],
            "tasks": [
                {"task": "示例任务", "assignee": "负责人", "deadline": "本周"}
            ],
            "action_items": ["行动项1", "行动项2"]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)