from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import dashscope
import json
import os
import uvicorn

# 数据库配置
SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 密码加密
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("JWT_SECRET", "change-me-in-production")
ALGORITHM = "HS256"

# 通义千问API配置
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "your-dashscope-api-key")

app = FastAPI(title="之江智慧 Meeting API")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据库模型
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Meeting(Base):
    __tablename__ = "meetings"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    transcript = Column(Text)
    analysis = Column(Text)
    tasks = Column(Text)
    summary = Column(Text)
    user_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# 依赖注入
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# API路由
@app.get("/")
async def health_check():
    return {"message": "之江智慧 AI会议记录工具 - API 运行正常", "status": "ok"}

@app.post("/auth/register")
async def register(username: str, email: str, password: str, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    user = User(
        username=username,
        email=email,
        password=get_password_hash(password)
    )
    db.add(user)
    db.commit()
    return {"message": "注册成功"}

@app.post("/auth/login")
async def login(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password):
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/meetings/create")
async def create_meeting(title: str, db: Session = Depends(get_db)):
    meeting = Meeting(title=title, user_id=1)  # 简化版本，固定用户ID
    db.add(meeting)
    db.commit()
    db.refresh(meeting)
    return {"id": meeting.id, "title": meeting.title}

@app.post("/meetings/{meeting_id}/upload-audio")
async def upload_audio(meeting_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="会议不存在")
    
    # 简化版音频转文本（实际应用中需要集成语音识别服务）
    transcript = f"这是会议 {meeting.title} 的转录内容示例。实际项目中这里会调用语音识别API。"
    
    meeting.transcript = transcript
    db.commit()
    
    return {"transcript": transcript}

@app.post("/meetings/{meeting_id}/analyze")
async def analyze_meeting(meeting_id: int, db: Session = Depends(get_db)):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting or not meeting.transcript:
        raise HTTPException(status_code=404, detail="会议或转录不存在")
    
    try:
        # 调用通义千问进行分析
        from dashscope import Generation
        
        prompt = f"""
        请分析以下会议内容，提取关键信息：
        {meeting.transcript}
        
        请按以下格式返回JSON：
        {{
            "summary": "会议总结",
            "key_points": ["要点1", "要点2"],
            "tasks": [
                {{"task": "任务描述", "assignee": "负责人", "deadline": "截止时间"}},
            ],
            "action_items": ["行动项1", "行动项2"]
        }}
        """
        
        response = Generation.call(
            model="qwen-plus",
            prompt=prompt,
            result_format='message'
        )
        
        analysis = response.output.text
        
        # 简化版分析结果（如果API调用失败）
        if not analysis:
            analysis = json.dumps({
                "summary": "这是AI分析的会议总结",
                "key_points": ["讨论了项目进展", "确定了下一步计划"],
                "tasks": [
                    {"task": "完成设计稿", "assignee": "设计师", "deadline": "本周五"},
                    {"task": "代码审查", "assignee": "开发团队", "deadline": "下周一"}
                ],
                "action_items": ["准备下次会议材料", "跟进客户反馈"]
            }, ensure_ascii=False)
        
        meeting.analysis = analysis
        db.commit()
        
        return {"analysis": json.loads(analysis)}
    
    except Exception as e:
        # 降级处理
        fallback_analysis = {
            "summary": "AI分析暂不可用，这是示例总结",
            "key_points": ["会议要点1", "会议要点2"],
            "tasks": [{"task": "示例任务", "assignee": "待分配", "deadline": "待定"}],
            "action_items": ["跟进会议决议"]
        }
        meeting.analysis = json.dumps(fallback_analysis, ensure_ascii=False)
        db.commit()
        
        return {"analysis": fallback_analysis}

@app.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: int, db: Session = Depends(get_db)):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="会议不存在")
    
    analysis = None
    if meeting.analysis:
        try:
            analysis = json.loads(meeting.analysis)
        except:
            analysis = {"summary": "解析失败"}
    
    return {
        "id": meeting.id,
        "title": meeting.title,
        "transcript": meeting.transcript,
        "analysis": analysis,
        "created_at": meeting.created_at
    }

@app.get("/meetings")
async def get_meetings(db: Session = Depends(get_db)):
    meetings = db.query(Meeting).order_by(Meeting.created_at.desc()).all()
    return [{"id": m.id, "title": m.title, "created_at": m.created_at} for m in meetings]

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
