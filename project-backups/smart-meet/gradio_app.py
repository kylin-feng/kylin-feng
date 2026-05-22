#!/usr/bin/env python3
"""
SmartMeet AI - 命令行风格的智能会议助手
使用 Gradio 实现简洁高效的界面
"""

import gradio as gr
import requests
import json
import time
import asyncio
import threading
from datetime import datetime
from typing import List, Dict, Any
import pyaudio
import wave
import tempfile
import os

# 配置
API_BASE_URL = "http://localhost:5001/api"
SAMPLE_RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

class SmartMeetApp:
    def __init__(self):
        self.session_id = None
        self.is_recording = False
        self.audio_thread = None
        self.agents_status = {}
        self.transcription_log = []
        self.meeting_stats = {
            "start_time": None,
            "segments_count": 0,
            "active_agents": 0
        }
        
    def check_backend_status(self):
        """检查后端服务状态"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                return "✅ 后端服务正常运行"
            else:
                return f"❌ 后端服务异常: {response.status_code}"
        except requests.RequestException as e:
            return f"❌ 无法连接后端服务: {str(e)}"
    
    def start_meeting(self):
        """启动会议"""
        try:
            meeting_data = {
                "title": f"AI会议 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "participants": ["用户"],
                "date": datetime.now().isoformat()
            }
            
            response = requests.post(
                f"{API_BASE_URL}/realtime/collaboration/start",
                json={"meetingData": meeting_data},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data["data"]["sessionId"]
                self.meeting_stats["start_time"] = datetime.now()
                return f"✅ 会议已启动\n会话ID: {self.session_id}", self.session_id
            else:
                return f"❌ 启动失败: {response.text}", None
                
        except Exception as e:
            return f"❌ 启动失败: {str(e)}", None
    
    def stop_meeting(self):
        """停止会议"""
        self.is_recording = False
        self.session_id = None
        self.meeting_stats = {"start_time": None, "segments_count": 0, "active_agents": 0}
        return "⏹️ 会议已停止"
    
    def get_agents_status(self):
        """获取智能体状态"""
        if not self.session_id:
            return "❌ 会议未启动"
            
        try:
            response = requests.get(
                f"{API_BASE_URL}/realtime/collaboration/{self.session_id}/status",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()["data"]
                agents = data.get("agents", {})
                
                status_text = "🤖 智能体状态:\n"
                for agent_type, agent_info in agents.items():
                    status = agent_info.get("status", "idle")
                    progress = agent_info.get("progress", 0)
                    
                    status_emoji = {
                        "idle": "⭕",
                        "working": "🟢", 
                        "analyzing": "🔵",
                        "completed": "✅"
                    }.get(status, "❓")
                    
                    status_text += f"{status_emoji} {agent_type}: {status}"
                    if status == "working" and progress > 0:
                        status_text += f" ({progress}%)"
                    status_text += "\n"
                
                return status_text
            else:
                return f"❌ 获取状态失败: {response.status_code}"
                
        except Exception as e:
            return f"❌ 获取状态失败: {str(e)}"
    
    def simulate_transcription(self, text_input):
        """模拟转录（用于测试）"""
        if not self.session_id:
            return "❌ 会议未启动", ""
            
        if not text_input.strip():
            return "❌ 请输入文本", ""
        
        try:
            # 创建模拟的音频数据
            segment = {
                "id": f"{self.session_id}_{int(time.time())}",
                "sessionId": self.session_id,
                "text": text_input,
                "speaker": "用户",
                "confidence": 0.95,
                "timestamp": datetime.now().isoformat(),
                "duration": 3000,
                "language": "zh-CN"
            }
            
            # 发送到后端处理
            response = requests.post(
                f"{API_BASE_URL}/realtime/collaboration/{self.session_id}/process",
                json={"transcriptionSegment": segment},
                timeout=10
            )
            
            if response.status_code == 200:
                self.transcription_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 用户: {text_input}")
                self.meeting_stats["segments_count"] += 1
                
                # 更新转录日志显示
                log_text = "\n".join(self.transcription_log[-10:])  # 只显示最近10条
                
                return "✅ 转录已处理", log_text
            else:
                return f"❌ 处理失败: {response.text}", ""
                
        except Exception as e:
            return f"❌ 处理失败: {str(e)}", ""
    
    def generate_minutes(self):
        """生成会议纪要"""
        if not self.session_id:
            return "❌ 会议未启动"
            
        try:
            response = requests.post(
                f"{API_BASE_URL}/realtime/collaboration/{self.session_id}/minutes",
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()["data"]
                
                minutes_text = "📝 会议纪要生成完成:\n\n"
                
                for version_type, content in data.items():
                    if isinstance(content, dict):
                        minutes_text += f"## {version_type.upper()}版本\n"
                        minutes_text += f"标题: {content.get('title', 'N/A')}\n"
                        minutes_text += f"摘要: {content.get('summary', 'N/A')}\n"
                        if content.get('keyPoints'):
                            minutes_text += "关键点:\n"
                            for point in content['keyPoints']:
                                minutes_text += f"  • {point}\n"
                        minutes_text += "\n"
                
                return minutes_text
            else:
                return f"❌ 生成失败: {response.text}"
                
        except Exception as e:
            return f"❌ 生成失败: {str(e)}"
    
    def get_meeting_stats(self):
        """获取会议统计"""
        if not self.meeting_stats["start_time"]:
            return "❌ 会议未启动"
        
        duration = datetime.now() - self.meeting_stats["start_time"]
        duration_str = str(duration).split('.')[0]  # 去掉微秒
        
        stats = f"""📊 会议统计:
• 会话ID: {self.session_id or 'N/A'}
• 开始时间: {self.meeting_stats['start_time'].strftime('%H:%M:%S') if self.meeting_stats['start_time'] else 'N/A'}
• 持续时间: {duration_str}
• 转录片段: {self.meeting_stats['segments_count']}
• 活跃智能体: {self.meeting_stats['active_agents']}"""
        
        return stats

# 创建应用实例
app = SmartMeetApp()

# 创建 Gradio 界面
def create_interface():
    with gr.Blocks(
        title="SmartMeet AI - 命令行风格", 
        theme=gr.themes.Monochrome(),
        css="""
        .gradio-container { 
            font-family: 'Courier New', monospace !important; 
            background: #0a0a0a !important; 
            color: #00ff00 !important; 
        }
        .gr-button { 
            background: #1a1a1a !important; 
            border: 1px solid #00ff00 !important; 
            color: #00ff00 !important; 
        }
        .gr-textbox { 
            background: #1a1a1a !important; 
            border: 1px solid #00ff00 !important; 
            color: #00ff00 !important; 
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🤖 SmartMeet AI - 智能会议助手
        ## 命令行风格界面 | 专注功能，拒绝花哨
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 🎛️ 控制面板")
                
                # 后端状态检查
                status_btn = gr.Button("🔍 检查后端状态", variant="secondary")
                status_output = gr.Textbox(
                    label="系统状态", 
                    lines=2, 
                    interactive=False,
                    placeholder="点击检查后端服务状态..."
                )
                
                # 会议控制
                with gr.Row():
                    start_btn = gr.Button("▶️ 启动会议", variant="primary")
                    stop_btn = gr.Button("⏹️ 停止会议", variant="stop")
                
                meeting_output = gr.Textbox(
                    label="会议状态", 
                    lines=3, 
                    interactive=False,
                    placeholder="会议未启动..."
                )
                
                # 文本输入模拟转录
                gr.Markdown("### 💬 模拟语音输入")
                text_input = gr.Textbox(
                    label="输入文本(模拟语音)", 
                    placeholder="输入要转录的文本内容...",
                    lines=2
                )
                process_btn = gr.Button("🎯 处理转录")
                process_output = gr.Textbox(
                    label="处理结果", 
                    lines=2, 
                    interactive=False
                )
                
            with gr.Column(scale=3):
                gr.Markdown("### 📊 实时监控")
                
                # 智能体状态
                agents_btn = gr.Button("🤖 刷新智能体状态")
                agents_output = gr.Textbox(
                    label="智能体状态", 
                    lines=8, 
                    interactive=False,
                    placeholder="智能体状态将在这里显示..."
                )
                
                # 转录日志
                transcription_output = gr.Textbox(
                    label="转录日志", 
                    lines=10, 
                    interactive=False,
                    placeholder="转录内容将在这里显示..."
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 会议统计")
                stats_btn = gr.Button("📊 获取统计")
                stats_output = gr.Textbox(
                    label="统计信息", 
                    lines=6, 
                    interactive=False
                )
                
            with gr.Column():
                gr.Markdown("### 📝 生成纪要")
                minutes_btn = gr.Button("📄 生成会议纪要")
                minutes_output = gr.Textbox(
                    label="会议纪要", 
                    lines=6, 
                    interactive=False
                )
        
        # 事件绑定
        status_btn.click(
            fn=app.check_backend_status,
            outputs=status_output
        )
        
        start_btn.click(
            fn=app.start_meeting,
            outputs=[meeting_output, gr.State()]
        )
        
        stop_btn.click(
            fn=app.stop_meeting,
            outputs=meeting_output
        )
        
        process_btn.click(
            fn=app.simulate_transcription,
            inputs=text_input,
            outputs=[process_output, transcription_output]
        )
        
        agents_btn.click(
            fn=app.get_agents_status,
            outputs=agents_output
        )
        
        stats_btn.click(
            fn=app.get_meeting_stats,
            outputs=stats_output
        )
        
        minutes_btn.click(
            fn=app.generate_minutes,
            outputs=minutes_output
        )
        
        # 自动刷新 (每5秒)
        def auto_refresh():
            return app.get_agents_status(), app.get_meeting_stats()
        
        # 定时器组件
        timer = gr.Timer(5.0)
        timer.tick(
            fn=auto_refresh,
            outputs=[agents_output, stats_output]
        )
    
    return demo

if __name__ == "__main__":
    print("🚀 启动 SmartMeet AI 命令行界面...")
    print("📍 请确保后端服务已在 http://localhost:5001 运行")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_tips=False,
        quiet=False
    )