#!/usr/bin/env python3
"""
启动脚本 - SmartMeet AI Gradio 版本
简单直接，专注功能
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_backend():
    """检查后端服务是否运行"""
    try:
        response = requests.get("http://localhost:5001/api/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def start_backend():
    """启动后端服务"""
    backend_dir = Path(__file__).parent / "backend"
    if backend_dir.exists():
        print("🚀 启动后端服务...")
        os.chdir(backend_dir)
        subprocess.Popen(["npm", "run", "dev"], shell=True)
        
        # 等待后端启动
        for i in range(30):
            if check_backend():
                print("✅ 后端服务已启动")
                return True
            time.sleep(1)
            print(f"⏳ 等待后端启动... ({i+1}/30)")
        
        print("❌ 后端启动超时")
        return False
    else:
        print("❌ 后端目录不存在")
        return False

def install_requirements():
    """安装Python依赖"""
    print("📦 检查Python依赖...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖安装完成")
        return True
    except subprocess.CalledProcessError:
        print("❌ 依赖安装失败")
        return False

def main():
    print("""
╔══════════════════════════════════════╗
║        SmartMeet AI - Gradio版       ║
║        命令行风格，专注功能          ║
╚══════════════════════════════════════╝
    """)
    
    # 切换到项目目录
    os.chdir(Path(__file__).parent)
    
    # 1. 安装依赖
    if not install_requirements():
        return
    
    # 2. 检查后端服务
    if not check_backend():
        print("⚠️  后端服务未运行，尝试启动...")
        if not start_backend():
            print("❌ 后端启动失败，请手动启动后端服务")
            print("   cd backend && npm run dev")
            return
    else:
        print("✅ 后端服务已运行")
    
    # 3. 启动Gradio界面
    print("🎛️  启动Gradio界面...")
    try:
        from gradio_app import create_interface
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_tips=False
        )
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出程序")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()