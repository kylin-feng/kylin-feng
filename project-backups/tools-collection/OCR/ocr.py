#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path

def ocr_image(image_path):
    """使用macOS Vision框架进行OCR识别"""
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在 {image_path}")
        return
    
    swift_script = "/Users/shixianping/ocr.swift"
    if not os.path.exists(swift_script):
        print("错误: OCR脚本未找到，请重新运行配置")
        return
    
    try:
        # 使用swift命令运行OCR
        result = subprocess.run([
            'swift', swift_script, image_path
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"OCR执行失败: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print("OCR超时，请检查图片文件")
    except FileNotFoundError:
        print("错误: Swift编译器未找到，请确保Xcode Command Line Tools已安装")
        print("安装命令: xcode-select --install")
    except Exception as e:
        print(f"执行OCR时出错: {e}")

def main():
    if len(sys.argv) != 2:
        print("macOS OCR工具")
        print("使用方法: python3 ocr.py <图片路径>")
        print("支持格式: JPG, PNG, TIFF, BMP")
        print("支持语言: 中文、英文")
        print()
        print("示例:")
        print("  python3 ocr.py screenshot.png")
        print("  python3 ocr.py ~/Documents/image.jpg")
        return
    
    image_path = sys.argv[1]
    
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)
    
    ocr_image(image_path)

if __name__ == "__main__":
    main()
