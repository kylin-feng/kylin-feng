"""
capture.py

图像捕获模块

试卷捕获流程：
1. 通过摄像头拍摄试卷页面
2. 图像增强处理
3. 逆时针旋转90°并裁剪至A4纸大小
4. 使用Qwen-VL模型进行OCR文字识别


Author: Zhu Jiahao
Date: 2025-07-16
"""

from src.utils.config import __config__
from src.utils.logger import __logger__

class CaptureManager:
    """ 管理图像捕获的类
    """
    def __init__(self):
        self.image_num = 0
        self.image_path = __config__.get_path_config().get("images")

    
