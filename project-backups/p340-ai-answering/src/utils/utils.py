"""
utils.py 
所有常见工具函数的集合

Author: Zhu Jiahao
Date: 2025-07-17
"""

import base64
import os
import re
import json

# A4纸张尺寸 (mm)
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0

def encode_image_to_base64(path: str) -> str:
    """
    将图像文件编码为Base64字符串

        处理过程:
        1. 以二进制模式读取文件内容
        2. 使用base64进行编码
        3. 将bytes类型结果解码为UTF-8字符串
    
    Args:
        path (str): 图像文件的路径
        
    Return:
        str: 经过Base64编码后的UTF-8格式字符串
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
def get_image_mime_type(path: str) -> str:
    """
    根据文件扩展名获取对应的MIME类型

        支持格式:
        - .jpg/.jpeg -> image/jpeg
        - .png -> image/png
        - .webp -> image/webp

    Args:
        path (str): 图像文件的路径
        
    Return:
        str: 标准MIME类型字符串
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    elif ext == '.webp':
        return 'image/webp'
    else:
        raise ValueError(f"Unsupported image format: {ext}")

def read_txt_file(path: str, encoding: str = "utf-8") -> str:
    """
    读取一个 .txt 文件，并返回其全部内容（str 格式）。

    Args:
        path (str): 文件路径
        encoding (str): 文件编码，默认为 'utf-8'

    Returns:
        str: 文件内容
    """
    with open(path, "r", encoding=encoding) as f:
        return f.read()

def format_text_to_json(
    text: str,
    output_json_path: str,
    start_x_mm: float = 55.1,
    start_y_mm: float = 128.34,
    char_height_mm: float = 6.39,
    char_spacing_ratio: float = 1.2,
    max_chars_per_line: int = 15
):
    """
    按标点优先、每15字强制换行，将文本写入 JSON 文件。
    """
    def split_text_by_punctuation(text, max_len):
        pattern = re.compile(r'[^，。！？、；：,.!?;:]+[，。！？、；：,.!?;:]?')
        chunks = pattern.findall(text)
        
        lines = []
        current_line = ''
        for chunk in chunks:
            for ch in chunk:
                current_line += ch
                if ch in '，。！？、；：,.!?;:':
                    lines.append(current_line)
                    current_line = ''
                elif len(current_line) >= max_len:
                    lines.append(current_line)
                    current_line = ''
        if current_line:
            lines.append(current_line)
        return lines

    lines = split_text_by_punctuation(text, max_chars_per_line)

    json_data = []
    for i, line in enumerate(lines):
        entry = {
            "text": line,
            "a4_x_mm": start_x_mm,
            "a4_y_mm": start_y_mm + i * char_height_mm,
            "char_height_mm": char_height_mm,
            "char_spacing_ratio": char_spacing_ratio
        }
        json_data.append(entry)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"写入完成，共 {len(json_data)} 行，已保存至 {output_json_path}")