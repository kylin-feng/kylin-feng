"""
qwen_api.py

阿里Qwen模型API调用模块, 包含Qwen3文本分割模型和Qwen-VL图像识别模型

Author: Zhu Jiahao
Date: 2025-07-15
"""

import os
from openai import OpenAI
from src.utils.utils import encode_image_to_base64, get_image_mime_type
from src.utils.config import __config__
from src.utils.logger import __logger__

__all__ = ['QwenClient']

qwen_logger = __logger__.get_module_logger("Qwen")

class QwenClient:
    """Qwen服务API
    """
    def __init__(self, api_key, base_url, vl_model, text_model):
        self.api_key = api_key
        self.base_url = base_url
        self.vl_model = vl_model
        self.text_model = text_model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def ocr_image(self, image_path: str, log_path: str, prompt: str=None) -> None:
        """ 对图片列表进行OCR, 返回并拼接成完整的文本。
        
        Args:
            image_path (str): 图片路径
            log_path (str): 日志记录路径
            prompt (str): 可选, 系统消息内容。
        """
        try:
            b64 = encode_image_to_base64(image_path)
            mime = get_image_mime_type(image_path)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": prompt or "你是一个试卷识别助手，请准确提取试卷中的所有文字内容，不要添加任何解释或说明，直接输出试卷原文。"}]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": "请准确提取这张试卷中的所有文字内容"}
                ]}
            ]
            response = self.client.chat.completions.create(
                model=self.vl_model,
                messages=messages,
                stream=True
            )
            result = ""
            with open(log_path, "a", encoding="utf-8") as f:
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        content = delta.content
                        print(content, end="", flush=True)  # 输出到终端
                        f.write(content)                    # 写入文件
                        result += content
            print(" ")
            
        except Exception as e:
            qwen_logger.error(f"QwenClient OCR error: {e}")
            exit()

    def text_split(self, text: str) -> str:
        """ 文本分割

        Args:
            text (str): 待分割的文本
        """
        qwen_logger.info(f"正在进行文本分割, 文本长度: {len(text)}...")

        prompt = (
            "请将下面的 OCR 完整文本切分成若干“单元”，每个单元严格按照如下格式输出：\n"
            "==== UnitN ====\n"
            "<单元N的标题>\n"
            "<该单元阅读材料原文（跨页内容一并写出）>\n"
            "每道题以“【第<index>题】”开头，后跟题干原文，题与题之间留一个空行。\n"
            "不要输出多余说明，保持和原文一样只是分开，当单元跨了多页，可以删去单元中间的（第x页/共x页【第x页】）除此之外不要比原文增加或者减少任何一个字。只输出上述格式的纯文本。\n\n"
            f"{text}"
        )

        try:
            response = self.client.chat.completions.create(
                self.text_model,
                messages=[
                    {"role": "system", "content": "你是一个文本切割助手，输出时严格按照约定格式。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                stream_options={"include_usage": False}
            )

        except Exception as e:
            qwen_logger.error(f"文本分割模块错误: {e}")
            return ""

        result = response.choices[0].message.content.strip()
        qwen_logger.info("非流式响应已完成")
        return result