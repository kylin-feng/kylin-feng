"""
deepseek_api.py

DeepSeek模型API调用模块

Author: Zhu Jiahao
Date: 2025-07-15
"""

import os
from openai import OpenAI
from src.utils.utils import read_txt_file
from src.utils.config import __config__
from src.utils.logger import __logger__

__all__ = ['DeepseekClient']

deepseek_logger = __logger__.get_module_logger("DeepSeek")

class DeepSeekClient:
    """ DeepSeek服务API
    """
    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def answer_reasoning_question(self, question_path: str, log_path: str) -> None:
        """ 调用DeepSeek, 回答一个推理类问题

        Args:
            question_path (str): 问题路径
            log_path (str): 日志记录路径
        """
        question = read_txt_file(question_path)
        prompt = (
            f"你是一位经验丰富的侦探，请根据以下题目文本推理并得出合理结论。"
            f"要求：简要说明推理过程并给出结论，总字数控制在50字以内，不得包含无关内容，答案不需要标注字数。\n\n"
            f"{question}\n\n"
            "请开始作答："
        )


        deepseek_logger.info("Deepseek正在作答...")
        try:
            # 调用DeepSeek API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位经验丰富的侦探，请严格按照用户要求的内容，分析题目并给出答案"},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            result = ""
            with open(log_path, "w", encoding="utf-8") as f:
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        content = delta.content
                        print(content, end="", flush=True)  # 输出到终端
                        f.write(content)                    # 写入文件
                        result += content
            print(" ")

        except Exception as e:
            deepseek_logger.error(f"DeepSeekClient Error: {e}")

    def answer_translation_question(self, question_path: str, log_path: str) -> None:
        """ 调用DeepSeek, 回答一个文言文翻译问题

        Args:
            question_path (str): 问题路径
            log_path (str): 日志记录路径
        """
        question = read_txt_file(question_path)
        prompt = (
            f"你是一位经验丰富的文言文翻译大师，请将下面这段文言文翻译成白话文。"
            f"要求：总字数控制在100字以内，不得包含无关内容，答案不需要标注字数。\n\n"
            f"{question}\n\n"
            "请开始作答："
        )

        deepseek_logger.info("Deepseek正在作答...")
        try:
            # 调用DeepSeek API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位经验丰富的文言文翻译大师，请严格按照用户要求的内容，进行翻译并给出答案"},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            result = ""
            with open(log_path, "w", encoding="utf-8") as f:
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        content = delta.content
                        print(content, end="", flush=True)  # 输出到终端
                        f.write(content)                    # 写入文件
                        result += content
            print(" ")

        except Exception as e:
            deepseek_logger.error(f"DeepSeekClient Error: {e}")

    def answer_english_question(self, question_path: str, log_path: str) -> None:
        """ 调用DeepSeek, 写一篇英语作文

        Args:
            question_path (str): 问题路径
            log_path (str): 日志记录路径
        """
        question = read_txt_file(question_path)
        prompt = (
            f"你是一位高考考生，请根据以下题目，撰写一篇英语作文"
            f"要求：总字数控制在100词以内，不得包含无关内容，答案不需要标注字数。\n\n"
            f"{question}\n\n"
            "请开始作答："
        )


        deepseek_logger.info("Deepseek正在作答...")
        try:
            # 调用DeepSeek API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位高考考生，请严格按照用户要求的内容，分析题目并给出答案"},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            result = ""
            with open(log_path, "w", encoding="utf-8") as f:
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        content = delta.content
                        print(content, end="", flush=True)  # 输出到终端
                        f.write(content)                    # 写入文件
                        result += content
            print(" ")

        except Exception as e:
            deepseek_logger.error(f"DeepSeekClient Error: {e}")

    def answer_math_question(self, question_path: str, log_path: str) -> None:
        """ 调用DeepSeek, 写三道数学题

        Args:
            question_path (str): 问题路径
            log_path (str): 日志记录路径
        """
        question = read_txt_file(question_path)
        prompt = (
            f"你是一位高考考生，请根据以下题目，回答三道数学算数题"
            f"要求：只需要答案， 三个答案使用逗号分隔，不得包含无关内容。\n\n"
            f"{question}\n\n"
            "请开始作答："
        )


        deepseek_logger.info("Deepseek正在作答...")
        try:
            # 调用DeepSeek API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位高考考生，请严格按照用户要求的内容，分析题目并给出答案"},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            result = ""
            with open(log_path, "w", encoding="utf-8") as f:
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        content = delta.content
                        print(content, end="", flush=True)  # 输出到终端
                        f.write(content)                    # 写入文件
                        result += content
            print(" ")

        except Exception as e:
            deepseek_logger.error(f"DeepSeekClient Error: {e}")