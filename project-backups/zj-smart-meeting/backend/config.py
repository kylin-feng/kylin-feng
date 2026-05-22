"""
配置管理模块
统一管理应用配置信息
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
from loguru import logger

# 加载环境变量
load_dotenv()

class Config:
    """应用配置类"""
    
    # 应用基础配置
    APP_NAME = "之江智会"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # 服务器配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # 通义千问配置
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    QWEN_MODEL_MAX = os.getenv("QWEN_MODEL_MAX", "qwen-max")
    QWEN_MODEL_PLUS = os.getenv("QWEN_MODEL_PLUS", "qwen-plus")
    
    # DeepSeek配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    
    # 阿里云ASR配置
    ASR_APP_KEY = os.getenv("ASR_APP_KEY", "")
    ASR_ACCESS_KEY_ID = os.getenv("ASR_ACCESS_KEY_ID", "")
    ASR_ACCESS_KEY_SECRET = os.getenv("ASR_ACCESS_KEY_SECRET", "")
    
    # Qdrant向量数据库配置
    QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "meeting_knowledge")
    VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 1536))
    
    # 数据库配置
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./meeting.db")
    
    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
    
    @classmethod
    def get_agent_config(cls) -> Dict[str, Any]:
        """获取所有Agent的配置"""
        return {
            "recorder": {
                "dashscope_api_key": cls.DASHSCOPE_API_KEY,
                "asr_app_key": cls.ASR_APP_KEY,
                "enable_speaker_detection": True,
                "sample_rate": 16000
            },
            "analyst": {
                "dashscope_api_key": cls.DASHSCOPE_API_KEY,
                "model_name": cls.QWEN_MODEL_MAX,
                "analysis_interval": 60,
                "min_segments": 5
            },
            "summarizer": {
                "dashscope_api_key": cls.DASHSCOPE_API_KEY,
                "model_name": cls.QWEN_MODEL_PLUS
            },
            "task_agent": {
                "deepseek_api_key": cls.DEEPSEEK_API_KEY,
                "api_base": cls.DEEPSEEK_API_BASE
            },
            "knowledge": {
                "dashscope_api_key": cls.DASHSCOPE_API_KEY,
                "qdrant_url": cls.QDRANT_URL,
                "qdrant_port": cls.QDRANT_PORT,
                "collection_name": cls.QDRANT_COLLECTION_NAME,
                "vector_size": cls.VECTOR_SIZE
            },
            "coordinator": {
                "enable_realtime_analysis": True,
                "analysis_threshold": 10
            }
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """验证配置是否完整"""
        required_keys = [
            "DASHSCOPE_API_KEY"
        ]
        
        missing_keys = []
        for key in required_keys:
            if not getattr(cls, key, None):
                missing_keys.append(key)
        
        if missing_keys:
            logger.warning(f"缺少配置项: {', '.join(missing_keys)}")
            logger.warning("部分功能可能受限，建议配置完整的环境变量")
            return False
        
        return True

# 全局配置实例
config = Config()


