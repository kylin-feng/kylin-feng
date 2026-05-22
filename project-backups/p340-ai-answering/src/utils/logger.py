"""
logger.py

该模块提供了一个全局的日志记录功能

Author: Zhu Jiahao
Date: 2025-07-14
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from .config import __config__

class Logger:
    """日志管理器
    """

    def __init__(self):
        self._loggers = {}
        self.setup_root_logger()

    def setup_root_logger(self) -> None:
        """ 设置根日志管理器
        """
        log_config = __config__.get_logging_config()
        log_level = getattr(logging, log_config.get('level', 'INFO'))

        # 初始化
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.handlers.clear()
        formatter = logging.Formatter(
            log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

        # 命令行Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # 文件Handler
        log_path = __config__.get('paths.output.logs')
        if log_path:
            log_file = Path(log_path) / "AIExam.log"
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=log_config.get('max_files', 7),
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """ 获取指定logger, 若无该logger, 则重新初始化一个
        
        Args:
            name: 名称
        """
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]

    def get_module_logger(self, module_name: str) -> logging.Logger:
        """ 获取指定模块的logger

        Args:
            module_name: 名称
        """
        return self.get_logger(f"AIExam.{module_name}")
    
# 全局实例
__logger__ = Logger()
