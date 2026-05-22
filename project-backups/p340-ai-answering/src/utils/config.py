"""
config.py
该模块提供配置文件的加载和初始化功能

Author: Zhu Jiahao
Date: 2025-07-14
"""

import os
import shutil
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """配置文件管理类

    """

    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self._config = {}
        self.load_config()

    def load_config(self) -> None:
        """加载配置文件
        """
        try:
            print("正在加载配置文件 ...")

            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)

            # @TODO: 处理环境变量

            # 创建路径
            self.create_dict()


        except FileNotFoundError as e:
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")

    def create_dict(self) -> None:
        """创建用于存储输入数据和输出数据的路径
        """
        paths = self._config.get('paths', {})
        for path_type, path_config in paths.items():
            if isinstance(path_config, dict):
                for key, path_str in path_config.items():
                    # Path(path_str).mkdir(parents=True, exist_ok=True)
                    self.ensure_empty_dir(path_str)
            elif isinstance(path_config, str):
                # Path(path_config).mkdir(parents=True, exist_ok=True)
                self.ensure_empty_dir(path_config)  

        print("所有目录都已经创建完毕!")

    def ensure_empty_dir(self, path_str: str) -> None:
        """确保每个路径都为空

        Args:
            path_str: 具体路径
        """
        path = Path(path_str)
        
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
        
        path.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """ 从字典中获取对应值(使用点分隔)

        Args:
            key: 键
            default: 默认值

        Returns:
            对应的存储值
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

    def get_api_config(self, service: str) -> Dict[str, str]:
        """ 获取API配置
        """
        return self.get(f'api.{service}', {})
    
    def get_path_config(self) -> Dict[str, Any]:
        """ 获取路径配置
        """
        return self.get('paths', {})
    
    def get_robot_config(self) -> Dict[str, Any]:
        """ 获取机器人配置
        """
        return self.get('robot', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """ 获取日志配置
        """
        return self.get('logging', {})
    
    def get_assets_config(self) -> Dict[str, Any]:
        """ 获取外部资产配置
        """
        return self.get('assets', {})
    
    def get_camera_config(self) -> Dict[str, Any]:
        """ 获取相机配置
        """
        return self.get('camera', {})

# Global instance of config manager
__config__ = ConfigManager()