# -*- coding: utf-8 -*-
"""
UDP发送器配置文件
"""

# 目标设备配置
TARGET_IP = '10.10.100.19'  # 目标设备IP地址
TARGET_PORT = 5001           # 目标端口

# 硬件设备配置
HARDWARE_IP = '192.168.43.103'  # 硬件设备IP地址
HARDWARE_PORT = 5001            # 硬件设备端口

# 发送配置
BUFSIZE = 1024               # 缓冲区大小
SEND_INTERVAL = 3           # 发送间隔(秒)
TIMEOUT = 5                 # 超时时间(秒)

# 消息配置
DEFAULT_NODE_ID = 2         # 默认节点ID
DEFAULT_COMMAND = "C0,R180,F1"  # 默认控制命令
DEFAULT_MESSAGE = "Today is a sunny day, someone say a lot, But you don't need care"  # 默认消息

# 幸运色配置
COLOR_NAMES = {
    1: "绿色", 2: "青色", 3: "红色", 4: "紫色", 5: "黄色", 6: "棕色",
    7: "白色", 8: "金色", 9: "黑色", 10: "蓝色", 11: "灰色", 12: "银色"
}

# 调试配置
DEBUG = True                # 是否启用调试模式
LOG_LEVEL = "INFO"         # 日志级别 