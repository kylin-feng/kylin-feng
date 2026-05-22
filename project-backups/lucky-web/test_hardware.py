#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试硬件设备UDP发送功能
"""

import socket
import time
import config

def test_hardware_connection():
    """测试硬件设备连接"""
    print("🔍 测试硬件设备连接...")
    
    try:
        # 创建UDP客户端
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(5)
        
        # 发送测试消息
        test_msg = "TEST_CONNECTION"
        client.sendto(test_msg.encode('utf-8'), (config.HARDWARE_IP, config.HARDWARE_PORT))
        print(f"✅ 测试消息已发送到 {config.HARDWARE_IP}:{config.HARDWARE_PORT}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return False

def test_command_send():
    """测试控制命令发送"""
    print("🎮 测试控制命令发送...")
    
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(5)
        
        # 发送控制命令
        command_msg = "C0,R180,F1"
        client.sendto(command_msg.encode('utf-8'), (config.HARDWARE_IP, config.HARDWARE_PORT))
        print(f"✅ 控制命令发送成功: {command_msg}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ 控制命令发送失败: {e}")
        return False

def test_data_send():
    """测试数据消息发送"""
    print("📊 测试数据消息发送...")
    
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(5)
        
        # 发送数据消息
        N_M = 2
        data_msg = f"A{str(int(N_M))}B测试消息: 今日幸运色是绿色"
        client.sendto(data_msg.encode('utf-8'), (config.HARDWARE_IP, config.HARDWARE_PORT))
        print(f"✅ 数据消息发送成功: {data_msg}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ 数据消息发送失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎯 硬件设备UDP发送测试")
    print("=" * 40)
    print(f"目标设备: {config.HARDWARE_IP}:{config.HARDWARE_PORT}")
    print("")
    
    # 测试连接
    if not test_hardware_connection():
        print("❌ 无法连接到硬件设备，请检查IP地址和网络连接")
        return
    
    print("")
    
    # 测试控制命令
    if not test_command_send():
        print("❌ 控制命令发送失败")
        return
    
    print("")
    
    # 测试数据消息
    if not test_data_send():
        print("❌ 数据消息发送失败")
        return
    
    print("")
    print("✅ 所有测试通过！硬件设备UDP发送功能正常")

if __name__ == "__main__":
    main() 