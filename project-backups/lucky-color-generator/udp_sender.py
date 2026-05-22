#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UDP发送器 - 向硬件设备发送数据
支持多种消息格式和配置
"""

import socket
import time
import json
import random
from datetime import datetime

class UDPSender:
    def __init__(self, target_ip='10.10.100.19', target_port=5001, buffer_size=1024):
        """
        初始化UDP发送器
        
        Args:
            target_ip (str): 目标设备IP地址
            target_port (int): 目标端口
            buffer_size (int): 缓冲区大小
        """
        self.target_ip = target_ip
        self.target_port = target_port
        self.buffer_size = buffer_size
        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client.settimeout(5)  # 设置超时时间
        
    def send_message(self, message, message_type="text"):
        """
        发送消息到目标设备
        
        Args:
            message (str): 要发送的消息
            message_type (str): 消息类型 ("text", "command", "data")
        """
        try:
            # 根据消息类型格式化数据
            if message_type == "command":
                # 命令格式: C0,R180,F1
                formatted_msg = message
            elif message_type == "data":
                # 数据格式: A2BToday is a sunny day...
                formatted_msg = message
            else:
                # 文本格式: 直接发送
                formatted_msg = message
            
            # 发送数据
            self.client.sendto(formatted_msg.encode('utf-8'), (self.target_ip, self.target_port))
            print(f"✅ 发送成功: {formatted_msg}")
            return True
            
        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False
    
    def send_command(self, command):
        """
        发送控制命令
        
        Args:
            command (str): 控制命令
        """
        return self.send_message(command, "command")
    
    def send_data(self, data, node_id=2):
        """
        发送数据包
        
        Args:
            data (str): 数据内容
            node_id (int): 节点ID
        """
        formatted_data = f"A{node_id}B{data}"
        return self.send_message(formatted_data, "data")
    
    def send_lucky_color_data(self):
        """
        发送幸运色数据到硬件设备
        """
        # 生成幸运色数据
        colors = random.sample(range(1, 13), random.randint(1, 3))
        color_names = {
            1: "绿色", 2: "青色", 3: "红色", 4: "紫色", 5: "黄色", 6: "棕色",
            7: "白色", 8: "金色", 9: "黑色", 10: "蓝色", 11: "灰色", 12: "银色"
        }
        
        color_text = "、".join([color_names.get(c, "未知") for c in colors])
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 构建消息
        message = f"今日幸运色: {color_text} | 时间: {timestamp}"
        
        return self.send_data(message)
    
    def continuous_send(self, message, interval=3, max_count=None):
        """
        持续发送消息
        
        Args:
            message (str): 要发送的消息
            interval (int): 发送间隔(秒)
            max_count (int): 最大发送次数，None表示无限发送
        """
        count = 0
        print(f"🚀 开始持续发送消息，间隔: {interval}秒")
        
        try:
            while max_count is None or count < max_count:
                success = self.send_message(message)
                count += 1
                
                if max_count:
                    print(f"📊 进度: {count}/{max_count}")
                
                if not success:
                    print("⚠️ 发送失败，等待重试...")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断发送")
        except Exception as e:
            print(f"❌ 发送过程中出现错误: {e}")
    
    def test_connection(self):
        """
        测试连接
        """
        print(f"🔍 测试连接到 {self.target_ip}:{self.target_port}")
        
        try:
            # 发送测试消息
            test_msg = "TEST_CONNECTION"
            self.client.sendto(test_msg.encode('utf-8'), (self.target_ip, self.target_port))
            print("✅ 连接测试成功")
            return True
        except Exception as e:
            print(f"❌ 连接测试失败: {e}")
            return False
    
    def close(self):
        """
        关闭连接
        """
        self.client.close()
        print("🔒 连接已关闭")

def main():
    """
    主函数 - 演示UDP发送器的使用
    """
    print("🎯 UDP发送器启动")
    print("=" * 50)
    
    # 创建发送器实例
    sender = UDPSender(target_ip='10.10.100.19', target_port=5001)
    
    # 测试连接
    if not sender.test_connection():
        print("❌ 无法连接到目标设备，请检查IP地址和端口")
        return
    
    print("\n📋 可用操作:")
    print("1. 发送控制命令")
    print("2. 发送数据包")
    print("3. 发送幸运色数据")
    print("4. 持续发送")
    print("5. 退出")
    
    while True:
        try:
            choice = input("\n请选择操作 (1-5): ").strip()
            
            if choice == "1":
                command = input("请输入控制命令 (如: C0,R180,F1): ")
                sender.send_command(command)
                
            elif choice == "2":
                data = input("请输入数据内容: ")
                node_id = input("请输入节点ID (默认2): ") or "2"
                sender.send_data(data, int(node_id))
                
            elif choice == "3":
                sender.send_lucky_color_data()
                
            elif choice == "4":
                message = input("请输入要持续发送的消息: ")
                interval = input("请输入发送间隔(秒，默认3): ") or "3"
                max_count = input("请输入最大发送次数(默认10): ") or "10"
                
                sender.continuous_send(
                    message, 
                    interval=int(interval), 
                    max_count=int(max_count) if max_count != "0" else None
                )
                
            elif choice == "5":
                print("👋 退出程序")
                break
                
            else:
                print("❌ 无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n👋 用户退出")
            break
        except Exception as e:
            print(f"❌ 操作失败: {e}")
    
    sender.close()

if __name__ == "__main__":
    main() 