#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版UDP发送器 - 基于原始代码优化
"""

import socket
import time
import random

# 配置参数
BUFSIZE = 1024
TARGET_IP = '10.10.100.19'  # 目标设备IP地址
TARGET_PORT = 5001           # 目标端口
SEND_INTERVAL = 3           # 发送间隔(秒)

def create_udp_client():
    """创建UDP客户端"""
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(5)  # 设置超时时间
    return client

def send_command_message(client, command):
    """发送控制命令消息"""
    try:
        client.sendto(command.encode('utf-8'), (TARGET_IP, TARGET_PORT))
        print(f"✅ 发送命令成功: {command}")
        return True
    except Exception as e:
        print(f"❌ 发送命令失败: {e}")
        return False

def send_data_message(client, data, node_id=2):
    """发送数据消息"""
    try:
        message = f"A{node_id}B{data}"
        client.sendto(message.encode('utf-8'), (TARGET_IP, TARGET_PORT))
        print(f"✅ 发送数据成功: {message}")
        return True
    except Exception as e:
        print(f"❌ 发送数据失败: {e}")
        return False

def send_lucky_color_message(client):
    """发送幸运色消息"""
    # 生成随机幸运色
    colors = random.sample(range(1, 13), random.randint(1, 3))
    color_names = {
        1: "绿色", 2: "青色", 3: "红色", 4: "紫色", 5: "黄色", 6: "棕色",
        7: "白色", 8: "金色", 9: "黑色", 10: "蓝色", 11: "灰色", 12: "银色"
    }
    
    color_text = "、".join([color_names.get(c, "未知") for c in colors])
    message = f"今日幸运色: {color_text}"
    
    return send_data_message(client, message)

def continuous_send_original():
    """原始代码的持续发送功能"""
    print(f"🚀 开始持续发送到 {TARGET_IP}:{TARGET_PORT}")
    print("按 Ctrl+C 停止发送")
    
    client = create_udp_client()
    
    try:
        while True:
            # 发送控制命令
            command_msg = "C0,R180,F1"
            send_command_message(client, command_msg)
            
            # 发送数据消息
            data_msg = "Today is a sunny day, someone say a lot, But you don't need care"
            send_data_message(client, data_msg, node_id=2)
            
            print(f"⏰ 等待 {SEND_INTERVAL} 秒...")
            time.sleep(SEND_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断发送")
    except Exception as e:
        print(f"❌ 发送过程中出现错误: {e}")
    finally:
        client.close()
        print("🔒 连接已关闭")

def continuous_send_lucky_colors():
    """持续发送幸运色数据"""
    print(f"🎨 开始持续发送幸运色到 {TARGET_IP}:{TARGET_PORT}")
    print("按 Ctrl+C 停止发送")
    
    client = create_udp_client()
    
    try:
        while True:
            send_lucky_color_message(client)
            print(f"⏰ 等待 {SEND_INTERVAL} 秒...")
            time.sleep(SEND_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断发送")
    except Exception as e:
        print(f"❌ 发送过程中出现错误: {e}")
    finally:
        client.close()
        print("🔒 连接已关闭")

def test_connection():
    """测试连接"""
    print(f"🔍 测试连接到 {TARGET_IP}:{TARGET_PORT}")
    
    client = create_udp_client()
    
    try:
        test_msg = "TEST_CONNECTION"
        client.sendto(test_msg.encode('utf-8'), (TARGET_IP, TARGET_PORT))
        print("✅ 连接测试成功")
        return True
    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return False
    finally:
        client.close()

def main():
    """主函数"""
    print("🎯 简化版UDP发送器")
    print("=" * 40)
    
    # 测试连接
    if not test_connection():
        print("❌ 无法连接到目标设备，请检查IP地址和端口")
        return
    
    print("\n📋 选择发送模式:")
    print("1. 原始模式 (控制命令 + 数据)")
    print("2. 幸运色模式 (发送幸运色数据)")
    print("3. 退出")
    
    while True:
        try:
            choice = input("\n请选择模式 (1-3): ").strip()
            
            if choice == "1":
                continuous_send_original()
                break
            elif choice == "2":
                continuous_send_lucky_colors()
                break
            elif choice == "3":
                print("👋 退出程序")
                break
            else:
                print("❌ 无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n👋 用户退出")
            break
        except Exception as e:
            print(f"❌ 操作失败: {e}")

if __name__ == "__main__":
    main() 