from flask import Flask, jsonify, make_response
import socket
import random
import json
import time
import threading

app = Flask(__name__)

# 添加CORS支持
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# 颜色定义
COLORS = {
    1: "绿色",
    2: "青色", 
    3: "红色",
    4: "紫色",
    5: "黄色",
    6: "棕色",
    7: "白色",
    8: "金色",
    9: "黑色",
    10: "蓝色",
    11: "灰色",
    12: "银色"
}

# 颜色描述
COLOR_DESCRIPTIONS = {
    1: "绿色代表生机与希望，今天适合开始新的计划",
    2: "青色象征清新与智慧，保持冷静思考会有好运",
    3: "红色充满热情与活力，大胆行动会带来机会",
    4: "紫色神秘而高贵，直觉敏锐的一天",
    5: "黄色明亮温暖，乐观的心态会吸引好运",
    6: "棕色稳重踏实，脚踏实地会有收获",
    7: "白色纯净简约，心境平和会带来好运",
    8: "金色富贵吉祥，财运亨通的好日子",
    9: "黑色深邃神秘，内在力量觉醒的时刻",
    10: "蓝色宁静深远，平静中蕴含无限可能",
    11: "灰色中庸平衡，保持低调会有意外收获",
    12: "银色优雅高贵，贵人运势强劲"
}

def send_udp_request(host='127.0.0.1', port=9999, message='get_lucky_color'):
    """
    向UDP服务器发送请求获取幸运色数据
    如果没有UDP服务器，会随机生成幸运色
    """
    try:
        # 创建UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(3)  # 设置3秒超时
        
        # 发送请求
        sock.sendto(message.encode('utf-8'), (host, port))
        
        # 接收响应
        response, addr = sock.recvfrom(1024)
        sock.close()
        
        # 尝试解析UDP服务器返回的数据
        try:
            data = json.loads(response.decode('utf-8'))
            return data.get('colors', [random.randint(1, 12)])
        except:
            # 如果解析失败，返回随机颜色
            return [random.randint(1, 12)]
            
    except socket.timeout:
        print("UDP请求超时，使用随机颜色")
        return [random.randint(1, 12)]
    except Exception as e:
        print(f"UDP请求失败: {e}，使用随机颜色")
        return [random.randint(1, 12)]

def generate_random_lucky_colors():
    """生成随机幸运色组合"""
    # 随机选择1-3个颜色
    num_colors = random.randint(1, 3)
    colors = random.sample(range(1, 13), num_colors)
    return colors

def send_to_hardware(colors, description):
    """
    将幸运色数据发送到硬件设备
    
    Args:
        colors (list): 颜色ID列表
        description (str): 颜色描述
    """
    try:
        # 导入配置
        import config
        
        # 硬件设备配置
        HARDWARE_IP = config.HARDWARE_IP
        HARDWARE_PORT = config.HARDWARE_PORT
        BUFSIZE = 1024
        
        print(f"🎯 准备发送到硬件设备: {HARDWARE_IP}:{HARDWARE_PORT}")
        
        # 创建UDP客户端
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(5)
        
        # 发送控制命令
        command_msg = "C0,R180,F1"
        client.sendto(command_msg.encode('utf-8'), (HARDWARE_IP, HARDWARE_PORT))
        print(f"✅ 发送控制命令成功: {command_msg}")
        
        # 构建幸运色消息
        color_names = [COLORS.get(c, "未知") for c in colors]
        color_text = "、".join(color_names)
        lucky_msg = f"今日幸运色: {color_text} - {description}"
        
        # 发送幸运色数据
        N_M = 2
        data_msg = f"A{str(int(N_M))}B{lucky_msg}"
        client.sendto(data_msg.encode('utf-8'), (HARDWARE_IP, HARDWARE_PORT))
        print(f"✅ 发送幸运色数据成功: {data_msg}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ 发送到硬件设备失败: {e}")
        return False

def send_to_hardware_async(colors, description):
    """
    异步发送到硬件设备（不阻塞API响应）
    """
    def send_thread():
        time.sleep(0.1)  # 稍微延迟，确保API响应先返回
        send_to_hardware(colors, description)
    
    thread = threading.Thread(target=send_thread)
    thread.daemon = True
    thread.start()

@app.route('/api/lucky-color', methods=['GET'])
def get_lucky_color():
    """获取幸运色接口"""
    try:
        # 方法1: 通过UDP获取（如果有UDP服务器）
        # colors = send_udp_request()
        
        # 方法2: 直接生成随机幸运色（当前使用）
        colors = generate_random_lucky_colors()
        
        # 生成描述
        if len(colors) == 1:
            description = COLOR_DESCRIPTIONS.get(colors[0], "未知颜色")
        else:
            color_names = [COLORS.get(c, "未知") for c in colors]
            description = f"今日幸运色组合：{' + '.join(color_names)}，多重好运加持，事事顺心如意"
        
        # 异步发送到硬件设备
        send_to_hardware_async(colors, description)
        
        response = {
            "color": colors,
            "description": description,
            "hardware_sent": True
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": f"获取幸运色失败: {str(e)}"
        }), 500

@app.route('/api/lucky-color-udp', methods=['GET'])
def get_lucky_color_with_udp():
    """通过UDP获取幸运色接口"""
    try:
        # 通过UDP请求获取幸运色
        colors = send_udp_request()
        
        # 生成描述
        if len(colors) == 1:
            description = COLOR_DESCRIPTIONS.get(colors[0], "未知颜色")
        else:
            color_names = [COLORS.get(c, "未知") for c in colors]
            description = f"今日幸运色组合：{' + '.join(color_names)}，多重好运加持，事事顺心如意"
        
        response = {
            "color": colors,
            "description": description
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": f"获取幸运色失败: {str(e)}"
        }), 500

@app.route('/api/colors', methods=['GET'])
def get_color_definitions():
    """获取颜色定义接口"""
    return jsonify({
        "colors": COLORS,
        "descriptions": COLOR_DESCRIPTIONS
    })

@app.route('/api/send-to-hardware', methods=['POST'])
def send_lucky_color_to_hardware():
    """发送幸运色到硬件设备接口"""
    try:
        # 生成幸运色
        colors = generate_random_lucky_colors()
        
        # 生成描述
        if len(colors) == 1:
            description = COLOR_DESCRIPTIONS.get(colors[0], "未知颜色")
        else:
            color_names = [COLORS.get(c, "未知") for c in colors]
            description = f"今日幸运色组合：{' + '.join(color_names)}，多重好运加持，事事顺心如意"
        
        # 发送到硬件设备
        success = send_to_hardware(colors, description)
        
        response = {
            "color": colors,
            "description": description,
            "hardware_sent": success,
            "message": "发送到硬件设备成功" if success else "发送到硬件设备失败"
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": f"发送到硬件设备失败: {str(e)}"
        }), 500

# 健康检查接口
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    print("启动Flask幸运色服务...")
    print("接口说明:")
    print("- GET /api/lucky-color : 获取随机幸运色（自动发送到硬件）")
    print("- POST /api/send-to-hardware : 发送幸运色到硬件设备")
    print("- GET /api/lucky-color-udp : 通过UDP获取幸运色") 
    print("- GET /api/colors : 获取颜色定义")
    print("- GET /health : 健康检查")
    print("")
    print("硬件设备配置:")
    print(f"- 目标IP: 192.168.43.103")
    print(f"- 目标端口: 5001")
    print(f"- 控制命令: C0,R180,F1")
    print(f"- 数据格式: A2B[幸运色消息]")
    
    app.run(host='0.0.0.0', port=5001, debug=True)