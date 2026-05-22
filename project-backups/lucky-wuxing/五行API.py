# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 12:01:13 2025

@author: 32118
"""

from flask import Flask, request, jsonify, Response
from lunar_python import Lunar, Solar
from collections import Counter
import random
import json
import 五行计算
import socket

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

def send_udp_message(host, port, message, max_retries=3, timeout=1):
    """
    通过UDP发送消息到指定主机和端口
    :param host: 目标主机
    :param port: 目标端口
    :param message: 要发送的消息
    :return: None
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)  # 设置超时
    
    for attempt in range(max_retries):
        try:
            sock.sendto(message.encode('utf-8'), (host, port))
            print("发送成功")
            return True
        except socket.timeout:
            print(f"尝试 {attempt + 1}/{max_retries} 超时")
        except Exception as e:
            print(f"发送失败: {e}")
            break
    
    sock.close()
    return False

@app.route('/send-udp', methods=['POST'])
def send_udp():
    """
    Flask API端点，接收JSON数据并通过UDP发送
    预期JSON格式:
    {
        "host": "目标IP或主机名",
        "port": 目标端口,
        "message": "要发送的消息"
    }
    """
    data = request.get_json()
    
    if not data or 'host' not in data or 'port' not in data or 'message' not in data:
        return jsonify({"error": "缺少必要参数: host, port 或 message"}), 400
    
    try:
        port = int(data['port'])
    except ValueError:
        return jsonify({"error": "端口必须是整数"}), 400
    
    success = send_udp_message(data['host'], port, data['message'])
    
    if success:
        return jsonify({"status": "消息已发送", 
                       "host": data['host'],
                       "port": port,
                       "message": data['message']})
    else:
        return jsonify({"error": "消息发送失败"}), 500

@app.route('/device/out', methods=['GET'])
def control_device_color():
    """
    控制设备颜色的接口
    随机返回1-5的整数，表示不同类型的颜色
    """
    color_code = random.randint(1, 5)
    return str(color_code)


@app.route('/api/bazi', methods=['GET'])
def calculate_bazi():
    # 获取查询参数
    year = request.args.get('year', type=int)
    month = request.args.get('month', type=int)
    day = request.args.get('day', type=int)
    hour = request.args.get('hour', type=int, default=0)
    minute = request.args.get('minute', type=int, default=0)
    
    # 验证必填参数
    if None in [year, month, day]:
        return jsonify({
            "error": "Missing required parameters",
            "message": "Year, month and day are required"
        }), 400
    
    try:
        output = 五行计算.calculate_bazi(year, month, day, hour, minute)
        outcome = {
            "color": output["幸运色"],
            "descripton": output["适宜和忌讳"]
            }
        
        color_str = "A" + output["幸运色"][0] + "B" + output["幸运色"][1] + "C" + output["幸运色"][2]
        print(color_str + "D" + outcome["descripton"])
        send_udp_message("127.0.0.1", 5001, color_str + "D" + outcome["descripton"])
        
        return Response(
            json.dumps(outcome, ensure_ascii=False),
            mimetype='application/json; charset=utf-8'
        )
    
    except Exception as e:
        return jsonify({
            "error": "error",
            "message": str(e)
        }), 400


if __name__ == '__main__':
    app.run(host='10.10.103.30', port=5000, debug=True)