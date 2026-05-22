from flask import Flask, request
from flask_cors import CORS
import socket
import time
import requests
import json
import random
from lunar_python import Lunar
from datetime import datetime

app = Flask(__name__)
CORS(app)  # 启用跨域支持

# UDP配置
BUFSIZE = 1024
client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip_port = ('192.168.43.103', 5001)  # 机器人IP地址

# 大模型API配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "your-siliconflow-api-key"
MODEL = "Qwen/QwQ-32B"

# 颜色对应的幸运数字
COLOR_LUCKY_NUMBERS = {
    "绿色": 1,
    "青色": 2,
    "红色": 3,
    "紫色": 4,
    "黄色": 5,
    "棕色": 6,
    "白色": 7,
    "金色": 8,
    "黑色": 9,
    "蓝色": 10,
    "灰色": 11,
    "银色": 12
}

# 运势测算提示词
FORTUNE_PROMPT = """你是一个专业的英文运势测算师。请根据用户的姓名、出生日期和随机生成的幸运颜色，生成一段个性化的英文运势测算。

颜色对应的幸运数字：
绿色(1), 青色(2), 红色(3), 紫色(4), 黄色(5), 棕色(6), 
白色(7), 金色(8), 黑色(9), 蓝色(10), 灰色(11), 银色(12)

请根据用户的个人信息和幸运颜色，生成一段100字以内的个性化运势测算，可以是吉兆也可以是凶兆，要简洁有趣且与用户信息相关。
格式：直接输出运势测算内容，不要加任何前缀或解释。
使用英文输出，只能输出英文，禁止输出中文
"""

# 天干地支对照表
TIAN_GAN = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
DI_ZHI = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
WU_XING = ["木", "火", "土", "金", "水"]

# 天干五行对照
TIAN_GAN_WU_XING = {
    "甲": "木", "乙": "木",
    "丙": "火", "丁": "火", 
    "戊": "土", "己": "土",
    "庚": "金", "辛": "金",
    "壬": "水", "癸": "水"
}

# 地支五行对照
DI_ZHI_WU_XING = {
    "子": "水", "丑": "土", "寅": "木", "卯": "木",
    "辰": "土", "巳": "火", "午": "火", "未": "土",
    "申": "金", "酉": "金", "戌": "土", "亥": "水"
}

# 运势测算模板（备用）
FORTUNE_TEMPLATES = [
    "今日财运亨通，适合投资理财",
    "小心谨慎，避免冲动消费",
    "贵人相助，事业有成",
    "注意健康，多休息少熬夜",
    "桃花运旺，感情甜蜜",
    "工作顺利，升职加薪有望",
    "出行平安，旅途愉快",
    "学习进步，考试顺利",
    "家庭和睦，幸福美满",
    "小心意外，注意安全",
    "财运不佳，量入为出",
    "心情愉悦，万事如意"
]

def get_random_lucky_color():
    """随机生成幸运颜色"""
    return random.choice(list(COLOR_LUCKY_NUMBERS.keys()))

def convert_solar_to_lunar(birthday_str):
    """将阳历日期转换为阴历日期"""
    try:
        # 解析阳历日期
        date_obj = datetime.strptime(birthday_str, '%Y-%m-%d')
        
        # 转换为农历
        lunar = Lunar.fromDate(date_obj)
        
        return {
            'year': lunar.getYear(),
            'month': lunar.getMonth(),
            'day': lunar.getDay(),
            'year_gz': lunar.getYearInGanZhi(),  # 年干支
            'month_gz': lunar.getMonthInGanZhi(),  # 月干支
            'day_gz': lunar.getDayInGanZhi(),  # 日干支
            'time_gz': lunar.getTimeInGanZhi(),  # 时干支
            'year_name': lunar.getYearInChinese(),
            'month_name': lunar.getMonthInChinese(),
            'day_name': lunar.getDayInChinese()
        }
    except Exception as e:
        print(f"日期转换失败: {str(e)}")
        return None

def calculate_bazi(lunar_info):
    """计算生辰八字"""
    if not lunar_info:
        return None
    
    try:
        # 提取四柱
        year_gz = lunar_info['year_gz']
        month_gz = lunar_info['month_gz']
        day_gz = lunar_info['day_gz']
        time_gz = lunar_info['time_gz']
        
        # 分析五行
        year_tiangan = year_gz[:1]
        year_dizhi = year_gz[1:]
        month_tiangan = month_gz[:1]
        month_dizhi = month_gz[1:]
        day_tiangan = day_gz[:1]
        day_dizhi = day_gz[1:]
        time_tiangan = time_gz[:1]
        time_dizhi = time_gz[1:]
        
        # 计算五行统计
        wuxing_count = {}
        for gan in [year_tiangan, month_tiangan, day_tiangan, time_tiangan]:
            wuxing = TIAN_GAN_WU_XING.get(gan, "")
            wuxing_count[wuxing] = wuxing_count.get(wuxing, 0) + 1
        
        for zhi in [year_dizhi, month_dizhi, day_dizhi, time_dizhi]:
            wuxing = DI_ZHI_WU_XING.get(zhi, "")
            wuxing_count[wuxing] = wuxing_count.get(wuxing, 0) + 1
        
        return {
            'bazi': f"{year_gz} {month_gz} {day_gz} {time_gz}",
            'year_gz': year_gz,
            'month_gz': month_gz,
            'day_gz': day_gz,
            'time_gz': time_gz,
            'wuxing_count': wuxing_count,
            'lunar_date': f"{lunar_info['year_name']}年{lunar_info['month_name']}月{lunar_info['day_name']}日"
        }
    except Exception as e:
        print(f"八字计算失败: {str(e)}")
        return None

def get_fortune_by_color(color, user_name=None, user_birthday=None):
    """根据颜色和用户信息获取运势测算"""
    try:
        # 构造大模型API请求
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # 构建包含用户信息的prompt
        user_info = ""
        bazi_info = ""
        
        if user_name and user_birthday:
            # 转换阳历为阴历
            lunar_info = convert_solar_to_lunar(user_birthday)
            if lunar_info:
                # 计算生辰八字
                bazi_result = calculate_bazi(lunar_info)
                if bazi_result:
                    bazi_info = f"\nLunar Calendar Date: {bazi_result['lunar_date']}\nBaZi (Eight Characters): {bazi_result['bazi']}\nWuXing Analysis: {dict(bazi_result['wuxing_count'])}"
            
            user_info = f"\nUser Name: {user_name}\nSolar Calendar Birth Date: {user_birthday}{bazi_info}"
        
        prompt = f"{FORTUNE_PROMPT}\n\nLucky Color: {color}{user_info}"
        
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 100,
            "temperature": 0.8
        }
        
        # 调用大模型API
        print(f"正在生成运势测算，幸运颜色: {color}")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            fortune = result['choices'][0]['message']['content'].strip()
            print(f"运势测算: {fortune}")
            
            # 在每个标点符号后面添加换行符
            fortune_with_newlines = ""
            for char in fortune:
                fortune_with_newlines += char
                if char in '.!?,;:':
                    fortune_with_newlines += '\n'
            return fortune_with_newlines
        else:
            print(f"API调用失败，使用备用模板")
            return random.choice(FORTUNE_TEMPLATES)
            
    except Exception as e:
        print(f"运势测算生成失败: {str(e)}，使用备用模板")
        return random.choice(FORTUNE_TEMPLATES)

@app.route('/send_udp', methods=['POST'])
def send_udp():
    """
    运势测算接口
    随机生成幸运颜色，生成幸运数字A和运势测算B，通过UDP发送
    """
    try:
        # 获取POST数据
        data = request.get_data(as_text=True)
        
        if not data:
            return "错误：没有接收到数据", 400
        
        # 检查是否是JSON格式
        try:
            json_data = json.loads(data)
            if 'fortune' in json_data or len(json_data) == 0:
                # 运势测算模式 - 随机生成幸运颜色
                lucky_color = get_random_lucky_color()
                
                # 获取幸运数字A
                lucky_number = COLOR_LUCKY_NUMBERS[lucky_color]
                
                # 获取用户信息
                user_name = json_data.get('userName', '')
                user_birthday = json_data.get('userBirthday', '')
                
                # 生成运势测算B
                fortune = get_fortune_by_color(lucky_color, user_name, user_birthday)
                
                # 构造UDP消息
                N_M = lucky_number
                msg = "A" + str(int(N_M)) + "B" + fortune
                
                # 发送UDP包
                client.sendto(msg.encode('utf-8'), ip_port)
                
                return {
                    "status": "success",
                    "message": "运势测算已发送",
                    "lucky_color": lucky_color,
                    "lucky_number": lucky_number,
                    "fortune": fortune
                }
        except json.JSONDecodeError:
            # 不是JSON格式，直接发送原始数据
            pass
        
        # 直接发送原始数据
        N_M = 2
        msg = "A" + str(int(N_M)) + "B" + data
        
        # 发送UDP包
        client.sendto(msg.encode('utf-8'), ip_port)
        
        return "OK"
        
    except requests.exceptions.Timeout:
        return "错误：大模型API请求超时", 500
    except requests.exceptions.RequestException as e:
        return f"错误：API请求失败 - {str(e)}", 500
    except Exception as e:
        return f"发送失败: {str(e)}", 500

if __name__ == '__main__':
    print("启动运势测算服务...")
    print(f"目标地址: {ip_port[0]}:{ip_port[1]}")
    print(f"大模型API: {API_URL}")
    print("支持的颜色:", list(COLOR_LUCKY_NUMBERS.keys()))
    app.run(host='0.0.0.0', port=5003, debug=True)