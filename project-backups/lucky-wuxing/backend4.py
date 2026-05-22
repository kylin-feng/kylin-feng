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
ip_port = ('192.168.206.138', 5001)  # 机器人IP地址

# 大模型API配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "your-siliconflow-api-key"
MODEL = "Qwen/QwQ-32B"

# Lucky numbers corresponding to colors
COLOR_LUCKY_NUMBERS = {
    "Green": 1,
    "Cyan": 2,
    "Red": 3,
    "Purple": 4,
    "Yellow": 5,
    "Brown": 6,
    "White": 7,
    "Gold": 8,
    "Black": 9,
    "Blue": 10,
    "Gray": 11,
    "Silver": 12
}

# Fortune telling prompt
FORTUNE_PROMPT = """You are a professional English fortune teller. Please generate a personalized English fortune based on the user's name, birth date, and randomly generated lucky color.

Lucky numbers corresponding to colors:
Green(1), Cyan(2), Red(3), Purple(4), Yellow(5), Brown(6), 
White(7), Gold(8), Black(9), Blue(10), Gray(11), Silver(12)

Please generate a personalized fortune within 20 words based on the user's personal information and lucky color. It can be auspicious or inauspicious, but should be concise, interesting, and relevant to the user's information.
Format: Output the fortune content directly without any prefix or explanation.
Use English output only, no Chinese characters allowed.
"""

# Heavenly Stems and Earthly Branches
TIAN_GAN = ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]
DI_ZHI = ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai"]
WU_XING = ["Wood", "Fire", "Earth", "Metal", "Water"]

# Heavenly Stems Five Elements
TIAN_GAN_WU_XING = {
    "Jia": "Wood", "Yi": "Wood",
    "Bing": "Fire", "Ding": "Fire", 
    "Wu": "Earth", "Ji": "Earth",
    "Geng": "Metal", "Xin": "Metal",
    "Ren": "Water", "Gui": "Water"
}

# Earthly Branches Five Elements
DI_ZHI_WU_XING = {
    "Zi": "Water", "Chou": "Earth", "Yin": "Wood", "Mao": "Wood",
    "Chen": "Earth", "Si": "Fire", "Wu": "Fire", "Wei": "Earth",
    "Shen": "Metal", "You": "Metal", "Xu": "Earth", "Hai": "Water"
}

# Chinese to English conversion mappings
CHINESE_MONTHS = {
    "正月": "January", "二月": "February", "三月": "March", "四月": "April",
    "五月": "May", "六月": "June", "七月": "July", "八月": "August",
    "九月": "September", "十月": "October", "十一月": "November", "十二月": "December"
}

CHINESE_DAYS = {
    "初一": "1st", "初二": "2nd", "初三": "3rd", "初四": "4th", "初五": "5th",
    "初六": "6th", "初七": "7th", "初八": "8th", "初九": "9th", "初十": "10th",
    "十一": "11th", "十二": "12th", "十三": "13th", "十四": "14th", "十五": "15th",
    "十六": "16th", "十七": "17th", "十八": "18th", "十九": "19th", "二十": "20th",
    "廿一": "21st", "廿二": "22nd", "廿三": "23rd", "廿四": "24th", "廿五": "25th",
    "廿六": "26th", "廿七": "27th", "廿八": "28th", "廿九": "29th", "三十": "30th"
}

# Chinese GanZhi to English conversion
CHINESE_GANZHI = {
    "甲": "Jia", "乙": "Yi", "丙": "Bing", "丁": "Ding", "戊": "Wu",
    "己": "Ji", "庚": "Geng", "辛": "Xin", "壬": "Ren", "癸": "Gui",
    "子": "Zi", "丑": "Chou", "寅": "Yin", "卯": "Mao", "辰": "Chen",
    "巳": "Si", "午": "Wu", "未": "Wei", "申": "Shen", "酉": "You",
    "戌": "Xu", "亥": "Hai"
}

# Fortune templates (backup)
FORTUNE_TEMPLATES = [
    "Today's wealth fortune is prosperous, suitable for investment and financial management",
    "Be cautious and avoid impulsive spending",
    "Noble people help, career success",
    "Pay attention to health, rest more and stay up late less",
    "Love fortune is flourishing, sweet relationships",
    "Work goes smoothly, promotion and salary increase are promising",
    "Safe travel, pleasant journey",
    "Learning progress, smooth exams",
    "Family harmony, happiness and beauty",
    "Be careful of accidents, pay attention to safety",
    "Poor wealth fortune, live within means",
    "Happy mood, everything goes well"
]

def get_random_lucky_color():
    """Generate random lucky color"""
    return random.choice(list(COLOR_LUCKY_NUMBERS.keys()))

def convert_chinese_to_english(text):
    """Convert Chinese text to English"""
    # Convert months
    for chinese, english in CHINESE_MONTHS.items():
        text = text.replace(chinese, english)
    
    # Convert days
    for chinese, english in CHINESE_DAYS.items():
        text = text.replace(chinese, english)
    
    # Convert GanZhi
    for chinese, english in CHINESE_GANZHI.items():
        text = text.replace(chinese, english)
    
    return text

def convert_solar_to_lunar(birthday_str):
    """Convert solar calendar date to lunar calendar date"""
    try:
        # Parse solar calendar date
        date_obj = datetime.strptime(birthday_str, '%Y-%m-%d')
        
        # Convert to lunar calendar
        lunar = Lunar.fromDate(date_obj)
        
        # Convert Chinese month and day names to English
        month_name = convert_chinese_to_english(lunar.getMonthInChinese())
        day_name = convert_chinese_to_english(lunar.getDayInChinese())
        
        return {
            'year': lunar.getYear(),
            'month': lunar.getMonth(),
            'day': lunar.getDay(),
            'year_gz': lunar.getYearInGanZhi(),  # Year GanZhi
            'month_gz': lunar.getMonthInGanZhi(),  # Month GanZhi
            'day_gz': lunar.getDayInGanZhi(),  # Day GanZhi
            'time_gz': lunar.getTimeInGanZhi(),  # Time GanZhi
            'year_name': lunar.getYearInChinese(),
            'month_name': month_name,
            'day_name': day_name
        }
    except Exception as e:
        print(f"Date conversion failed: {str(e)}")
        return None

def calculate_bazi(lunar_info):
    """Calculate BaZi (Eight Characters)"""
    if not lunar_info:
        return None
    
    try:
        # Extract Four Pillars
        year_gz = lunar_info['year_gz']
        month_gz = lunar_info['month_gz']
        day_gz = lunar_info['day_gz']
        time_gz = lunar_info['time_gz']
        
        # Analyze Five Elements
        year_tiangan = year_gz[:1]
        year_dizhi = year_gz[1:]
        month_tiangan = month_gz[:1]
        month_dizhi = month_gz[1:]
        day_tiangan = day_gz[:1]
        day_dizhi = day_gz[1:]
        time_tiangan = time_gz[:1]
        time_dizhi = time_gz[1:]
        
        # Calculate Five Elements statistics
        wuxing_count = {}
        for gan in [year_tiangan, month_tiangan, day_tiangan, time_tiangan]:
            wuxing = TIAN_GAN_WU_XING.get(gan, "")
            wuxing_count[wuxing] = wuxing_count.get(wuxing, 0) + 1
        
        for zhi in [year_dizhi, month_dizhi, day_dizhi, time_dizhi]:
            wuxing = DI_ZHI_WU_XING.get(zhi, "")
            wuxing_count[wuxing] = wuxing_count.get(wuxing, 0) + 1
        
        # Convert GanZhi to English
        year_gz_en = convert_chinese_to_english(year_gz)
        month_gz_en = convert_chinese_to_english(month_gz)
        day_gz_en = convert_chinese_to_english(day_gz)
        time_gz_en = convert_chinese_to_english(time_gz)
        
        return {
            'bazi': f"{year_gz_en} {month_gz_en} {day_gz_en} {time_gz_en}",
            'year_gz': year_gz_en,
            'month_gz': month_gz_en,
            'day_gz': day_gz_en,
            'time_gz': time_gz_en,
            'wuxing_count': wuxing_count,
            'lunar_date': f"{lunar_info['year_name']} Year {lunar_info['month_name']} Month {lunar_info['day_name']} Day"
        }
    except Exception as e:
        print(f"BaZi calculation failed: {str(e)}")
        return None

def get_fortune_by_color(color, user_name=None, user_birthday=None):
    """Get fortune based on color and user information"""
    try:
        # Construct large model API request
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Build prompt with user information
        user_info = ""
        bazi_info = ""
        
        if user_name and user_birthday:
            # Convert solar to lunar calendar
            lunar_info = convert_solar_to_lunar(user_birthday)
            if lunar_info:
                # Calculate BaZi
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
        
        # Call large model API
        print(f"Generating fortune, lucky color: {color}")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            fortune = result['choices'][0]['message']['content'].strip()
            print(f"Fortune: {fortune}")
            
            # Add newline after each punctuation mark
            fortune_with_newlines = ""
            for char in fortune:
                fortune_with_newlines += char
                if char in '.!?,;:':
                    fortune_with_newlines += '\n'
            print(f"Original fortune: {repr(fortune)}")
            print(f"Fortune with newlines: {repr(fortune_with_newlines)}")
            return fortune_with_newlines
        else:
            print(f"API call failed, using backup template")
            fortune = random.choice(FORTUNE_TEMPLATES)
            # Add newline after each punctuation mark for backup template too
            fortune_with_newlines = ""
            for char in fortune:
                fortune_with_newlines += char
                if char in '.!?,;:':
                    fortune_with_newlines += '\n'
            return fortune_with_newlines
            
    except Exception as e:
        print(f"Fortune generation failed: {str(e)}, using backup template")
        fortune = random.choice(FORTUNE_TEMPLATES)
        # Add newline after each punctuation mark for backup template too
        fortune_with_newlines = ""
        for char in fortune:
            fortune_with_newlines += char
            if char in '.!?,;:':
                fortune_with_newlines += '\n'
        return fortune_with_newlines

@app.route('/send_udp', methods=['POST'])
def send_udp():
    """
    Fortune telling interface
    Randomly generate lucky color, generate lucky number A and fortune B, send via UDP
    """
    try:
        # Get POST data
        data = request.get_data(as_text=True)
        
        if not data:
            return "Error: No data received", 400
        
        # Check if it's JSON format
        try:
            json_data = json.loads(data)
            if 'fortune' in json_data or len(json_data) == 0:
                # Fortune telling mode - randomly generate lucky color
                lucky_color = get_random_lucky_color()
                
                # Get lucky number A
                lucky_number = COLOR_LUCKY_NUMBERS[lucky_color]
                
                # Get user information
                user_name = json_data.get('userName', '')
                user_birthday = json_data.get('userBirthday', '')
                
                # Generate fortune B
                fortune = get_fortune_by_color(lucky_color, user_name, user_birthday)
                
                # Construct UDP message
                N_M = lucky_number
                msg = "A" + str(int(N_M)) + "B" + fortune
                
                # Send UDP packet
                client.sendto(msg.encode('utf-8'), ip_port)
                
                return {
                    "status": "success",
                    "message": "Fortune sent",
                    "lucky_color": lucky_color,
                    "lucky_number": lucky_number,
                    "fortune": fortune
                }
        except json.JSONDecodeError:
            # Not JSON format, send raw data directly
            pass
        
        # Send raw data directly
        N_M = 2
        msg = "A" + str(int(N_M)) + "B" + data
        
        # Send UDP packet
        client.sendto(msg.encode('utf-8'), ip_port)
        
        return "OK"
        
    except requests.exceptions.Timeout:
        return "Error: Large model API request timeout", 500
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed - {str(e)}", 500
    except Exception as e:
        return f"Send failed: {str(e)}", 500

if __name__ == '__main__':
    print("Starting fortune telling service...")
    print(f"Target address: {ip_port[0]}:{ip_port[1]}")
    print(f"Large model API: {API_URL}")
    print("Supported colors:", list(COLOR_LUCKY_NUMBERS.keys()))
    app.run(host='0.0.0.0', port=5003, debug=True)