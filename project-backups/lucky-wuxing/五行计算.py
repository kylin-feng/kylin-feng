# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 14:48:58 2025

@author: 32118
"""

from flask import Flask, request, jsonify, Response
from lunar_python import Lunar, Solar
from collections import Counter
import random
import json
import requests


def calculate_bazi(year, month, day, hour, minute):

    # 验证必填参数
    if None in [year, month, day]:
        return jsonify({
            "error": "Missing required parameters",
            "message": "Year, month and day are required"
        }), 400
    
    # 创建阳历日期对象
    solar = Solar.fromYmdHms(year, month, day, hour, minute, 0)
    # 转换为农历日期
    bazi = solar.getLunar()

    # 构建响应数据
    result = {
        "年柱": bazi.getYearInGanZhi(),
        "月柱": bazi.getMonthInGanZhi(),
        "日柱": bazi.getDayInGanZhi(),
        "时柱": bazi.getTimeInGanZhi(),
        "八字": bazi.getEightChar().toString(),
        "十神": {
            "年柱": bazi.getBaZiShiShenYearZhi()[0],
            "月柱": bazi.getBaZiShiShenMonthZhi()[0],
            "日柱": bazi.getBaZiShiShenDayZhi()[0],
            "时柱": bazi.getBaZiShiShenTimeZhi()[0],
            }
    }
    
    colors_and_lines = {
        "幸运色": get_lucky_colors(result),
        "适宜和忌讳": get_lines(result)
        }
    
    output = {**result, **colors_and_lines}
    
    return output

def get_lucky_colors(result):
    gan_zhi_list = [result["年柱"], result["月柱"], result["日柱"], result["时柱"]]

    wu_xing_map = {
        '甲': '木', '乙': '木', '寅': '木', '卯': '木',
        '丙': '火', '丁': '火', '巳': '火', '午': '火',
        '戊': '土', '己': '土', '辰': '土', '戌': '土', '丑': '土', '未': '土',
        '庚': '金', '辛': '金', '申': '金', '酉': '金',
        '壬': '水', '癸': '水', '亥': '水', '子': '水',
    }

    wu_xing_elements = []
    for gan_zhi in gan_zhi_list:
        for char in gan_zhi:
            if char in wu_xing_map:
                wu_xing_elements.append(wu_xing_map[char])
                
    wu_xing_count = Counter(wu_xing_elements)

    all_elements = ['木', '火', '土', '金', '水']
    missing_or_weak = [e for e in all_elements if wu_xing_count.get(e, 0) < 1]
    strong_elements = [e for e in all_elements if wu_xing_count.get(e, 0) >= 2]

    wu_xing_ke = {
        '木': '土',  # 木克土
        '火': '金',  # 火克金
        '土': '水',  # 土克水
        '金': '木',  # 金克木
        '水': '火',  # 水克火
    }

    wu_xing_xie = {
        '木': '火',  # 木生火（木被火泄）
        '火': '土',  # 火生土（火被土泄）
        '土': '金',  # 土生金（土被金泄）
        '金': '水',  # 金生水（金被水泄）
        '水': '木',  # 水生木（水被木泄）
    }

    color_map = {
        '木': random.choice(["绿色","青色"]),
        '火': random.choice(["红色","紫色"]),
        '土': random.choice(["黄色","棕色"]),
        '金': random.choice(["白色","金色"]),
        '水': random.choice(["黑色","蓝色"]),
    }
    
    color_list = ["绿色","青色","红色","紫色", "黄色","棕色","白色","金色","黑色","蓝色", '灰色', '银色']

    lucky_colors = []

    # 1. 优先补缺失或弱的五行（最多2个）
    for element in missing_or_weak[:2]:
        lucky_colors.append(color_map[element])

    # 2. 如果还有空位（<3），则克过旺的五行
    if len(lucky_colors) < 3:
        for element in strong_elements:
            ke_element = wu_xing_ke[element]  # 找到能克它的五行
            # 如果克它的五行不弱，并且颜色未重复
            if (ke_element not in missing_or_weak and 
                color_map[ke_element] not in lucky_colors):
                lucky_colors.append(color_map[ke_element])
                if len(lucky_colors) >= 3:
                    break

    # 3. 如果仍然不足3个，用泄过旺的五行
    if len(lucky_colors) < 3:
        for element in strong_elements:
            xie_element = wu_xing_xie[element]  # 找到能泄它的五行
            # 如果泄它的五行不弱，并且颜色未重复
            if (xie_element not in missing_or_weak and 
                color_map[xie_element] not in lucky_colors):
                lucky_colors.append(color_map[xie_element])
                if len(lucky_colors) >= 3:
                    break

    # 如果仍然不足3个，补充中性色（如灰色、紫色）
    if len(lucky_colors) < 3:
        neutral_colors = ['灰色', '银色', '紫色']
        for color in neutral_colors:
            if color not in lucky_colors:
                lucky_colors.append(color)
                if len(lucky_colors) >= 3:
                    break
    
    color_index = []
    for color in lucky_colors:
        color_index.append(str(color_list.index(color) + 1))
    
    return color_index

def get_lines(result):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    api_key = "your-siliconflow-api-key"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    bazi = result["八字"]
    year = result["年柱"]
    month = result["月柱"]
    day = result["日柱"]
    time = result["时柱"]
    
    prompt = f"以下是五行八卦测算得出的信息：八字：{bazi}，年柱：{year}，月柱：{month}，日柱：{day}，时柱：{time}。请在<适宜>标签内写下今日适宜的一句话，在<忌讳>标签内写下今日忌讳的一句话。语句应简洁明了、符合逻辑且具有一定的实用性。"

    
    data = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [
            {"role": "system", "content": "你的任务是根据五行八卦测出来的八字、年柱、月柱、日柱和时柱，生成今日适宜和忌讳各一句话。"},
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    # 打印响应
    lines = response.json()["choices"][0]["message"]["content"].replace("\n", "").replace("<适宜>", "").replace("<忌讳>", "").replace("</适宜>", "").replace("</忌讳>", "")
    
    return lines