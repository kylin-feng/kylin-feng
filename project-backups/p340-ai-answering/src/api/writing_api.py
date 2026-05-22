"""
writing_api.py

机械臂书写API

Author: Zhu Jiahao
Date: 2025-07-18
"""

import pickle
import time
import json
from pymycobot.ultraArmP340 import ultraArmP340
from HersheyFonts import HersheyFonts
from src.utils.config import __config__
from src.utils.logger import __logger__

__all__ = ['RobotWritingClient']

writing_logger = __logger__.get_module_logger("Writing")

# ASCII码偏移量
ASCII_OFFSET_X_FACTOR = -0.7
ASCII_OFFSET_Y_FACTOR = -0.23

# 全角标点映射
PUNCTUATION_MAP = {
    '，': ',', '。': '.', '“': '"', '”': '"', '：': ':', 
    '；': ';', '？': '?', '！': '!', '（': '(', '）': ')',
    '《': '<', '》': '>', '‘': "'", '’': "'", '、': ','
}

class RobotWritingClient:
    """机器人书写服务类
    """
    def __init__(self,
                com_port: str, 
                baudrate: int, 
                z_up: float, 
                z_down: float, 
                speed_move: int, 
                speed_write: int, 
                origin_x: float,
                origin_y: float,
                chinese_font_path: str):
        """
        初始化

        Args:
            com_port (str): 端口号
            baudrate (int): 波特率
            z_up (float): 抬笔高度
            z_down (float): 落笔高度
            origin_x (float): 机械臂默认X坐标 (即机械臂位于A4纸[0, 0]时的X坐标)
            origin_y (float): 机械臂默认Y坐标 (即机械臂位于A4纸[0, 0]时的Y坐标)
            speed_move (int): 移动画笔的速度
            speed_write (int): 写字的速度    
        """
        self.z_up = z_up
        self.z_down = z_down
        self.speed_move = speed_move
        self.speed_write = speed_write
        self.origin_x = origin_x
        self.origin_y = origin_y

        try:
            self.ua = ultraArmP340(com_port, baudrate)
        except:
            writing_logger.error("机器人无法连接")
            exit()
        
        try:
            with open(chinese_font_path, "rb") as f:
                self.chinese_font = pickle.load(f)
        except FileNotFoundError:
            writing_logger.error("找不到中文字体")
            exit()

        writing_logger.info("机器人正在回零...")
        self.ua.go_zero()
        self.ua.set_speed_mode(2)
        time.sleep(0.1)
        self.stand_by()
        writing_logger.info("机器人书写服务部署完成...")

    # def calibrate_origin(self) -> None:
    #     """
    #     试卷位置校准: 
    #         1. 将机械臂移动到origin_x, origin_y处
    #         2. 手动将试卷左上角对齐笔尖位置
    #     """
    #     # self.ua.set_coords([self.origin_x, self.origin_y, self.z_up], self.speed_move)
    #     writing_logger.info("开始进行位置校准, 请等待机械臂运动完毕, 并将试卷左上角对齐笔尖位置")
    #     writing_logger.info("机械臂正在归位...")
    #     self.__move_sync(self.origin_x, self.origin_y)
    #     input("请对齐试卷，按下回车键继续...")
    #     writing_logger.info("已完成校准")
        
    def stand_by(self) -> None:
        """控制机械臂回到待机位置
        """
        writing_logger.info("机械臂正在归位...")
        self.ua.set_angles([90, 0, 0], self.speed_move)
        # 控制误差在±0.1
        while True:
            angles = self.ua.get_angles_info()
            if abs(angles[0] - 90) <= 0.1 and abs(angles[1]) <= 0.1 and abs(abs(angles[1]) <= 0.1):
                break
            time.sleep(0.02)
        writing_logger.info("机械臂已归位")

    def go_center(self) -> None:
        """ 控制机械臂前往A4纸中心
        """
        writing_logger.info("机械臂正在前往纸张中心...")
        self.ua.set_angles([0, 0, 0], self.speed_move)
        time.sleep(0.1)
        self.ua.set_coords([235.55, 0, self.z_up], self.speed_move)
        # 控制误差在±0.1
        while True:
            coords = self.ua.get_coords_info()
            if abs(coords[0] - 235.55) <= 0.1 and abs(coords[1]) <= 0.1:
                break
            time.sleep(0.02)
        writing_logger.info("机械臂已到达纸张中心")

    def write_chinese_char(self, ch: str, center_x: float, center_y: float, height: float) -> None:
        """
        写一个中文汉字到指定位置

        Args:
            ch (str): 要写的汉字
            center_x (float): 该汉字中心点x坐标 (mm)
            center_y (float): 该汉字中心点y坐标 (mm)
            height (float): 字体高度 (mm)
        """
        # 1. 检查并获取字符数据
        if ch not in self.chinese_font:
            return
        strokes = self.chinese_font[ch]
        # print(strokes)

        # 2. 确定缩放比例
        char_spacing_base = 70.0
        x_ratio = 1.0
        y_ratio = 0.8
        scale_x = height / char_spacing_base * x_ratio
        scale_y = height / char_spacing_base * y_ratio

        # 3. 逐笔画绘制
        for stroke in strokes:
            # 找到该笔画的第一个点
            x0_raw = stroke[0]["y"] / 10.0
            y0_raw = stroke[0]["x"] / 10.0 - char_spacing_base
            # 将该点转换为机械臂坐标
            px0 = center_x - x0_raw * scale_x
            py0 = center_y + y0_raw * scale_y
            # 将笔移动到起笔点 (空中)
            # self.ua.set_coords([px0, py0, self.z_up], self.speed_move)
            self.__move_sync(px0, py0)
            # 笔下落开始绘制
            # self.ua.set_coords([px0, py0, self.z_down], self.speed_write)
            self.__write_sync(px0, py0)
            # 继续绘制这一笔的后续点
            for pt in stroke[1:]:
                x_raw = pt["y"] / 10.0
                y_raw = pt["x"] / 10.0 - char_spacing_base
                px = center_x - x_raw * scale_x
                py = center_y + y_raw * scale_y
                # self.ua.set_coords([px, py, self.z_down], self.speed_write)
                self.__write_sync(px, py)

        # 4. 写完一个字，提起笔
        self.ua.set_coord("z", self.z_up, self.speed_move)

    def write_ascii_char(self, ch: str, center_x: float, center_y: float, height: float) -> None:
        """
        写一个ASCII字符到指定位置, 用于处理英文字母, 数字, 符号等

        Args:
            ch (str): 要写的汉字
            center_x (float): 该汉字中心点x坐标 (mm)
            center_y (float): 该汉字中心点y坐标 (mm)
            height (float): 字体高度 (mm)
        """
        # 1. 加载字体对象并初始化
        hf = HersheyFonts()
        hf.load_default_font("futural")
        # 2. 定义字体缩放规范
        ascii_unit_height = 100.0
        hf.normalize_rendering(ascii_unit_height)
        ascii_scale = height / ascii_unit_height
        # 3. 获取字符线段数据
        raw_segments = hf.lines_for_text(ch)
        if not raw_segments: return
        # 4. 将线段合并为连续路径
        paths = self.__merge_segments_to_paths(raw_segments)
        if not paths: return
        # 5. 对齐
        all_x_raw = [pt[0] for path_item in paths for pt in path_item]
        if not all_x_raw: return
        center_offset_x_raw = (min(all_x_raw) + max(all_x_raw)) / 2.0
        # 6. 调整offset
        ascii_offset_x = height * ASCII_OFFSET_X_FACTOR
        ascii_offset_y = height * ASCII_OFFSET_Y_FACTOR
        # 7. 遍历每条路径，逐路径绘图
        for path in paths:
            # 计算起点
            x_h0, y_h0 = path[0]
            x_adj_raw0 = x_h0 - center_offset_x_raw
            x_pen_offset0 = (y_h0 * ascii_scale) - (height / 2.0)
            y_pen_offset0 = x_adj_raw0 * ascii_scale
            px0 = center_x + x_pen_offset0 + ascii_offset_x
            py0 = center_y + y_pen_offset0 + ascii_offset_y
            # 落笔
            self.ua.set_coords([px0, py0, self.z_up], self.speed_move)
            self.ua.set_coords([px0, py0, self.z_down], self.speed_write)
            time.sleep(0.01)
            # 绘制后续所有路径
            for (x_h, y_h) in path[1:]:
                x_adj_raw = x_h - center_offset_x_raw
                x_pen_offset = (y_h * ascii_scale) - (height / 2.0)
                y_pen_offset = x_adj_raw * ascii_scale
                px = center_x + x_pen_offset + ascii_offset_x
                py = center_y + y_pen_offset + ascii_offset_y
                self.ua.set_coords([px, py, self.z_down], self.speed_write)
                time.sleep(0.01)
            
        # 8. 写完收笔
        self.ua.set_coord("z", self.z_up, self.speed_move)
        time.sleep(0.1)

    def write_text_line(self, text: str, start_x: float, start_y: float, height: float, spacing_ratio: float) -> None:
        """
        从指定位置开始, 写一行文本

        Args:
            text (str): 要写的文本
            start_x (float): 起始点的X坐标 (A4纸坐标)
            start_y (float): 起始点的Y坐标 (A4纸坐标)
            height (float): 字体高度
            spacing_ratio (float): 字符间水平间隔比例
        """
        writing_logger.info(f"正在书写: '{text}', 书写起点位置(A4纸坐标): [{start_x:.2f}, {start_y:.2f}]")
        current_a4_x_offset = 0

        # 遍历字符
        for char in text:
            # 1. 处理全角标点映射
            is_full_width_punct = char in PUNCTUATION_MAP
            char_to_write = PUNCTUATION_MAP.get(char, char)
            # 2. 字符类型判断
            is_chinese = char_to_write in self.chinese_font
            is_space = char_to_write == ' '
            is_ascii = char_to_write.isascii() and char_to_write.isprintable() and not is_space and not is_chinese
            # 3. 计算字符宽度
            width = height if (is_chinese or is_full_width_punct) else height / 2
            # 4. 跳过空格
            if is_space:
                current_a4_x_offset += width * spacing_ratio
                continue
            # 5. 坐标转换
            center_start_x = start_x + current_a4_x_offset + width / 2.0
            center_start_y = start_y + height / 2.0
            # print(center_start_x, center_start_y)
            robot_x, robot_y = self.__a4_to_robot_coords(center_start_x, center_start_y)

            # 6. 调用绘制函数
            if is_chinese:
                self.write_chinese_char(char_to_write, robot_x, robot_y, height)
            elif is_ascii:
                self.write_ascii_char(char_to_write, robot_x, robot_y, height)
            else:
                writing_logger.warning(f"出现了无法识别的字符: {char}")
            # 7. 更新偏移量
            offset = width * spacing_ratio
            current_a4_x_offset += offset

    def load_writing_tasks(self, json_path) -> list:
        """
        读取写字任务 JSON 文件并返回任务列表。

        Args:
            json_path (str): JSON 文件的路径

        Returns:
            List[dict]: 写字任务的列表，每个任务是一个包含文字内容和坐标的字典
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                tasks = json.load(f)
            writing_logger.info(f"✅ 成功读取写字任务: {json_path}")
            return tasks
        except FileNotFoundError:
            writing_logger.error(f"❌ 文件未找到: {json_path}")
            return []
        except json.JSONDecodeError as e:
            writing_logger.error(f"❌ JSON 解码失败: {e}")
            return []

    def __move_sync(self, target_coords_x: float, target_coords_y: float, timeout: float = 5.0):
        """同步控制机械臂移动 (非写字状态)，带超时控制
        
        Args:
            target_coords_x (float): 目标坐标X (机械臂坐标)
            target_coords_y (float): 目标坐标Y (机械臂坐标)
            timeout (float): 超时时间, 默认5秒
        """
        self.ua.set_coords([target_coords_x, target_coords_y, self.z_up], self.speed_move)
        
        start_time = time.time()
        while True:
            # 检查是否超时
            if time.time() - start_time > timeout:
                writing_logger.error("机械臂运动存在误差, 请检查...")
                break
            
            coords = self.ua.get_coords_info()
            if abs(coords[0] - target_coords_x) <= 0.1 and abs(coords[1] - target_coords_y) <= 0.1:
                break
            time.sleep(0.02)

    def __write_sync(self, target_coords_x: float, target_coords_y: float, timeout: float = 5.0):
        """同步控制机械臂移动 (写字状态)，带超时控制
        
        Args:
            target_coords_x (float): 目标坐标X (机械臂坐标)
            target_coords_y (float): 目标坐标Y (机械臂坐标)
            timeout (float): 超时时间, 默认5秒
        """
        self.ua.set_coords([target_coords_x, target_coords_y, self.z_down], self.speed_write)
        
        start_time = time.time()
        while True:
            # 检查是否超时
            if time.time() - start_time > timeout:
                writing_logger.error("机械臂运动存在误差, 请检查...")
                break
            
            coords = self.ua.get_coords_info()
            if abs(coords[0] - target_coords_x) <= 0.1 and abs(coords[1] - target_coords_y) <= 0.1:
                break
            time.sleep(0.02)

    def __merge_segments_to_paths(self, segments):
        """将一堆线段首尾拼接成连续的路径, 是write_ascii_char函数的子函数

        Examples:
            Input:
                segments = [
                    ((1, 1), (2, 2)),
                    ((2, 2), (3, 3)),
                    ((4, 4), (5, 5)),
                ]
            Output:
                [
                    [(1, 1), (2, 2), (3, 3)],
                    [(4, 4), (5, 5)]
                ]
        """
        segs = [((x1, y1), (x2, y2)) for ((x1, y1), (x2, y2)) in segments]
        paths = []
        # 不断从segs中取出线段并合并成路径
        while segs:
            (start, end) = segs.pop(0)
            path = [start, end]
            # 1. 向后拓展路径
            extended = True
            while extended:
                extended = False
                for idx, ((x1, y1), (x2, y2)) in enumerate(segs):
                    if (x1, y1) == path[-1]: path.append((x2, y2)); segs.pop(idx); extended = True; break
                    elif (x2, y2) == path[-1]: path.append((x1, y1)); segs.pop(idx); extended = True; break
            # 2. 向前拓展路径
            extended = True
            while extended:
                extended = False
                for idx, ((x1, y1), (x2, y2)) in enumerate(segs):
                    if (x2, y2) == path[0]: path.insert(0, (x1, y1)); segs.pop(idx); extended = True; break
                    elif (x1, y1) == path[0]: path.insert(0, (x2, y2)); segs.pop(idx); extended = True; break
            paths.append(path)
        return paths

    def __a4_to_robot_coords(self, a4_x_mm: float, a4_y_mm: float):
        """
        将A4坐标系的点转换为机械臂坐标系的点

        Args:
            a4_x_mm (float): A4上的x坐标
            a4_y_mm (float): A4上的y坐标

        Returns:
            robot_x, robot_y (float): 对应的机械臂坐标
        """
        robot_x = self.origin_x - a4_y_mm
        robot_y = self.origin_y + a4_x_mm

        return robot_x, robot_y
