"""
qwen_api.py

OpenCV调用模块, 主要负责进行图像处理

Author: Zhu Jiahao
Date: 2025-07-17
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import base64
import os
import json
from src.utils.config import __config__
from src.utils.logger import __logger__

__all__ = ['OpenCVImageClient']

cv_logger = __logger__.get_module_logger("OpenCV")

class OpenCVImageClient:
    """ OpenCV图像处理类
    """
    def __init__(self,
                camera_id: int):
        """
        初始化

        Args:
            camera_id (int): 摄像头索引
        """
        self.camera_id = camera_id
        self.image_num = 0
        self.cap: Optional[cv2.VideoCapture] = None

    def capture_single_image(self, capture_path: str) -> None:
        """ 打开摄像头并捕获一张图像

        Args:
            capture_path (str): 图像存储路径
        """
        cv_logger.info("正在开启摄像头...")

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            cv_logger.error("无法打开摄像头")
            return
        
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

        cv_logger.info("摄像头已启动, 按空格拍照...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                cv_logger.error("摄像头获取失败")
                break

            preview = cv2.resize(frame, (0, 0), fx=0.33, fy=0.33)
            cv2.imshow("image capture", preview)
            key = cv2.waitKey(1)

            if key == 32:     # SPACE
                rotated = self.__rotate_image(frame, 90, True)
                enhanced = self.__enhance_image(rotated)
                final = self.__process_for_a4(enhanced)

                cv2.imwrite(capture_path, final)
                cv_logger.info(f"图片已保存: {capture_path}")
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def capture_multi_images(self, capture_path: str) -> List:
        """ 打开摄像头并捕获多张图像
        
        Args:
            capture_path (str): 图像存储路径

        Return:
            文件列表
        """
        cv_logger.info("正在开启摄像头...")

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            cv_logger.error("无法打开摄像头")
        
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

        file_list = []
        cv_logger.info("摄像头已启动, 按空格拍照, ESC结束。")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                cv_logger.error("摄像头获取失败")
                break

            preview = cv2.resize(frame, (0, 0), fx=0.33, fy=0.33)
            cv2.imshow("image capture", preview)
            key = cv2.waitKey(1)

            if key == 27:       # ESC
                cv_logger.info("用户退出图像捕获模式")
                break
            elif key == 32:     # SPACE
                self.image_num += 1
                cv_logger.info(f"拍摄了第{self.image_num}页")

                rotated = self.__rotate_image(frame, 90, True)
                enhanced = self.__enhance_image(rotated)
                final = self.__process_for_a4(enhanced)

                filename = os.path.join(capture_path, f"{self.image_num}.jpg")
                cv2.imwrite(filename, final)
                cv_logger.info(f"图片已保存: {filename}")
                file_list.append(filename)

        self.cap.release()
        cv2.destroyAllWindows()
        return file_list

    def load_image_and_get_scale(self, image_path: str) -> Tuple:
        """ 加载图像并计算毫米和像素间的转换比例

        Args:
            image_path (str): 图片路径

        Return:
            tuple: 包含以下元素的元组
                - img (numpy.ndarray): 加载的图像数组
                - img_w (int): 图像宽度(像素)
                - img_h (int): 图像高度(像素)
                - mm_per_pixel_x (float): 水平方向每像素对应的毫米数
                - mm_per_pixel_y (float): 垂直方向每像素对应的毫米数
                - px_per_mm_y (float): 垂直方向每毫米对应的像素数
        """
        # 定义A4纸的标准尺寸(毫米)
        A4_WIDTH_MM = 210
        A4_HEIGHT_MM = 297

        # 加载图像
        img = cv2.imread(image_path)
        if img is None:
            cv_logger.error("未能打开图像...")
            return 
        
        # 获取图像尺寸
        img_h, img_w = img.shape[:2]
        
        mm_per_pixel_x = A4_WIDTH_MM / img_w        
        mm_per_pixel_y = A4_HEIGHT_MM / img_h       
        px_per_mm_y = img_h / A4_HEIGHT_MM          

        return img, img_w, img_h, mm_per_pixel_x, mm_per_pixel_y, px_per_mm_y

    def detect_single_black_box(self, img: np.ndarray, log_path: str, min_area: int=500) -> Tuple:
        """
        检测图像中唯一的黑色闭合矩形框，并将其绘制到本地图片。
        要求图像只存在一个黑框，自动排除小轮廓或噪声。

        Args:
            img (np.ndarray): BGR图像
            log_path (str): 日志存储路径
            min_area (int): 最小有效区域（单位：像素平方），用于排除噪声小框

        Return:
            Tuple(x, y, w, h, area)
        """        
        bin_mask = self.__get_black_mask(img)
        contours = self.__find_external_contours(bin_mask)

        # 过滤掉面积太小的噪声
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h, area))

        # 应用非极大值抑制合并重叠框
        boxes = self.__non_max_suppression(boxes, iou_threshold=0.6)

        if len(boxes) == 0:
            raise ValueError("未检测到黑框，请检查图像质量或阈值设置")
        elif len(boxes) == 1:
            box = boxes[0][:4]
        else:
            # 返回面积最大的一个框，假设它是黑框
            boxes.sort(key=lambda b: b[4], reverse=True)
            box = boxes[0][:4]

        self.__save_boxes_visualization(img, box, log_path)

        return box

    def generate_writing_task(self,
                            img: np.ndarray,
                            box: Tuple,
                            answer: str, 
                            mm_per_pixel_x: float, 
                            mm_per_pixel_y: float,
                            px_per_mm_y: float,
                            preview_path: str,
                            task_path: str) -> None:
        """ 规划书写任务, 并生成预览图

        Args:
            img (np.ndarray): BGR图像
            box (Tuple[int, int, int, int]): 黑框的(x, y, w, h)矩形框坐标
            answer (str): 答案
            log_path (str): 日志保存路径
            mm_per_pixel_x (float): 水平方向每像素对应的毫米数
            mm_per_pixel_y (float): 垂直方向每像素对应的毫米数
            px_per_mm_y (float): 垂直方向每毫米对应的像素数
            preview_path (str): 预览图生成路径
            task_path (str): 任务文件生成路径
        """
        pil_img = Image.fromarray(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font_path = r"C:\Windows\Fonts\simfang.ttf"
        if not os.path.exists(font_path):
            cv_logger.error(f"仿宋字体文件不存在: {font_path}")
            return
        
        writing_tasks = []
        cv_logger.info("")

        x, y, w, h = box
        # 将像素换算成毫米
        box_w_mm = w * mm_per_pixel_x
        box_h_mm = h * mm_per_pixel_y
        # 字号控制
        max_font_mm = min(10.0, box_h_mm, box_w_mm / len(answer) * 1.5 if answer else 10.0)
        min_font_mm = 8.0
        if box_h_mm < min_font_mm: min_font_mm = box_h_mm
        # 换算回像素
        max_font_px = max(int(max_font_mm * px_per_mm_y), 1)    
        min_font_px = max(int(min_font_mm * px_per_mm_y), 1)
        # 尝试合适的字号并自动换行
        text_box_w = int(w * 0.8)
        text_box_h = int(h * 0.95)

        # final_lines, final_font, line_height = [], None, 0
        # for font_px in range(max_font_px, min_font_px - 1, -1):
        #     font = ImageFont.truetype(font_path, font_px)
        #     lines = self.__wrap_text(answer, font, text_box_w, draw)
        #     lh = font.getbbox("汉")[3] - font.getbbox("汉")[1]
        #     if lh * len(lines) <= text_box_h:
        #         final_lines, final_font, line_height = lines, font, lh
        #         break

        # print(max_font_px, max_font_mm, line_height)

        # # 如果没找到合适字号，强行用最小字号
        # if not final_font:
        # 使用固定字号 (不再自适应)
        final_font = ImageFont.truetype(font_path, min_font_px)
        final_lines = self.__wrap_text(answer, final_font, text_box_w, draw)
        line_height = final_font.getbbox("汉")[3] - final_font.getbbox("汉")[1]

        # 计算文字起始位置, 实现居中排版
        # max_line_width = max(draw.textlength(line, font=final_font) for line in final_lines)
        # x_start = x + (w - max_line_width) / 2
        # y_start = y + (h - line_height * len(final_lines)) / 2
        max_line_w_px = max(draw.textbbox((0,0), line, font=final_font)[2] for line in final_lines) if final_lines else 0
        total_text_h_px = line_height * len(final_lines)
        x_start = x + (w - max_line_w_px) / 2
        y_start = y + (h - total_text_h_px) / 3

        # 绘制文字, 保存书写任务
        char_height_mm = (final_font.size / px_per_mm_y) * 0.8  # 估算字符高度
        for i, line in enumerate(final_lines):
            y_line = y_start + i * line_height
            draw.text((x_start, y_line), line, font=final_font, fill=(0, 0, 0))
            writing_tasks.append({
                "text": line,
                "a4_x_mm": x_start * mm_per_pixel_x,
                "a4_y_mm": y_line * mm_per_pixel_y,
                "char_height_mm": char_height_mm,
                "char_spacing_ratio": 1.2
            })

        # 保存预览图
        annotated_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(preview_path, annotated_bgr)
        cv_logger.info(f"预览图已保存至: {preview_path}")
        
        with open(task_path, "w", encoding="utf-8") as f:
            json.dump(writing_tasks, f, ensure_ascii=False, indent=2)
        print(f"✅ 写字任务已保存至: {task_path}")




    """
    =================================================================================
                                    以下是私有函数
    =================================================================================
    """
     
    def __enhance_image(self, image: np.ndarray) -> np.ndarray:
        """ 图像增强: 增加对比度, 突出红色

        Args:
            image (np.ndarray): 输入图像

        Returns:
            np.ndarray: 旋转后的图像
        """
        enhanced_image = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
        hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        red_mask = cv2.inRange(h, 0, 10) + cv2.inRange(h, 170, 180)
        s_red = np.where(red_mask > 0, np.clip(s * 2.0, 0, 255).astype(np.uint8), s)
        non_red_mask = cv2.bitwise_not(red_mask)
        s_combined = np.where(non_red_mask > 0, np.clip(s * 0.6, 0, 255).astype(np.uint8), s_red)

        final_hsv = cv2.merge([h, s_combined, v])
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    def __rotate_image(self, image: np.ndarray, angle: int = 90, clockwise: bool = False) -> np.ndarray:
        """
        旋转图像

        Args:
            image (np.ndarray): 输入图像
            angle (int): 旋转角度，可选值为 90, 180, 270
            clockwise (bool): 是否顺时针旋转，默认 False 表示逆时针

        Returns:
            np.ndarray: 旋转后的图像

        Raises:
            ValueError: 如果角度无效
        """
        angle = angle % 360
        if angle not in (90, 180, 270):
            raise ValueError("仅支持旋转角度为 90, 180, 270")

        # OpenCV 的旋转标志映射
        rotate_map = {
            (90, False): cv2.ROTATE_90_COUNTERCLOCKWISE,
            (90, True): cv2.ROTATE_90_CLOCKWISE,
            (180, False): cv2.ROTATE_180,
            (180, True): cv2.ROTATE_180,
            (270, False): cv2.ROTATE_90_CLOCKWISE,
            (270, True): cv2.ROTATE_90_COUNTERCLOCKWISE,
        }

        rotate_code = rotate_map.get((angle, clockwise))
        return cv2.rotate(image, rotate_code)
    
    def __process_for_a4(self, image: np.ndarray) -> np.ndarray:
        """ 将图像裁剪成A4比例

        Args:
            image (np.ndarray): 输入图像

        Returns:
            np.ndarray: 旋转后的图像
        """
        TARGET_WIDTH = 2100
        TOP_CROP_MM = 62
        BOTTOM_CROP_MM = 10

        h, w, _ = image.shape
        ratio = TARGET_WIDTH / w
        new_height = int(h * ratio)
        resized = cv2.resize(image, (TARGET_WIDTH, new_height), interpolation=cv2.INTER_AREA)

        px_per_mm = TARGET_WIDTH / 210
        top = int(TOP_CROP_MM * px_per_mm)
        bottom = int(BOTTOM_CROP_MM * px_per_mm)

        if top >= new_height - bottom:
            cv_logger.info(f"警告：图像太小无法裁剪，返回原图。")
            return resized
        
        cropped = resized[top:new_height - bottom, :]
        cv_logger.info(f"图片已处理：缩放({TARGET_WIDTH}x{new_height})，裁剪上下({top}px, {bottom}px)")
        return cropped

    def __get_black_mask(self, image: np.ndarray) -> np.ndarray:
        """ 获取图像中的黑色区域掩码

        Args: 
            image (np.ndarray): BGR图像

        Return:
            掩码
        """    
        # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # lower_red1, upper_red1 = np.array([0, 43, 46]), np.array([10, 255, 255])
        # lower_red2, upper_red2 = np.array([156, 43, 46]), np.array([180, 255, 255])
        # mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # mask_clean = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)
        # mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定义黑色的HSV范围
        lower_black = np.array([0, 0, 0])     
        upper_black = np.array([180, 255, 100]) 

        # 生成黑色区域的掩膜
        mask_black = cv2.inRange(hsv, lower_black, upper_black)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_clean = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask_clean
    
    def __find_external_contours(self, mask: np.ndarray) -> list:
        """ 从掩码中提取外部轮廓

        Args:
            mask: 掩码

        Return:
            外部轮廓
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def __save_boxes_visualization(self, img: np.ndarray, box: Tuple, log_path: str) -> None:
        """
        保存画出黑框和角点坐标的中间图像，便于调试和 UI 显示。
        
        参数:
            img (np.ndarray): 原始图像（BGR格式）
            box (Tuple[int, int, int, int]): 黑框的(x, y, w, h)矩形框坐标
            log_path (str): 保存绘图图像的目标目录
        """
        img_vis = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        x, y, w, h = box  # 解包单个矩形框
        cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        coords = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        for cx, cy in coords:
            text = f"({cx},{cy})"
            cv2.putText(img_vis, text, (cx, cy - 5), font, 0.4, (255, 0, 0), 1)

        cv2.imwrite(log_path, img_vis)
        cv_logger.info(f"坐标图保存至: {log_path}")

    def __non_max_suppression(self, boxes, iou_threshold=0.6):
        """
        对输入的边界框列表执行非极大值抑制，合并高度重叠的框。
        与传统的NMS不同，这里我们合并框而不是简单删除，以应对同一个框被分割检测的情况。
        
        Args:
            boxes (list): 格式为 [(x, y, w, h), ...] 的框列表。
            iou_threshold (float): IoU阈值，超过此值的框将被合并。

        Returns:
            list: 合并后的框列表。
        """
        if not boxes:
            return []

        # 将 (x, y, w, h) 转换为 (x1, y1, x2, y2)
        rects = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes])
        
        # 计算面积
        areas = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
        # 按y1坐标排序
        indices = np.argsort(rects[:, 1])

        merged_boxes = []
        while len(indices) > 0:
            last = len(indices) - 1
            i = indices[last]
            
            # 将当前框加入到合并列表中
            current_rect = rects[i]
            current_area = areas[i]  # 保留当前面积
            indices = np.delete(indices, last)

            # 寻找与当前框高度重叠的其他框
            suppress = [last]
            for pos in range(len(indices)):
                j = indices[pos]
                
                # 计算 IoU
                xx1 = np.maximum(current_rect[0], rects[j][0])
                yy1 = np.maximum(current_rect[1], rects[j][1])
                xx2 = np.minimum(current_rect[2], rects[j][2])
                yy2 = np.minimum(current_rect[3], rects[j][3])

                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                
                intersection = w * h
                union = areas[i] + areas[j] - intersection
                iou = intersection / union if union > 0 else 0
                
                # 如果 IoU 超过阈值，则合并这两个框
                if iou > iou_threshold:
                    # 合并框：取两个框的最大外接矩形
                    current_rect[0] = min(current_rect[0], rects[j][0])
                    current_rect[1] = min(current_rect[1], rects[j][1])
                    current_rect[2] = max(current_rect[2], rects[j][2])
                    current_rect[3] = max(current_rect[3], rects[j][3])
                    current_area += areas[j]  # 累加合并的面积
                    
                    # 标记此框，以便后续删除
                    suppress.append(pos)
            
            # 从索引中删除已被合并的框
            indices = np.delete(indices, [s for s in suppress if s != last])
            
            # 将合并后的大框 (x1, y1, x2, y2) 转换回 (x, y, w, h)
            merged_w = current_rect[2] - current_rect[0]
            merged_h = current_rect[3] - current_rect[1]
            merged_boxes.append((int(current_rect[0]), int(current_rect[1]), int(merged_w), int(merged_h), int(current_area)))
            
        cv_logger.info(f"📦 非极大值抑制：原始检测到 {len(boxes)} 个框，合并后剩余 {len(merged_boxes)} 个框。")
        return merged_boxes

    def __wrap_text(self, text: str, font: str, max_width: float, draw):
        """将文本在指定宽度内自动换行
        
        Args:
            text (str): 要书写的文本
            font (ImageFont): 
        """
        lines, current = [], ""
        for ch in text:
            # 使用 getbbox 计算当前文本宽度
            bbox = draw.textbbox((0, 0), current + ch, font=font)
            text_width = bbox[2] - bbox[0]  # x_max - x_min
            
            if text_width > max_width:
                lines.append(current)
                current = ch
            else:
                current += ch
        
        if current:
            lines.append(current)
        
        return lines

