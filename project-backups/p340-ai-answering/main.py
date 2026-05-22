"""
main.py
运行整个项目完整的流程管线

Author: Zhu Jiahao
Date: 2025-07-14
"""

import os
from src.api.image_api import OpenCVImageClient
from src.api.qwen_api import QwenClient
from src.api.deepseek_api import DeepSeekClient
from src.api.writing_api import RobotWritingClient
from src.utils.utils import read_txt_file, format_text_to_json
from src.utils.config import __config__
from src.utils.logger import __logger__

def main():
    """
    @TODO: Describe the whole pipeline
    """
    pipeline_logger = __logger__.get_module_logger("pipeline")

    # 初始化
    path_config = __config__.get_path_config()
    camera_config = __config__.get_camera_config()
    qwen_config = __config__.get_api_config("qwen")
    qwen_vl_config = __config__.get_api_config("qwen_vl")
    deepseek_config = __config__.get_api_config("deepseek")
    robot_config = __config__.get_robot_config()
    assets_confog = __config__.get_assets_config()

    # 文件路径
    INPUT_IMAGE_PATH = path_config.get("input", {}).get("images")               # 输入照片路径
    OUTPUT_LOG_PATH = path_config.get("output", {}).get("logs")                 # 输出日志路径
    # OUTPUT_UNIT_PATH = path_config.get("output", {}).get("units")               # 单元分割结果输出路径

    IMAGE_FILENAME = os.path.join(INPUT_IMAGE_PATH, "raw_image.jpg")                    # 原始图像文件
    OCR_FILENAME = os.path.join(OUTPUT_LOG_PATH, "ocr_result.txt")                      # OCR结果文件
    ANSWER_FILENAME = os.path.join(OUTPUT_LOG_PATH, "answer.txt")                       # AI答案文件
    BOX_VIZ_IMAGE_FILENAME = os.path.join(OUTPUT_LOG_PATH, "box_viz_image.png")         # 标注答题框的图片
    PREVIEW_IMAGE_FILENAME = os.path.join(OUTPUT_LOG_PATH, "preview.png")               # 预览图
    TASK_FILENAME = os.path.join(OUTPUT_LOG_PATH, "task.json")                          # 任务编排

    image_client = OpenCVImageClient(
        camera_config.get("id")
    )

    robot_writer = RobotWritingClient(
        robot_config.get("com_port"),
        robot_config.get("baudrate"),
        robot_config.get("z_up"),
        robot_config.get("z_down"),
        robot_config.get("speed_move"),
        robot_config.get("speed_write"),
        robot_config.get("origin_x"),
        robot_config.get("origin_y"),
        assets_confog.get("chinese_fonts")
    )

    # qwen_client = None
    # deepseek_client = None
    qwen_client = QwenClient(
        api_key=qwen_config.get("api_key"),
        base_url=qwen_config.get("base_url"),
        vl_model=qwen_vl_config.get("model"),
        text_model=qwen_config.get("model")
    )

    deepseek_client = DeepSeekClient(
        api_key=deepseek_config.get("api_key"),
        base_url=deepseek_config.get("base_url"),
        model=deepseek_config.get("model")
    )

    print("请选择操作类型: ")
    print("[1] 直接书写")
    print("[2] AI答题")
    
    strategy = input()

    if strategy == "1":
        print("请输入想要书写的文本: ")
        text = input()
        format_text_to_json(text, TASK_FILENAME)
        robot_writer.go_center()
        tasks = robot_writer.load_writing_tasks(TASK_FILENAME)
        for task in tasks:
            robot_writer.write_text_line(
                task.get("text"),
                task.get("a4_x_mm"),
                task.get("a4_y_mm"),
                task.get("char_height_mm"),
                task.get("char_spacing_ratio")
            )
        robot_writer.stand_by()


    if strategy == "2":
        pipeline_logger.info("=====================================================")
        pipeline_logger.info("===               Start the Pipeline              ===")
        pipeline_logger.info("=====================================================")
        pipeline_logger.info("")

        # Step 1: 捕获图片
        pipeline_logger.info("=====================================================")
        pipeline_logger.info("===              Step1: Capture Image             ===")
        pipeline_logger.info("=====================================================")
        pipeline_logger.info("")

        image_client.capture_single_image(IMAGE_FILENAME)           # 试卷实体 -> IMAGE

        # Step 2: OCR生成文本
        pipeline_logger.info("=====================================================")
        pipeline_logger.info("===                Step2: OCR Image               ===")
        pipeline_logger.info("=====================================================")
        pipeline_logger.info("")

        qwen_client.ocr_image(IMAGE_FILENAME, OCR_FILENAME)         # IMAGE -> OCR_TXT

        # Step 3: AI生成答案
        pipeline_logger.info("=====================================================")
        pipeline_logger.info("===            Step3: Answer Generation           ===")
        pipeline_logger.info("=====================================================")
        pipeline_logger.info("")

        deepseek_client.answer_reasoning_question(OCR_FILENAME, ANSWER_FILENAME)        # OCR_TXT -> ANSWER_TXT
        
        

        # Step 4: 位置映射
        pipeline_logger.info("=====================================================")
        pipeline_logger.info("===             Step4: Position Mapping           ===")
        pipeline_logger.info("=====================================================")
        pipeline_logger.info("")

        answer = read_txt_file(ANSWER_FILENAME)
        img, img_w, img_h, mm_per_pixel_x, mm_per_pixel_y, px_per_mm_y = image_client.load_image_and_get_scale(IMAGE_FILENAME)
        box = image_client.detect_single_black_box(img, BOX_VIZ_IMAGE_FILENAME)
        image_client.generate_writing_task(img, box, answer, mm_per_pixel_x, mm_per_pixel_y,
                                        px_per_mm_y, PREVIEW_IMAGE_FILENAME, TASK_FILENAME)      # ANSWER_TXT -> TASK_JSON
        

        # Step 5: 机械臂书写
        pipeline_logger.info("=====================================================")
        pipeline_logger.info("===              Step5: Robot Writing             ===")
        pipeline_logger.info("=====================================================")
        pipeline_logger.info("")

        robot_writer.go_center()
        tasks = robot_writer.load_writing_tasks(TASK_FILENAME)
        for task in tasks:
            robot_writer.write_text_line(
                task.get("text"),
                task.get("a4_x_mm"),
                task.get("a4_y_mm"),
                task.get("char_height_mm"),
                task.get("char_spacing_ratio")
            )

        pipeline_logger.info("=====================================================")
        pipeline_logger.info("===               Pipeline Finished               ===")
        pipeline_logger.info("=====================================================")
        robot_writer.stand_by()



if __name__ == "__main__":
    main()