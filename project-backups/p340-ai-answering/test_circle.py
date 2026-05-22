"""
让机械臂在空中画一个圆
"""
import time
import math
from pymycobot.ultraArmP340 import ultraArmP340

COM_PORT = "/dev/cu.usbserial-1140"
BAUDRATE = 115200

# 圆心坐标（机械臂坐标系，z_up=-65 悬空）
CENTER_X = 235.0
CENTER_Y = 0.0
Z_HEIGHT = -65
RADIUS = 40       # 圆半径 mm
STEPS = 36        # 步数（每步10度）
SPEED = 40

print("连接机械臂...")
ua = ultraArmP340(COM_PORT, BAUDRATE)

print("回零...")
ua.go_zero()
ua.set_speed_mode(2)
time.sleep(2)

print("移动到圆起点...")
ua.set_coords([CENTER_X + RADIUS, CENTER_Y, Z_HEIGHT], SPEED)
time.sleep(2)

print("开始画圆...")
for i in range(STEPS + 1):
    angle = 2 * math.pi * i / STEPS
    x = CENTER_X + RADIUS * math.cos(angle)
    y = CENTER_Y + RADIUS * math.sin(angle)
    ua.set_coords([x, y, Z_HEIGHT], SPEED)
    time.sleep(0.3)

print("画圆完成，回待机位...")
ua.set_angles([90, 0, 0], SPEED)
time.sleep(2)
print("完成！")
