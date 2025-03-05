import cv2
import mediapipe as mp
import numpy as np
import os

# 初始化MediaPipe Pose模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 初始化绘图工具
mp_drawing = mp.solutions.drawing_utils

# 创建输出目录
output_dir_c = './res/c'
output_dir_s = './res/s'
if not os.path.exists(output_dir_c):
    os.makedirs(output_dir_c)
if not os.path.exists(output_dir_s):
    os.makedirs(output_dir_s)

# 打开视频文件
cap = cv2.VideoCapture('./6.mp4')

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将BGR图像转换为RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像以提取关键点
    results = pose.process(rgb_frame)

    # 如果检测到关键点，绘制它们
    if results.pose_landmarks:
        # 自定义绘制样式
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=10, circle_radius=10),  # 关键点颜色和大小
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=10)  # 连接线颜色和粗细
        )

        # 保存在原图上绘制关键点的图像
        output_path_c = os.path.join(output_dir_c, f'frame_{frame_count:04d}.png')
        cv2.imwrite(output_path_c, frame)

        # 创建一个空白图像用于绘制关键点
        blank_image = np.zeros_like(frame)
        mp_drawing.draw_landmarks(
            blank_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=10, circle_radius=10),  # 关键点颜色和大小
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=10)  # 连接线颜色和粗细
        )

        # 保存只包含关键点的图像
        output_path_s = os.path.join(output_dir_s, f'frame_{frame_count:04d}.png')
        cv2.imwrite(output_path_s, blank_image)

    # 显示结果图像
    cv2.imshow('Pose Estimation', frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# 释放视频捕获对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()

print(f"Processed {frame_count} frames and saved results to {output_dir_c} and {output_dir_s}")