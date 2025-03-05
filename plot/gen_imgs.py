import cv2
import os

# 创建输出目录
output_dir = './imgs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

    # 保存每一帧为图片
    output_path = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
    cv2.imwrite(output_path, frame)

    frame_count += 1

# 释放视频捕获对象
cap.release()

print(f"Processed {frame_count} frames and saved results to {output_dir}")