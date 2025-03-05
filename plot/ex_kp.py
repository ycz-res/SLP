import cv2
import mediapipe as mp
import os

# 初始化 MediaPipe Pose 模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 初始化绘图工具
mp_drawing = mp.solutions.drawing_utils

# 打开视频文件
video_path = './1.mp4'  # 视频文件路径
cap = cv2.VideoCapture(video_path)

# 创建保存关键点图像的文件夹
output_folder = './res'  # 输出路径
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像从 BGR 转换为 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像以获取关键点
    result = pose.process(rgb_frame)

    # 如果检测到关键点，则在图像上绘制关键点
    if result.pose_landmarks:
        annotated_frame = frame.copy()
        mp_drawing.draw_landmarks(
            annotated_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # 保存关键点图像
        output_path = os.path.join(output_folder, f'keypoint_frame_{frame_count:04d}.png')
        cv2.imwrite(output_path, annotated_frame)
        print(f'Saved keypoint image to {output_path}')

    frame_count += 1

cap.release()
print('Finished processing video.')