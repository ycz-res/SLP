import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import numpy as np
from pathlib import Path
import os
import json
from utils import load_json, save_json  # 自定义工具函数，假设你有这些模块


def process(src_path, res_path):
    # 检查并创建输出文件夹
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # 初始化 CLIP 模型和处理器
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # 模型输出适配 (模型维度、适配维度)
    clip_output_transform = nn.Linear(512, 274)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 定义情感倾向及其文本描述
    emotions = ['positive', 'negative']
    text_inputs = processor(text=emotions, return_tensors="pt", padding=True)

    # 获取文本特征
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = clip_output_transform(text_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        print('text_features.shape: ', text_features.shape)

    # 处理每个 JSON 文件
    path = Path(src_path)
    for file in path.rglob('*.json'):
        keypoint_s = load_json(file)
        keypoint_s = np.array(keypoint_s)
        print('keypoint_s.shape: ', keypoint_s.shape)

        if keypoint_s.ndim != 2:  # 确保关键点格式为 [帧数, 维度]
            print(f"文件 {file} 的关键点数据格式不正确，跳过处理。")
            continue

        # 处理每一帧的关键点
        updated_keypoint_s = []
        for frame in keypoint_s:
            frame_features = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
            frame_features = frame_features / frame_features.norm(dim=-1, keepdim=True)  # 归一化

            with torch.no_grad():
                similarity = (frame_features @ text_features.T).softmax(dim=-1)  # 计算情感概率

            positive_prob = similarity[0, 0].item()  # 取出正向情感概率

            # 添加情感概率到当前帧
            updated_frame = np.hstack([frame, [positive_prob]])
            updated_keypoint_s.append(updated_frame)

        updated_keypoint_s = np.array(updated_keypoint_s)  # 转换为 NumPy 数组
        print('updated_keypoint_s.shape: ', updated_keypoint_s.shape)
        print(updated_keypoint_s[0][274])

        # 保存数据
        save_path = os.path.join(res_path, file.name)
        save_json(save_path, updated_keypoint_s.tolist())
        print(f"{file.name} 已经处理完成并保存到: {save_path}")


if __name__ == '__main__':
    # 处理前修改 src_path（待处理数据目录） 和 res_path（结果存放目录）
    process('../keypoints', '../res')
