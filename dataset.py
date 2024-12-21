import os
import re
import torch
import cv2
import utils
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from vidaug import augmentors as va
from augmentation import *
import numpy as np


# How2Sign数据集
class How2SignDataset(Dataset):
    def __init__(self, path, tokenizer, config, args, phase, training_refurbish=False):
        self.dataset = utils.load_h2s_dataset(path)

        self.tokenizer = tokenizer
        self.config = config
        self.args = args
        self.phase = phase
        self.training_refurbish = training_refurbish

        self.keypoints_dir = config[self.args['dataset']]['keypoints_dir']
        self.max_length = config[self.args['dataset']]['max_length']

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        src_sample = sample['text']
        keypoints_name = sample['keypoints_name']

        tgt_sample = []

        return src_sample, tgt_sample

    def _load_keypoints(self, path):
        data = utils.load_json(path)
        video_vectors = [frame_data['bodies'][0]['joints'] for frame_data in data.values()]
        # print("video_vectors[0]:", len(video_vectors[0]))
        # 如果关键点向量数量超过最大长度，随机抽取最大长度的关键点向量，并保持顺序
        if len(video_vectors) > self.max_length:
            video_vectors = [video_vectors[i] for i in
                             sorted(random.sample(range(len(video_vectors)), self.max_length))]
        return video_vectors

    def collate_fn(self, batch):
        imgs_batch_tmp, emo_batch_tmp, tgt_batch, src_length_batch, keypoints_batch = [], [], [], [], []

        # 将批序列的name、imgs、tgt分别包装成列表
        for _, imgs_sample, tgt_sample, *other_data in batch:
            imgs_batch_tmp.append(imgs_sample)
            # tgt_sample 加入情感占位符
            tgt_sample = '<pad>' + tgt_sample
            # 一个batch情感收集
            # emo_batch_tmp.append('positive')
            tgt_batch.append(tgt_sample)
            if self.args['need_keypoints'] and other_data:
                keypoints_sample = torch.tensor(other_data[0])
                print('keypoints_sample.shape: ', keypoints_sample.shape)
                keypoints_batch.append(keypoints_sample)

        # 每个视频真实长度
        imgs_batch_len = [len(vid) for vid in imgs_batch_tmp]
        # print(imgs_batch_len)

        # 视频批序列最大长度
        imgs_batch_max_len = max(imgs_batch_len)

        # 将batch每个video的imgs序列填充成长度为imgs_batch_max_len
        imgs_batch = [torch.cat(
            (
                vid,
                torch.zeros(imgs_batch_max_len - len(vid), vid.size(1), vid.size(2), vid.size(3)).to(vid.device)
            ), dim=0)
            for vid in imgs_batch_tmp]

        imgs_batch = torch.stack(imgs_batch, dim=0)

        # 视频序列掩码
        img_padding_mask = torch.tensor(
            [[1] * length + [0] * (imgs_batch_max_len - length) for length in imgs_batch_len],
            dtype=torch.long
        )

        src_input = {
            'imgs_ids': imgs_batch,
            'attention_mask': img_padding_mask,

            'src_length_batch': imgs_batch_max_len}

        # 是否需要 need_keypoints
        if self.args['need_keypoints']:
            # 找到最大长度
            keypoints_batch_max_len = max(len(keypoints) for keypoints in keypoints_batch)
            # 将所有序列填充到最大长度
            keypoints_batch_padded = [torch.cat(
                (
                    keypoints,
                    torch.zeros(keypoints_batch_max_len - len(keypoints), keypoints.size(1)).to(keypoints.device)
                ),
                dim=0)
                for keypoints in keypoints_batch]

            # 将填充后的序列堆叠成张量
            keypoints_batch_tensor = torch.stack(keypoints_batch_padded, dim=0)
            src_input['keypoints_ids'] = keypoints_batch_tensor

        # 将一个batch的文本进行tokenizer
        # 对于批次中不同长度的文本进行填充
        # 截断过长的文本
        # with self.tokenizer.as_target_tokenizer():
        tgt_input = self.tokenizer(tgt_batch,
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True)

        print(f"正在加载数据集 {self.args['dataset']} ...")

        # # 情感pad初进行情感注入
        # for i, value in enumerate(utils.tokenizer(emo_batch_tmp)):
        #     tgt_input['input_ids'][i, 0] = value

        # 训练阶段需要mask掉一些，用来训练解码器
        if self.training_refurbish:
            masked_tgt = utils.noise_injecting(tgt_batch, self.args['noise_rate'],
                                               random_shuffle=self.args['random_shuffle'],
                                               is_train=(self.phase == 'train'))
            # with self.tokenizer.as_target_tokenizer():
            masked_tgt_input = self.tokenizer(masked_tgt,
                                              return_tensors="pt",
                                              padding=True,
                                              truncation=True)
            return src_input, tgt_input, masked_tgt_input

        # 返回一个batch视频集合 目标翻译的文本
        return src_input, tgt_input

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return f'# total {self.phase} set: {len(self.raw_data)}.'
