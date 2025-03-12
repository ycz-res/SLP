import os
import torch
import utils
from torch.utils.data import Dataset
from augmentation import *


# 数据集
class SLPDataset(Dataset):
    def __init__(self, path, tokenizer, config, args, phase):
        self.dataset = utils.load_h2s_dataset(path)

        self.tokenizer = tokenizer
        self.config = config
        self.args = args
        self.phase = phase

        self.kps_dir = config[self.args['dataset']]['kps_dir']
        self.max_length = config[self.args['dataset']]['max_length']

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        src_sample = sample['text']
        kp_file = sample['kp_file']

        try:
            tgt_sample = self._load_kps(os.path.join(self.kps_dir, kp_file))
        except Exception:
            print("json不存在，设置为默认值.")
            tgt_sample = torch.zeros(156, 55, dtype=torch.float)
        # print(tgt_sample.dtype)
        return src_sample, tgt_sample

    def _load_kps(self, path):
        data = utils.load_json(path)
        # 如果关键点向量数量超过最大长度，随机抽取最大长度的关键点向量，并保持顺序
        if len(data) > self.max_length:
            data = [data[i] for i in
                    sorted(random.sample(range(len(data)), self.max_length))]
        # data_tensor = torch.tensor(data, dtype=torch.float16)
        data_tensor = torch.tensor(data, dtype=torch.float)
        return data_tensor

    def collate_fn(self, batch):
        print(f"loading {self.args['dataset']} ...")
        src_batch, tgt_batch = [], []

        # 将批序列的name、tgt分别包装成列表
        for src_sample, tgt_sample in batch:
            # src_sample = '<pad>' + src_sample
            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)

        # --文本--
        src_input = self.tokenizer(src_batch,
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True)
        src_input['input_ids'] = src_input['input_ids'].long()

        # --关键点--
        tgt_batch_len = [len(vid) for vid in tgt_batch]
        tgt_batch_max_len = max(tgt_batch_len)
        # 将 batch 每个项长度填充到 tgt_batch_max_len
        tgt_batch = torch.stack([
            torch.cat(
                (
                    vid,
                    torch.zeros(tgt_batch_max_len - len(vid), *vid.shape[1:])
                ), dim=0)
            for vid in tgt_batch
        ]).float()

        # 关键点序列掩码
        tgt_batch_mask = torch.tensor(
            [[1] * length + [0] * (tgt_batch_max_len - length) for length in tgt_batch_len],
            dtype=torch.long
        )

        tgt_input = {
            'input_ids': tgt_batch,
            'attention_mask': tgt_batch_mask}

        # 返回一个batch视频集合 目标翻译的文本
        return src_input, tgt_input

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return f'# total {self.phase} set: {len(self.dataset)}.'
