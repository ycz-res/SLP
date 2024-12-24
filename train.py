import utils
import torch
import argparse
import random
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import TransformerPose
# 动态调用引入 How2SignDataset
from dataset import How2SignDataset


def get_args_parser():
    a_parser = argparse.ArgumentParser('SLP scripts', add_help=False)
    a_parser.add_argument('--batch_size', default=2, type=int)
    a_parser.add_argument('--epochs', default=10, type=int)
    a_parser.add_argument('--num_workers', default=1, type=int)
    a_parser.add_argument('--config', type=str, default='./config.yaml')
    a_parser.add_argument('--device', default='cpu')
    a_parser.add_argument('--seed', default=0, type=int)
    a_parser.add_argument('--checkpoints_dir', default='./checkpoints/')
    # 锁定内存
    a_parser.add_argument('--pin_mem', action='store_true', default=True)

    a_parser.add_argument('--dataset', default='How2SignDataset', type=str,
                          choices=['How2SignDataset', 'P14TDataset', 'CSLDailyDataset'])

    return a_parser


def main(args, config):
    # 获取设备
    device = torch.device(args['device'])
    print("starting on...", device, sep=' ')

    # 设置随机种子
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 加载预训练模型的分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # 加载训练数据集
    # 训练数据集
    train_data = eval(args['dataset'])(path=config[args['dataset']]['train_label_path'],
                                       tokenizer=tokenizer,
                                       config=config,
                                       args=args,
                                       phase='train')
    train_dataloader = DataLoader(train_data,
                                  batch_size=args['batch_size'],
                                  num_workers=args['num_workers'],
                                  collate_fn=train_data.collate_fn,
                                  pin_memory=args['pin_mem'],
                                  drop_last=True)

    # SLP Model
    model = TransformerPose(k_p_nums=10,
                            k_p_dim=3)

    # 移动到设备上
    model.to(device)

    # utils.print_iterator(train_dataloader)

    for item in train_dataloader:
        pass
    #     # for src_input, tgt_input in item:
    #     print(type(item[0]))
    # print(tgt_input['input_ids'].size)

    # print(train_dataloader)
    # 实例化模型
    # model = TransformerPose(k_p_nums=10,
    #                         k_p_dim=3)
    #
    # src = torch.randn(1, 20, 64)
    # tgt = torch.randn(1, 30, 30)
    # out = model(src, tgt)
    # print(out)
    # print(out.shape)


if __name__ == '__main__':
    # 加载参数
    parser = argparse.ArgumentParser('SLP scripts', parents=[get_args_parser()])
    args = vars(parser.parse_args())
    config = utils.load_config(args['config'])

    # 创建输出文件夹
    if args['checkpoints_dir']:
        Path(args['checkpoints_dir']).mkdir(parents=True, exist_ok=True)

    # 开始训练
    main(args, config)
