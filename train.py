# 忽略警告---
import warnings

warnings.filterwarnings('ignore')
# ---忽略警告

import utils
import torch
import argparse
import random
import numpy as np
import os
from pathlib import Path
from transformers import AutoTokenizer, MBartTokenizer
from torch.utils.data import DataLoader
from model import TransformerPose
# 动态调用引入 How2SignDataset
from dataset import How2SignDataset
from timm.optim import create_optimizer
from torch.optim import lr_scheduler as scheduler
from torch.cuda.amp import GradScaler, autocast
from loss import MAELoss


def get_args_parser():
    a_parser = argparse.ArgumentParser('SLP scripts', add_help=False)
    a_parser.add_argument('--batch_size', default=1, type=int)
    a_parser.add_argument('--epochs', default=100, type=int)
    a_parser.add_argument('--num_workers', default=1, type=int)
    a_parser.add_argument('--config', type=str, default='./config.yaml')
    a_parser.add_argument('--device', default='cpu')
    a_parser.add_argument('--seed', default=0, type=int)
    a_parser.add_argument('--checkpoints_dir', default='./checkpoints/')
    # 锁定内存
    a_parser.add_argument('--pin_mem', action='store_true', default=True)

    # * Optimize参数
    a_parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    a_parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON')
    a_parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA')
    a_parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM')
    a_parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    a_parser.add_argument('--weight-decay', type=float, default=0.01)

    # * Learning rate 参数
    a_parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER')
    a_parser.add_argument('--lr', type=float, default=1.0e-4, metavar='LR')
    a_parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct')
    a_parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT')
    a_parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV')
    a_parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR')
    a_parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR')
    a_parser.add_argument('--decay-epochs', type=float, default=30, metavar='N')
    a_parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N')
    a_parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N')
    a_parser.add_argument('--patience-epochs', type=int, default=10, metavar='N')
    a_parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE')

    a_parser.add_argument('--dataset', default='How2SignDataset', type=str,
                          choices=['How2SignDataset', 'P14TDataset', 'CSLDailyDataset'])

    return a_parser


def main(args_, config):
    args = vars(args_)
    # 获取设备
    device = torch.device(args['device'])
    print("starting on...", device, sep=' ')

    # 设置随机种子
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 加载预训练模型的分词器
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # args['vocab_size'] = tokenizer.vocab_size
    # print('tokenizer.vocab_size', tokenizer.vocab_size)
    # tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")

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

    # 验证数据集
    val_data = eval(args['dataset'])(path=config[args['dataset']]['dev_label_path'],
                                     tokenizer=tokenizer,
                                     config=config,
                                     args=args,
                                     phase='val')
    val_dataloader = DataLoader(val_data,
                                batch_size=args['batch_size'],
                                num_workers=args['num_workers'],
                                collate_fn=val_data.collate_fn,
                                pin_memory=args['pin_mem'],
                                drop_last=True)

    # SLP Model
    model = TransformerPose()

    # 移动到设备上
    # model.to(device)

    optimizer = create_optimizer(args_, model)
    criterion = MAELoss()
    scaler = GradScaler()

    lr_scheduler = scheduler.CosineAnnealingLR(optimizer, eta_min=args['min_lr'], T_max=args['epochs'])

    best_loss = float('inf')
    for epoch in range(args['epochs']):
        torch.cuda.empty_cache()
        train_loss = train_one_epoch(model,
                                     train_dataloader,
                                     optimizer,
                                     criterion,
                                     device,
                                     scaler)
        print('train_loss: ', train_loss)

        # TUDO

        val_loss = evaluate(model, val_dataloader,
                            criterion, device)


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(dataloader):
        print('---step---: ', step)
        try:
            optimizer.zero_grad()
            src_input, tgt_input = batch
            with autocast():
                out = model(src_input, tgt_input)
                # # 输出调整
                x_values = out[:, 0::3, :]
                y_values = out[:, 1::3, :]
                p_values = out[:, 2::3, :]

                scale = 800
                # # 缩放x_values和y_values到 0到scale 的范围
                x_values = torch.abs(x_values) * (scale / torch.max(torch.abs(x_values)))
                y_values = torch.abs(y_values) * (scale / torch.max(torch.abs(y_values)))

                # # 保留三位小数
                x_values = torch.round(x_values * 1000) / 1000
                y_values = torch.round(y_values * 1000) / 1000

                # # 将p_values限制在0到1之间
                p_values = torch.sigmoid(p_values)
                p_values = torch.clamp(p_values, 0, 1)

                # # 可以选择将x_values, y_values, p_values重新组合到out中
                out = torch.cat((x_values, y_values, p_values), dim=1)
                predicted_pose = out[:, :, torch.arange(out.size(-1)) % 3 != 2]
                target_pose = tgt_input['input_ids']

                print('predicted_pose', predicted_pose.shape)
                print('target_pose', target_pose.shape)
                loss = criterion(predicted_pose, target_pose)
                print('loss', loss)
                # 反向传播和梯度更新
                scaler.scale(loss).backward()

                # 梯度裁剪
                print('here1')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                print('here')
                scaler.step(optimizer)
                scaler.update()

                # 计算损失
                running_loss += loss.item() * src_input['input_ids'].size(0)

        except Exception as e:
            print("数据错误，摒弃本数据。", e)
            continue

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            print('---step---: ', step)
            try:
                src_input, tgt_input = batch
                with autocast():
                    out = model(src_input, tgt_input)
                    print('val_out.shape', out.shape)
                    # 计算评价指标
            except Exception as e:
                print("数据错误，摒弃本数据。", e)
                continue

    return running_loss


if __name__ == '__main__':
    # 显式启用并行化
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # 加载参数
    parser = argparse.ArgumentParser('SLP scripts', parents=[get_args_parser()])
    args_ = parser.parse_args()
    args = vars(args_)
    config = utils.load_config(args['config'])

    # 创建输出文件夹
    if args['checkpoints_dir']:
        Path(args['checkpoints_dir']).mkdir(parents=True, exist_ok=True)

    # 开始训练
    main(args_, config)
