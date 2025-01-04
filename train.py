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


def get_args_parser():
    a_parser = argparse.ArgumentParser('SLP scripts', add_help=False)
    a_parser.add_argument('--batch_size', default=1, type=int)
    a_parser.add_argument('--epochs', default=2, type=int)
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
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")

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
    model = TransformerPose()

    # 移动到设备上
    # model.to(device)

    optimizer = create_optimizer(args_, model)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()

    lr_scheduler = scheduler.CosineAnnealingLR(optimizer, eta_min=args['min_lr'], T_max=args['epochs'])

    best_loss = float('inf')
    for epoch in range(args['epochs']):
        torch.cuda.empty_cache()
        train_one_epoch(model,
                        train_dataloader,
                        optimizer,
                        criterion,
                        device,
                        scaler,
                        tokenizer)
    #     train_one_epoch(model, train_dataloader, optimizer, criterion, device, scaler,
    #                     tokenizer)

    # utils.print_iterator(train_dataloader)
    # for item in train_dataloader:
    #     pass
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


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, tokenizer, clip_grad_norm=1.0, alpha=0.5):
    model.train()

    for step, batch in enumerate(dataloader):
        print('---step---: ', step)
        try:
            optimizer.zero_grad()
            src_input, tgt_input = batch
            with autocast():
                out = model(src_input, src_input)
                print(out.shape)

        except Exception as e:
            print("数据错误，摒弃本数据。", e)
            continue


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
