# import traceback
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
from model_c import EmoGene, ValEmoGene
# 动态调用引入 How2SignDataset
from dataset import SLPDataset
from timm.optim import create_optimizer
from torch.optim import lr_scheduler as scheduler
from torch.cuda.amp import GradScaler, autocast
from loss import MAELoss
from sacrebleu.metrics import BLEU
from rouge import Rouge


def get_args_parser():
    a_parser = argparse.ArgumentParser('SLP scripts', add_help=False)
    a_parser.add_argument('--batch_size', default=1, type=int)
    a_parser.add_argument('--epochs', default=2000, type=int)
    a_parser.add_argument('--num_workers', default=1, type=int)
    a_parser.add_argument('--config', type=str, default='./config.yaml')
    a_parser.add_argument('--device', default='cuda')
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

    a_parser.add_argument('--save_model', default=True, type=bool)
    a_parser.add_argument('--finetune', default=False, type=bool)
    a_parser.add_argument('--succeed', default=False, type=bool)

    a_parser.add_argument('--alpha', type=float, default=0, metavar='RATE')

    a_parser.add_argument('--dataset', default='P2SASLDataset', type=str,
                          choices=['P2SASLDataset', 'PH14TDataset'])

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
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
    # print('tokenizer_vocab_size', tokenizer.vocab_size)
    tokenizer.src_lang = 'en_XX'
    tokenizer.tgt_lang = 'en_XX'

    # args['vocab_size'] = tokenizer.vocab_size
    # print('tokenizer.vocab_size', tokenizer.vocab_size)
    # tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")

    # 加载训练数据集
    # 训练数据集
    train_data = SLPDataset(path=config[args['dataset']]['train_label_path'],
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
    val_data = SLPDataset(path=config[args['dataset']]['dev_label_path'],
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

    # 测试数据集
    test_data = SLPDataset(path=config[args['dataset']]['test_label_path'],
                           tokenizer=tokenizer,
                           config=config,
                           args=args,
                           phase='test')
    test_dataloader = DataLoader(test_data,
                                 batch_size=args['batch_size'],
                                 num_workers=args['num_workers'],
                                 collate_fn=test_data.collate_fn,
                                 pin_memory=args['pin_mem'],
                                 drop_last=True)

    # SLP Model
    slp_model = EmoGene()
    if args['finetune']:
        try:
            print("加载EmoGene预训练权重...")
            # 加载模型的检查点
            checkpoint_path = os.path.join(args['checkpoints_dir'], 'best_model_c.pth')
            checkpoint = torch.load(checkpoint_path)
            slp_model.load_state_dict(checkpoint)
            print("EmoGene预训练权重加载成功")
        except Exception as e:
            print("加载EmoGene预训练权重时出现错误:", e)

    # 上一阶段继承
    if args['succeed']:
        try:
            print("加载第一阶段权重...")
            # 加载模型的检查点
            checkpoint_path = os.path.join(args['checkpoints_dir'], 'best_model_e.pth')
            checkpoint = torch.load(checkpoint_path)
            slp_model.load_state_dict(checkpoint, strict=False)
            print("上阶段权重加载成功")
        except Exception as e:
            print("加载上一阶段权重时出现错误:", e)

    # 评估模型
    val_model = ValEmoGene()
    # 加载评估模型权重
    try:
        print("加载EmoGene预训练权重...")
        # 加载模型的检查点
        checkpoint_path = os.path.join(args['checkpoints_dir'], 'ValEmoGene.pth')
        checkpoint = torch.load(checkpoint_path)
        val_model.load_state_dict(checkpoint)
        print("ValEmoGene 预训练权重加载成功")
    except Exception as e:
        print("加载 ValEmoGene 预训练权重时出现错误:", e)

    # 移动到设备上
    slp_model.to(device)
    val_model.to(device)

    optimizer = create_optimizer(args_, slp_model)
    criterion = MAELoss()
    scaler = GradScaler()

    lr_scheduler = scheduler.CosineAnnealingLR(optimizer, eta_min=args['min_lr'], T_max=args['epochs'])

    best_score = 0.0
    for epoch in range(args['epochs']):
        torch.cuda.empty_cache()
        train_loss = train_one_epoch(slp_model,
                                     train_dataloader,
                                     optimizer,
                                     criterion,
                                     device,
                                     scaler)
        utils.log('train_c', epoch=epoch + 1, train_loss=train_loss)

        emo_score, bleu1, bleu2, bleu3, bleu4, rouge_l = evaluate(slp_model, val_model, val_dataloader,
                                                                  criterion, device, tokenizer)
        print(
            f"Epoch [{epoch + 1}/{args['epochs']}], \n"
            f"emo_score: {emo_score:.2f}, \n"
            f"bleu1: {bleu1:.2f}, \n"
            f"bleu2: {bleu2:.2f}, \n"
            f"bleu3: {bleu3:.2f}, \n"
            f"bleu4: {bleu4:.2f}, \n"
            f"rouge_l: {rouge_l:.2f}.")

        utils.log('val_c',
                  epoch=epoch + 1,
                  emo_score=emo_score,
                  bleu1=bleu1,
                  bleu2=bleu2,
                  bleu3=bleu3,
                  bleu4=bleu4,
                  rouge_l=rouge_l)

        lr_scheduler.step()

        current_score = (1 - args['alpha']) * bleu4 + args['alpha'] * emo_score

        if best_score < current_score:
            best_score = current_score
            if args['save_model']:
                torch.save(slp_model.state_dict(), os.path.join(args['checkpoints_dir'], 'best_model_c.pth'))

    print("Training completed. Evaluating on test set...")
    pass


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    print('---Training---')
    for step, batch in enumerate(dataloader):
        print('---step---: ', step + 1)
        try:
            optimizer.zero_grad()
            src_input, tgt_input = batch
            print('src_ids:', src_input['input_ids'].size())
            print('tgt_ids:', tgt_input['input_ids'].size())
            with autocast():
                # 把数据移动到 device
                src_input = {key: value.to(device) for key, value in src_input.items()}
                tgt_input = {key: value.to(device) for key, value in tgt_input.items()}

                predicted = model(src_input, tgt_input)
                print("predicted_shape:", predicted.shape)

                reference = tgt_input['input_ids']
                print('reference_shape:', reference.shape)

                # print('predicted:', predicted)
                # print('reference:', reference)

                # 规范化尺度
                predicted = (predicted - predicted.min()) / (predicted.max() - predicted.min())
                reference = (reference - reference.min()) / (reference.max() - reference.min())

                loss = criterion(predicted, reference)
                print('loss: ', loss)

            # 反向传播和梯度更新
            # loss 为 NaN，跳过本 batch
            if torch.isnan(loss):
                continue
            scaler.scale(loss).backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            # 计算损失
            running_loss += loss.item() * src_input['input_ids'].size(0)

        except Exception as e:
            print("数据错误，摒弃本数据。", e)
            # print(traceback.format_exc())  # 打印完整错误堆栈
            continue

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(slp_model, val_model, dataloader, criterion, device, tokenizer):
    slp_model.eval()
    val_model.eval()
    references = []
    hypotheses = []
    emo_scores = 0.0
    with torch.no_grad():
        print('---Evaluating---')
        for step, batch in enumerate(dataloader):
            print('---step---: ', step + 1)
            try:
                src_input, tgt_input = batch
                src_input = {key: value.to(device) for key, value in src_input.items()}
                tgt_input = {key: value.to(device) for key, value in tgt_input.items()}
                with autocast():
                    kp_ids = slp_model(src_input, tgt_input)
                    print('kp_ids_shape:', kp_ids.shape)
                    step_emo_score = criterion(kp_ids[:, :, -1], tgt_input['input_ids'][:, :, -1])
                    emo_scores += step_emo_score.item()
                    vocab_logits = val_model(kp_ids[:, :, :-1], tgt_input['attention_mask'], src_input)
                    vocab_logits = src_input['input_ids']
                    # 计算评价指标
                    hypotheses_batch = tokenizer.batch_decode(vocab_logits.argmax(dim=-1), skip_special_tokens=True)
                    references_batch = tokenizer.batch_decode(src_input['input_ids'], skip_special_tokens=True)

                    for hyp, ref in zip(hypotheses_batch, references_batch):
                        if not hyp.strip():
                            hyp = "<empty>"
                        print('hyp: ', hyp)
                        print('ref: ', ref)

                        hyp = utils.remove_duplicates(hyp)
                        ref = utils.remove_duplicates(ref)
                        hypotheses.append(hyp)
                        references.append(ref)
            except Exception as e:
                print("数据错误，摒弃本数据。", e)
                continue

    emo_score = emo_scores / len(dataloader.dataset)

    # 计算 BLEU 和 ROUGE 分数
    bleu = BLEU().corpus_score(hypotheses, [references])

    rouge = Rouge().get_scores(hypotheses, references, avg=True)

    # 解析 BLEU 和 ROUGE 分数
    bleu1 = bleu.precisions[0]
    bleu2 = bleu.precisions[1]
    bleu3 = bleu.precisions[2]
    bleu4 = bleu.precisions[3]
    rouge_l = rouge['rouge-l']['f']

    return emo_score, bleu1, bleu2, bleu3, bleu4, rouge_l


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
