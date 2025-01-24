import torch
import torch.nn as nn


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, predicted, target):
        """
        计算生成姿态和真实标签姿态的一致性损失（MAE）。

        :param predicted: 生成的姿态，形状为 [batch_size, features, joints]
        :param target: 真实标签姿态，形状为 [batch_size, features, joints]
        :return: 平均绝对误差损失值
        """
        return torch.mean(torch.abs(predicted - target))
