import torch
import torch.nn as nn


# 特征提取类
class KPEmbedding(nn.Module):
    def __init__(self, input_dim=274, output_dim=128):
        super(KPEmbedding, self).__init__()
        # 初始化线性层，将输入维度映射到输出维度
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 应用线性变换
        return self.linear(x)


class TransformerPose(nn.Module):
    def __init__(self,
                 src_dim=64,
                 model_dim=64,
                 tgt_dim=274,
                 kp_nums=137,
                 scale=800,
                 num_heads=2,
                 num_layers=1,
                 dim_feedforward=128,
                 dropout=0.1):
        super(TransformerPose, self).__init__()
        self.scale = scale

        # 编码器和解码器的层数
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输入
        self.input_linear = nn.Linear(src_dim, model_dim)
        # Decoder的输入
        self.kp_embedding = nn.Linear(tgt_dim,  model_dim)
        # Decoder的输出
        self.output_linear = nn.Linear(model_dim, kp_nums*3)

    def forward(self, src, tgt):
        src = self.input_linear(src['input_ids'])
        hidden = self.encoder(src)

        kp_embedding = self.kp_embedding(tgt['input_ids'])
        out = self.decoder(kp_embedding, hidden)
        out = self.output_linear(out)

        return out

        # # 输出调整
        # x_values = out[:, 0::3, :]
        # y_values = out[:, 1::3, :]
        # p_values = out[:, 2::3, :]
        #
        # # 缩放x_values和y_values到 0到self.scale 的范围
        # x_values = torch.abs(x_values) * (self.scale / torch.max(torch.abs(x_values)))
        # y_values = torch.abs(y_values) * (self.scale / torch.max(torch.abs(y_values)))
        #
        # # 保留三位小数
        # x_values = torch.round(x_values * 1000) / 1000
        # y_values = torch.round(y_values * 1000) / 1000
        #
        # # 将p_values限制在0到1之间
        # p_values = torch.sigmoid(p_values)
        # p_values = torch.clamp(p_values, 0, 1)
        #
        # # 可以选择将x_values, y_values, p_values重新组合到out中
        # out = torch.cat((x_values, y_values, p_values), dim=1)

        # return out
