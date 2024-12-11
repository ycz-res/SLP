import torch
import torch.nn as nn


class TransformerPose(nn.Module):
    def __init__(self,
                 k_p_nums=10,
                 k_p_dim=3,
                 txt_input_dim=64,
                 num_heads=2,
                 num_layers=1,
                 dim_feedforward=128,
                 dropout=0.1):
        super(TransformerPose, self).__init__()
        self.input_dim = txt_input_dim
        self.output_dim = k_p_nums * k_p_dim
        self.model_dim = k_p_nums * k_p_dim
        self.scale = 800

        # 编码器和解码器的层数
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输入
        self.input_linear = nn.Linear(self.input_dim, self.model_dim)
        # 输出
        self.output_linear = nn.Linear(self.model_dim, self.output_dim)

    def forward(self, src, tgt):
        src = self.input_linear(src)

        hidden = self.encoder(src)
        out = self.decoder(tgt, hidden)

        out = self.output_linear(out)

        # 输出调整
        x_values = out[:, 0::3, :]
        y_values = out[:, 1::3, :]
        p_values = out[:, 2::3, :]

        # 缩放x_values和y_values到 0到self.scale 的范围
        x_values = torch.abs(x_values) * (self.scale / torch.max(torch.abs(x_values)))
        y_values = torch.abs(y_values) * (self.scale / torch.max(torch.abs(y_values)))

        # 保留三位小数
        x_values = torch.round(x_values * 1000) / 1000
        y_values = torch.round(y_values * 1000) / 1000

        # 将p_values限制在0到1之间
        p_values = torch.sigmoid(p_values)
        p_values = torch.clamp(p_values, 0, 1)

        # 可以选择将x_values, y_values, p_values重新组合到out中
        out = torch.cat((x_values, y_values, p_values), dim=1)

        return out
