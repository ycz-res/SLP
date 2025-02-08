import torch
import torch.nn as nn

# 姿态生成模型
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
                 dropout=0.1,
                 num_embeddings=30522):
        super(TransformerPose, self).__init__()
        self.scale = scale
        self.src_embedding = nn.Embedding(num_embeddings, src_dim)

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
        self.src_linear = nn.Linear(src_dim, model_dim)
        # Decoder的输入
        self.tgt_linear = nn.Linear(tgt_dim, model_dim)
        # Decoder的输出
        self.output_linear = nn.Linear(model_dim, kp_nums * 3)

    def forward(self, src, tgt):
        src = self.src_embedding(src['input_ids'])
        src = self.src_linear(src)
        hidden = self.encoder(src)

        tgt = self.tgt_linear(tgt['input_ids'])
        print('model_src-dim-dtype', src.shape, src.dtype)
        print('model_hidden-dim-dtype', hidden.shape, hidden.dtype)
        print('model_tgt-dim-dtype', tgt.shape, tgt.dtype)

        out = self.decoder(tgt, hidden)
        out = self.output_linear(out)

        return out

        # print('tgt[input_ids]', tgt['input_ids'].shape)
        # tgt = tgt['input_ids'].mean(dim=2)


# 结果评价模型
