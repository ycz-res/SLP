import torch
import torch.nn as nn
import torch.nn.functional as F


# 文本编码器 cell
class EncoderLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLayer, self).__init__()
        self.hidden_size = hidden_size

        # 过滤门
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size)
        self.bf = nn.Parameter(torch.randn(hidden_size))

        # 注意力
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wv = nn.Linear(input_size + hidden_size, hidden_size)

        # 更新门
        self.Wu = nn.Linear(hidden_size, hidden_size)
        self.bu = nn.Parameter(torch.randn(hidden_size))

    def forward(self, xt, ht_1, ct_1, et_1):
        # concat
        concat = torch.cat((xt, ht_1), dim=1)

        # ct_1_hidden
        ft = torch.sigmoid(self.Wf(concat) + self.bf)
        ct_1_hidden = ft * ct_1

        Q = self.Wq(ct_1_hidden)
        K = self.Wk(concat)
        V = self.Wv(concat)

        # 计算注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 应用注意力权重到V
        attention_output = torch.matmul(attention_weights, V).squeeze(1)

        ut = torch.sigmoid(self.Wu(attention_output) + self.bu)
        et = ut * et_1

        ct = ct_1_hidden + attention_output
        ht = attention_output

        return ht, ct, et


# 文本编码器
class Encoder(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        # num_layers 个 EncoderLayer
        self.cells = nn.ModuleList(
            [EncoderLayer(hidden_size, hidden_size) for _ in range(num_layers)]
        )

    def forward(self, src):
        batch_size, seq_length, _ = src.size()
        ht_prev = torch.zeros(batch_size, self.hidden_size).to(src.device)
        ct_prev = torch.zeros(batch_size, self.hidden_size).to(src.device)
        et_prev = torch.zeros(batch_size, self.hidden_size).to(src.device)

        for t in range(seq_length):
            xt = src[:, t, :]

            for cell in self.cells:
                ht_prev, ct_prev, et_prev = cell(xt, ht_prev, ct_prev, et_prev)

        ht = ht_prev
        ct = ct_prev
        et = et_prev
        return ht, ct, et


# 姿态解码模型
class EmoGene(nn.Module):
    def __init__(self, src_dim=64, model_dim=64, tgt_dim=274, kp_nums=137,
                 scale=800, num_heads=2, num_layers=1, seq_length=156,
                 num_embeddings=30522):
        super(EmoGene, self).__init__()
        # self.scale = scale
        self.seq_length = seq_length
        self.src_embedding = nn.Embedding(num_embeddings, src_dim)
        self.position_embedding = torch.nn.Embedding(num_embeddings=seq_length, embedding_dim=model_dim)

        self.encoder = Encoder(hidden_size=model_dim, num_layers=num_layers)
        # 注意力
        self.Wq = nn.Linear(model_dim, model_dim)
        self.Wk = nn.Linear(model_dim + model_dim, model_dim)
        self.Wv = nn.Linear(model_dim + model_dim, model_dim)

        # 输入
        self.src_linear = nn.Linear(src_dim, model_dim)
        # Decoder的输入
        self.tgt_linear = nn.Linear(tgt_dim, model_dim)
        # Decoder的输出
        self.output_linear = nn.Linear(model_dim, kp_nums * 3)

        self.mha = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads)

    def forward(self, src, tgt):
        # 获取 ht、et
        src = self.src_embedding(src['input_ids'])
        src = self.src_linear(src)
        print('here')
        ht, _, et = self.encoder(src)

        # 位置编码
        position_indices = torch.arange(self.seq_length).unsqueeze(0).expand(tgt.size(0), -1)  # 形状为 [2, 156]
        pos = self.position_embedding(position_indices)  # 形状为 [2, 156, 64]

        # 注意力
        Q = self.Wq(ht)
        K = self.Wk(ht)

        tgt = self.tgt_linear(tgt['input_ids'])
        et_expanded = et.unsqueeze(1)
        V = tgt + et_expanded + pos
        V = self.Wv(V)

        output = self.mha(Q, K, V)

        print('output.shape:', output.shape)

        return output

# 结果评价模型
# 待写
