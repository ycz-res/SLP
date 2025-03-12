import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration
import random
from transformers.models.mbart.modeling_mbart import shift_tokens_right


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

        # 初始化
        nn.init.xavier_uniform_(self.Wf.weight)
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)
        nn.init.xavier_uniform_(self.Wu.weight)

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

        # ut = torch.sigmoid(self.Wu(attention_output) + self.bu)
        ut = torch.sigmoid(self.Wu(attention_output))
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
        # et_prev = torch.zeros(batch_size, self.hidden_size).to(src.device)
        et_prev = torch.rand(batch_size, self.hidden_size).to(src.device) * 1.0 + 0.5

        for t in range(seq_length):
            xt = src[:, t, :]

            for cell in self.cells:
                ht_prev, ct_prev, et_prev = cell(xt, ht_prev, ct_prev, et_prev)

        ht = ht_prev
        ct = ct_prev
        et = et_prev / (seq_length * 100)

        # L2 归一化，使向量范数变成 1
        ht = F.normalize(ht, p=2, dim=-1)
        et = F.normalize(et, p=2, dim=-1)
        # print('ct', ct)
        # print('et', et)
        return ht, ct, et


# 姿态解码模型
class EmoGene(nn.Module):
    def __init__(self, src_dim=64, model_dim=64, tgt_dim=55, kp_nums=27,
                 num_heads=2, num_layers=1, seq_length=156,
                 num_embeddings=250027):
        super(EmoGene, self).__init__()
        # self.scale = scale
        self.seq_length = seq_length
        self.src_embedding = nn.Embedding(num_embeddings, src_dim)
        self.position_embedding = torch.nn.Embedding(num_embeddings=seq_length, embedding_dim=model_dim)

        self.encoder = Encoder(hidden_size=model_dim, num_layers=num_layers)
        # 注意力
        self.Wq = nn.Linear(model_dim, model_dim)
        self.Wk = nn.Linear(model_dim, model_dim)
        self.Wv = nn.Linear(model_dim, model_dim)

        # 初始化
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)

        # 输入
        self.src_linear = nn.Linear(src_dim, model_dim)
        # Decoder的输入
        self.tgt_linear = nn.Linear(tgt_dim, model_dim)
        # Decoder的输出
        self.output_linear = nn.Linear(model_dim, kp_nums * 2 + 1)

        self.mha = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads)

    def _gen_pos(self, tgt):
        batch_size, seq_len, _ = tgt.size()
        position_indices = torch.arange(seq_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
        pos = self.position_embedding(position_indices)
        return pos

    def forward(self, src, tgt):
        # 获取 ht、et
        src = self.src_embedding(src['input_ids'])
        src = self.src_linear(src)
        # 归一化 src
        src = F.normalize(src, p=2, dim=-1)  # L2 归一化
        print('src_embedding:', src.shape)

        ht, _, et = self.encoder(src)
        print('ht_shape:', ht.shape)

        print('et_shape:', et.shape)

        # 位置编码
        pos = self._gen_pos(tgt['input_ids'])
        print('pos_shape:', pos.shape)

        # 注意力
        _, seq_len, _ = tgt['input_ids'].size()
        Q = self.Wq(ht.unsqueeze(1).repeat(1, seq_len, 1))
        K = self.Wk(ht.unsqueeze(1).repeat(1, seq_len, 1))

        tgt = self.tgt_linear(tgt['input_ids'])
        print('tgt_embedding:', tgt.shape)
        et_expanded = et.unsqueeze(1)

        tgt = F.normalize(tgt, p=2, dim=-1)
        pos = F.normalize(pos, p=2, dim=-1)

        V = tgt + et_expanded + pos
        V = self.Wv(V)
        # 归一化 Q 和 K，避免注意力分数数值过大
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)
        V = F.normalize(V, p=2, dim=-1)
        print('Q_shape:', Q.shape)
        print('K_shape:', K.shape)
        print('V_shape:', V.shape)
        out, _ = self.mha(Q, K, V)
        # out = F.normalize(out, p=2, dim=-1)
        out = self.output_linear(out)
        # 调整为正数
        out = torch.abs(out)
        print('out.shape:', out.size())

        # 映射成姿态信息
        # 映射索引 1,2,4,5,7,8,10,11...
        idx_100_850 = torch.tensor([i for i in range(54) if i % 3 != 2],
                                   device=out.device)
        # 3,6,9,... 和最后一维
        idx_0_1 = torch.tensor([i for i in range(54) if i % 3 == 2] + [54],
                               device=out.device)

        # 映射到 100~850
        out_min_100_850 = out[:, :, idx_100_850].min()
        out_max_100_850 = out[:, :, idx_100_850].max()
        out[:, :, idx_100_850] = (out[:, :, idx_100_850] - out_min_100_850) / (out_max_100_850 - out_min_100_850)
        out[:, :, idx_100_850] = out[:, :, idx_100_850] * (850 - 100) + 100

        # 映射到 0~1
        out_min_0_1 = out[:, :, idx_0_1].min()
        out_max_0_1 = out[:, :, idx_0_1].max()
        out[:, :, idx_0_1] = (out[:, :, idx_0_1] - out_min_0_1) / (out_max_0_1 - out_min_0_1)
        # out[:, :, idx_0_1] = out[:, :, idx_0_1] * (1 - 0) + 0
        return out


# 投影层
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim=128, output_dim=1024):
        super(ProjectionLayer, self).__init__()
        # self.projection = nn.Linear(input_dim, output_dim)
        self.projection = nn.Sequential(
            # nn.ReLU(),  # 非线性激活
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            # nn.ReLU()  # 非线性激活
        )

        # 初始化
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

    def forward(self, x):
        projected_x = self.projection(x)
        return projected_x


# 结果评价模型
class ValEmoGene(nn.Module):
    def __init__(self, input_size=54, hidden_size=128, num_layers=1, batch_first=True):
        super(ValEmoGene, self).__init__()
        # 编码器
        # predicted_shape: torch.Size([1, 156, 275])
        # reference_shape: torch.Size([1, 156, 275])
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=batch_first)

        # 解码器
        self.MBart = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-cc25")
        self.txt_decoder = self.MBart.get_decoder()

        self.lm_head = self.MBart.get_output_embeddings()
        # 映射层
        self.projector_128_1024 = ProjectionLayer(input_dim=128, output_dim=1024)

    def forward(self, kp_ids, kp_mask, txt_input):
        # h0 = torch.randn(self.gru.num_layers, kp_ids.size(0), self.gru.hidden_size) * 0.01
        h0 = torch.randn(
            self.gru.num_layers, kp_ids.size(0), self.gru.hidden_size, device=kp_ids.device
        )
        hidden, _ = self.gru(kp_ids, h0)
        hidden = self.projector_128_1024(hidden)
        hidden = F.normalize(hidden, p=2, dim=-1)
        print('hidden.shape:', hidden.shape)
        # print('hidden:', hidden)

        # 增加随机性，防止模型过度自信
        if random.random() < 1:
            decoder_input_ids = shift_tokens_right(txt_input['input_ids'], self.txt_decoder.config.pad_token_id)
        else:
            decoder_input_ids = txt_input['input_ids']
        # print('decoder_input_ids:', decoder_input_ids)

        # **构造特殊的 attention_mask**
        batch_size, seq_len = txt_input['input_ids'].shape
        attention_mask = torch.ones((batch_size, seq_len), device=txt_input['input_ids'].device)

        decoder_out = self.txt_decoder(
            input_ids=decoder_input_ids,
            attention_mask=attention_mask,

            encoder_hidden_states=hidden,
            encoder_attention_mask=kp_mask,

            return_dict=True,
        )

        vocab_logits = self.lm_head(decoder_out.last_hidden_state)
        return vocab_logits
