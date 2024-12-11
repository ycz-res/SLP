from model import TransformerPose
import torch

# 实例化模型
model = TransformerPose(k_p_nums=10,
                        k_p_dim=3)
src = torch.randn(1, 20, 64)
tgt = torch.randn(1, 30, 30)
out = model(src, tgt)
print(out)
print(out.shape)
