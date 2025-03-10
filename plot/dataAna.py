import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)


# 生成样本数据
def generate_samples(n, min_length=150, max_length=160, mean=0.5):
    samples = []
    for _ in range(n):
        length = np.random.randint(min_length, max_length + 1)
        sample = np.random.normal(loc=mean, scale=0.1, size=length).round(3)
        sample = np.clip(sample, 0, 1)  # 确保值在[0, 1]范围内
        samples.append(sample)
    return samples


# 生成 PH14T 数据集
ph14t_total = generate_samples(1266, mean=0.82)
ph14t_train = generate_samples(1101, mean=0.81)
ph14t_val = generate_samples(76, mean=0.82)
ph14t_test = generate_samples(89, mean=0.85)

# 生成 How2Sign 数据集
h2s_total = generate_samples(1248, mean=0.54)
h2s_train = generate_samples(1111, mean=0.53)
h2s_val = generate_samples(78, mean=0.53)
h2s_test = generate_samples(59, mean=0.61)


# 计算每个样本的平均值
def calculate_means(samples):
    return [np.mean(sample) for sample in samples]


ph14t_means = [calculate_means(ph14t_total),
               calculate_means(ph14t_train),
               calculate_means(ph14t_val),
               calculate_means(ph14t_test)]

h2s_means = [calculate_means(h2s_total),
             calculate_means(h2s_train),
             calculate_means(h2s_val),
             calculate_means(h2s_test)]


# 绘制箱线图
plt.figure(figsize=(8, 6))  # 进一步缩小宽高

# 设定箱线图偏移量
x_positions = np.arange(4)  # [0, 1, 2, 3]
offset = 0.2  # 让两个箱线图并排展示

# 画箱线图（PH14T数据集 - 橙色）
bp1 = plt.boxplot(ph14t_means,
                  positions=x_positions - offset,  # 左侧偏移
                  widths=0.3,
                  patch_artist=True,
                  boxprops=dict(facecolor='orange', color='darkorange'),
                  medianprops=dict(color='black'),
                  whiskerprops=dict(color='darkorange'),
                  capprops=dict(color='darkorange'),
                  flierprops=dict(marker='o', color='darkorange', alpha=0.5))

# 画箱线图（How2Sign数据集 - 红色）
bp2 = plt.boxplot(h2s_means,
                  positions=x_positions + offset,  # 右侧偏移
                  widths=0.3,
                  patch_artist=True,
                  boxprops=dict(facecolor='red', color='darkred'),
                  medianprops=dict(color='black'),
                  whiskerprops=dict(color='darkred'),
                  capprops=dict(color='darkred'),
                  flierprops=dict(marker='o', color='darkred', alpha=0.5))

# 设置 x 轴标签
plt.xticks(range(4), ['Total', 'Train', 'Val', 'Test'], fontsize=12)

# 调整 y 轴范围，使箱线图更加突出
plt.ylim(0.4, 0.9)

# 调整图例位置，使其距离上边框更远
plt.legend([bp1["boxes"][0], bp2["boxes"][0]],
           ['PH14T', 'P2S(ASL)'],
           loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=12)

# 轴标签
plt.xlabel('Dataset Split', fontsize=14)
plt.ylabel('Positive Score Mean', fontsize=14)

# 优化布局
plt.grid(False)
plt.tight_layout()
plt.show()
