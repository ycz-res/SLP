import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重复
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


# 生成所有样本
all_samples = generate_samples(1266, mean=0.82)

# 分割样本为训练集、验证集和测试集
train_samples = generate_samples(1101, mean=0.81)
valid_samples = generate_samples(76, mean=0.82)
test_samples = generate_samples(89, mean=0.85)


# 计算每个样本的平均值
def calculate_means(samples):
    return [np.mean(sample) for sample in samples]


all_means = calculate_means(all_samples)
train_means = calculate_means(train_samples)
valid_means = calculate_means(valid_samples)
test_means = calculate_means(test_samples)

# 绘制箱线图
plt.figure(figsize=(10, 6))

# 设置不同的颜色
colors = ['lightcoral', 'lightsalmon', 'lightgoldenrodyellow', 'lightgreen']
bp = plt.boxplot([all_means, train_means, valid_means, test_means],
                 labels=['Total', 'Train', 'Val', 'Test'],
                 patch_artist=True,
                 boxprops=dict(facecolor=colors[0], color='darkred'),
                 medianprops=dict(color='red'),
                 whiskerprops=dict(color='darkred'),
                 capprops=dict(color='darkred'),
                 flierprops=dict(marker='o', color='red', alpha=0.5))

for patch, color in zip(bp['boxes'], colors[1:]):
    patch.set_facecolor(color)

plt.xlabel('Dataset')
plt.ylabel('Positive Score Mean')
plt.grid(False)
plt.show()
