import numpy as np
import matplotlib.pyplot as plt
import re

# 打开并读取文件
file_path = 'slt.txt'  # 替换为你的文件路径
with open(file_path, 'r', encoding='utf-8') as file:
    file_content = file.read()

# 使用正则表达式提取所有的 train_loss 值
train_loss_pattern = re.compile(r'train_loss=(\d+\.\d+)')
train_loss_values = train_loss_pattern.findall(file_content)

# 将提取的值转换为浮点数
train_loss_values = [float(value) for value in train_loss_values]
train_loss_values = train_loss_values[1:400]
print(train_loss_values[-1])

# 生成 x 轴数据（epoch）
epochs = np.arange(1, len(train_loss_values) + 1)

# 绘制曲线
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_loss_values, marker='.', linestyle='-', color='b', label='Emotion MAE Value')

plt.xlabel('Epoch')
plt.ylabel('Emotion MAE Value')
plt.legend()

plt.show()
