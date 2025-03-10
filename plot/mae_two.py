import numpy as np
import matplotlib.pyplot as plt
import re

# 打开并读取文件
file_path = 'slt_1.txt'  # 替换为你的文件路径
with open(file_path, 'r', encoding='utf-8') as file:
    file_content = file.read()

# 使用正则表达式提取所有的 train_loss 值
train_loss_pattern = re.compile(r'train_loss=(\d+\.\d+)')
train_loss_values = train_loss_pattern.findall(file_content)

# 使用正则表达式提取所有的 val_loss 值
val_loss_pattern = re.compile(r'val_loss=(\d+\.\d+)')
val_loss_values = val_loss_pattern.findall(file_content)

# 将提取的值转换为浮点数
train_loss_values = [float(value) for value in train_loss_values]
val_loss_values = [float(value) for value in val_loss_values]

# 截取前400个数据点
train_loss_values = train_loss_values[1:400]
val_loss_values = val_loss_values[1:400]

# 生成 x 轴数据（epoch）
epochs = np.arange(1, len(train_loss_values) + 1)

# 绘制曲线
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_loss_values, marker='.', linestyle='-', color='orange', label='Train Loss')
plt.plot(epochs, val_loss_values, marker='.', linestyle='-', color='red', label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()

plt.show()