import re

# 打开并读取文件
file_path = 'res/slt.txt'  # 替换为你的文件路径
with open(file_path, 'r', encoding='utf-8') as file:
    file_content = file.read()

# 使用正则表达式提取所有的 train_loss 值
train_loss_pattern = re.compile(r'train_loss=(\d+\.\d+)')
train_loss_values = train_loss_pattern.findall(file_content)

# 将提取的值转换为浮点数
train_loss_values = [float(value) for value in train_loss_values]

# 打印结果
print(len(train_loss_values))
print(train_loss_values)