import os
import pandas as pd

# 创建目标文件夹
if not os.path.exists('./res'):
    os.makedirs('./res')

# 遍历当前目录下的所有csv文件
for file_name in os.listdir('./'):
    if file_name.endswith('.csv'):
        print(f"正在处理文件: {file_name}")  # 添加提示信息
        # 读取csv文件
        df = pd.read_csv(file_name, sep='\t')

        # 保留SENTENCE_NAME列并改名为FILE_NAME，保留SENTENCE列并去掉第一个单词
        df = df[['SENTENCE_NAME', 'SENTENCE']]
        df.rename(columns={'SENTENCE_NAME': 'FILE_NAME'}, inplace=True)
        df['SENTENCE'] = df['SENTENCE'].apply(lambda x: ' '.join(x.split()[1:]))

        # 根据文件名保存到目标文件夹
        output_file_name = os.path.join('./res', file_name)
        df.to_csv(output_file_name, sep='\t', index=False)

print("所有文件处理完成")