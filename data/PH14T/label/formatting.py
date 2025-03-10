import os
import pickle
import csv
import gzip

# 创建res文件夹，如果不存在的话
if not os.path.exists('./res'):
    os.makedirs('./res')


def load_pkl_file(file_path):
    print('正在处理：', file_path)
    with open(file_path, 'rb') as file:
        # 尝试加载文件内容
        try:
            data = pickle.load(file)
        except pickle.UnpicklingError:
            # 如果出现 UnpicklingError，尝试以 gzip 格式读取
            with gzip.open(file_path, 'rb') as gzip_file:
                data = pickle.load(gzip_file)
    return data


# 遍历当前目录下的所有pkl文件
for file in os.listdir('./'):
    if file.endswith('.pkl'):
        file_path = os.path.join('./', file)  # 修正路径拼接
        data = load_pkl_file(file_path)

        # 准备csv文件的路径
        csv_file_path = os.path.join('./res', file.replace('.pkl', '.csv'))

        # 写入csv文件
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            # 使用制表符作为分隔符
            writer = csv.writer(csvfile, delimiter='\t')
            # 写入表头
            writer.writerow(['FILE_NAME', 'SENTENCE'])
            for item in data:
                for key, value in item.items():
                    # 处理 value['name']，只保留 / 后面的内容
                    file_name = value['name'].split('/')[-1]
                    # 处理 value['text']，去掉第一个单词
                    sentence = ' '.join(value['text'].split()[1:])
                    writer.writerow([file_name, sentence])

print('处理完成.')
