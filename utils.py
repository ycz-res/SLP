import pandas as pd
import json
import yaml


# 打印迭代器的内容
def print_iterator(iterator):
    for item in iterator:
        print(item)


# yaml 文件加载
def load_config(file_path):
    with open(file_path, 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


# 访问 json 文件
def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()
        # 将列表转换为PyTorch张量
        return json.loads(data)


# How2Sign 数据集加载(文件类型 cvs)
def load_h2s_dataset(path):
    data_raws = pd.read_csv(path, sep='\t')
    res = [{
        'text': row['SENTENCE'],
        'keypoints_name': '' + row['SENTENCE_NAME'] + '.json'
    } for _, row in data_raws.iterrows()]

    return res


if __name__ == '__main__':
    pass
