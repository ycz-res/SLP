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


def save_json(path, data):
    try:
        # 保存数据到 JSON 文件
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"文件已成功保存: {path}")
    except Exception as e:
        print(f"保存文件失败: {path}\n错误信息: {e}")


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
