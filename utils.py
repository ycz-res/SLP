import pandas as pd
import json


# 访问 json 文件
def read_json_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()

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
