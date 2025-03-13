import pandas as pd
import json
import yaml
from datetime import datetime
from colorama import init, Back
import logging
import os


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
        'kp_file': '' + row['FILE_NAME'] + '.json'
    } for _, row in data_raws.iterrows()]

    return res


# ------
# 日志
def write_log(filename, phase, **kwargs):
    if 'pl' in phase:
        content = '\n'
    else:
        # 获取当前时间
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 将kwargs解包并用空格隔开
        content = '[' + current_time + ']|' + phase + '|' + '|'.join(f"{key}={value}" for key, value in kwargs.items())
        # 追加写入到文件
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(content + '\n')
        print(f"{Back.GREEN}保存成功{Back.RESET}")


def log(phase, **kwargs):
    if '_s' in phase:
        write_log('log/s.txt', phase, **kwargs)
    elif '_e' in phase:
        write_log('log/e.txt', phase, **kwargs)
    elif '_c' in phase:
        write_log('log/c.txt', phase, **kwargs)
    else:
        print(f"{Back.RED}保存失败{Back.RESET}")


# 文本修正, 提高BLEU4分数
# 去除重复
def remove_duplicates(text):
    words = text.split()
    d_words = []
    seen = set()
    for word in words:
        if word not in seen:
            d_words.append(word)
            seen.add(word)
    return ' '.join(d_words)


if __name__ == '__main__':
    pass
