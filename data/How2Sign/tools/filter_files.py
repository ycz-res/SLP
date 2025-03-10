import os
import json


def preprocess(src_path, tgt_path):
    # 遍历src_path下所有文件夹
    folder_names = os.listdir(src_path)
    total_folders = len(folder_names)
    success_folders = 0

    for idx, name in enumerate(folder_names, start=1):
        folder_path = os.path.join(src_path, name)
        if os.path.isdir(folder_path):
            print(f"正在处理第{idx}个文件夹：{name}")
            # 读取文件夹下的alphapose-results.json文件
            json_file_path = os.path.join(folder_path, 'alphapose-results.json')
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as f:
                    data = json.load(f)

                # 提取所有的joints
                joints_list = []
                for img_name in data:
                    for body in data[img_name]['bodies']:
                        joints_list.append(body['joints'])

                # 存储为nametgt_path.json
                tgt_file_path = os.path.join(tgt_path, f'{name}.json')
                with open(tgt_file_path, 'w') as f:
                    json.dump(joints_list, f)
                success_folders += 1
            else:
                print(f"文件夹{name}下没有找到alphapose-results.json文件，跳过处理")
        else:
            print(f"{name}不是文件夹，跳过处理")

    print(f"总共处理了{total_folders}个文件夹，成功处理了{success_folders}个文件夹")


if __name__ == '__main__':
    preprocess('../src', '../tgt')
