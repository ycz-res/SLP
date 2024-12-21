import os
from utils import read_json_file
import json


def extract_keypoints(data, key):
    """Extract keypoints by removing every third value (z-coordinates)."""
    keypoints = data['people'][0][key]
    return [keypoints[i] for i in range(len(keypoints)) if i % 3 != 2]


def keypoints_preprocess(src_path, res_path):
    """Gather keypoints from JSON files organized in subdirectories."""
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    for sub_dir in os.listdir(src_path):
        sub_dir_path = os.path.join(src_path, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue

        video_keypoints = []

        for json_name in os.listdir(sub_dir_path):
            if not json_name.endswith('.json'):
                continue

            json_path = os.path.join(sub_dir_path, json_name)
            data = read_json_file(json_path)

            # Extract and combine keypoints
            pose_keypoints = extract_keypoints(data, 'pose_keypoints_2d')
            face_keypoints = extract_keypoints(data, 'face_keypoints_2d')
            hand_left_keypoints = extract_keypoints(data, 'hand_left_keypoints_2d')
            hand_right_keypoints = extract_keypoints(data, 'hand_right_keypoints_2d')

            video_keypoints.append(pose_keypoints + face_keypoints + hand_left_keypoints + hand_right_keypoints)

        # print(len(video_keypoints[1]))
        # 写入文件
        filename = os.path.join(res_path, sub_dir + '.json')
        with open(filename, 'w') as f:
            json.dump(video_keypoints, f, indent=4)
        print('current processing file: ', sub_dir + '.json')


if __name__ == '__main__':
    keypoints_preprocess('../src', '../keypoints')
