import argparse
import json
import os
import cv2
import numpy as np


def create_black_img():
    # 设置图片的宽度和高度
    width = 1280
    height = 720

    # 创建一个黑色背景的图像
    # 参数 (height, width, 3) 分别代表高度、宽度和颜色通道数（RGB）
    black_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 保存图片
    if not os.path.exists('./data/res'):
        os.makedirs('./data/res')
    cv2.imwrite('./data/res/black.jpg', black_image)


# python.exe main.py --directory jsons
# 骨骼关键点连接对
pose_pairs = [
    [0, 1], [0, 15], [0, 16],
    [15, 17],
    [16, 18],
    [1, 2], [1, 5], [1, 8],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [8, 9], [8, 12],

    [9, 10],
    [10, 11],
    [11, 22], [11, 24],
    [22, 23],
    [12, 13],
    [13, 14],
    [14, 21], [14, 19],
    [19, 20]
]
# 手部关键点连接对
hand_pairs = [
    [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
    [1, 2],
    [2, 3],
    [3, 4],
    [5, 6], [6, 7], [7, 8],
    [9, 10], [10, 11], [11, 12],

    [13, 14], [14, 15], [15, 16],
    [17, 18], [18, 19], [19, 20]
]
# 绘制用的颜色
pose_colors = [
    (255., 0., 85.), (255., 0., 0.), (255., 85., 0.), (255., 170., 0.),
    (255., 255., 0.), (170., 255., 0.), (85., 255., 0.), (0., 255., 0.),
    (255., 0., 0.), (0., 255., 85.), (0., 255., 170.), (0., 255., 255.),
    (0., 170., 255.), (0., 85., 255.), (0., 0., 255.), (255., 0., 170.),
    (170., 0., 255.), (255., 0., 255.), (85., 0., 255.), (0., 0., 255.),
    (0., 0., 255.), (0., 0., 255.), (0., 255.,
                                     255.), (0., 255., 255.), (0., 255., 255.)
]
hand_colors = [
    (100., 100., 100.),
    (100, 0, 0),

    (150, 0, 0),
    (200, 0, 0), (255, 0, 0), (100, 100, 0), (150,
                                              150, 0), (200, 200, 0), (255, 255, 0),
    (0, 100, 50), (0, 150, 75), (0, 200, 100), (0,
                                                255, 125), (0, 50, 100), (0, 75, 150),
    (0, 100, 200), (0, 125, 255), (100, 0, 100), (150, 0, 150),
    (200, 0, 200), (255, 0, 255)
]


def handle_json(jsonfile):
    print('json file: {}'.format(jsonfile))
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    # 纯黑色背景
    img = cv2.imread('./data/res/black.jpg')

    for d in data['people']:
        kpt = np.array(d['pose_keypoints_2d']).reshape((25, 3))
        for p in pose_pairs:
            pt1 = tuple(list(map(int, kpt[p[0], 0:2])))
            c1 = kpt[p[0], 2]
            pt2 = tuple(list(map(int, kpt[p[1], 0:2])))
            c2 = kpt[p[1], 2]
            print('== {}, {}, {}, {} =='.format(pt1, c1, pt2, c2))
            if c1 == 0.0 or c2 == 0.0:
                continue
            color = tuple(list(map(int, pose_colors[p[0]])))
            img = cv2.line(img, pt1, pt2, color, thickness=4)
            img = cv2.circle(img, pt1, 4, color, thickness=-
            1, lineType=8, shift=0)

            img = cv2.circle(img, pt2, 4, color, thickness=-
            1, lineType=8, shift=0)
        kpt_left_hand = np.array(d['hand_left_keypoints_2d']).reshape((21, 3))
        for q in hand_pairs:
            pt1 = tuple(list(map(int, kpt_left_hand[q[0], 0:2])))
            c1 = kpt_left_hand[p[0], 2]
            pt2 = tuple(list(map(int, kpt_left_hand[q[1], 0:2])))
            c2 = kpt_left_hand[q[1], 2]
            # print('** {}, {}, {}, {} **'.format(pt1, c1, pt2, c2))
            if c1 == 0.0 or c2 == 0.0:
                continue
            color = tuple(list(map(int, hand_colors[q[0]])))
            img = cv2.line(img, pt1, pt2, color, thickness=4)
        kpt_right_hand = np.array(
            d['hand_right_keypoints_2d']).reshape((21, 3))
        for k in hand_pairs:
            pt1 = tuple(list(map(int, kpt_right_hand[k[0], 0:2])))
            c1 = kpt_right_hand[k[0], 2]
            pt2 = tuple(list(map(int, kpt_right_hand[k[1], 0:2])))
            c2 = kpt_right_hand[k[1], 2]
            print('** {}, {}, {}, {} **'.format(pt1, c1, pt2, c2))
            if c1 == 0.0 or c2 == 0.0:
                continue
            color = tuple(list(map(int, hand_colors[q[0]])))
            img = cv2.line(img, pt1, pt2, color, thickness=4)
    if not os.path.exists('./data/res'):
        os.makedirs('./data/res')
    # 保存图片
    cv2.imwrite('./data/res/{}.jpg'.format(jsonfile.split("/")[-1][0:-5]), img)


if __name__ == '__main__':
    # create_black_img()

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str,
                        default='.', help='keypoints json directory')
    opt = parser.parse_args()
    for jsonfile in os.listdir(opt.directory):
        if jsonfile.endswith('.json'):
            handle_json(os.path.join(opt.directory, jsonfile))
