import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

# 创建存储图像的目录
output_dir = "./plot/res"
os.makedirs(output_dir, exist_ok=True)


def plot_ll():
    # 绘制论文图表
    # 绘制折线图 用于论文图表
    # 横坐标 lambda 值：  0	0.1	0.2	0.3	0.4	0.5	0.6	0.7	0.8	0.9	1
    # y_p14t 纵坐标 20.62	21.82	21.1	21.69	21.82	22.41	22.16	22.22	22.35	22.32	22.59
    # y_how2sign 纵坐标 20	21	21	21	21	22	22	22.2	22.35	22.3	22.19

    x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    y_p14t = np.array([20.62, 21.82, 21.1, 21.69, 21.82, 22.41, 22.16, 22.22, 22.35, 22.32, 21.59])
    y_how2sign = np.array([17.79, 20.35, 20.06, 20.8, 20.86, 21.46, 21.12, 21.18, 20.93, 21.23, 20.48])

    plt.plot(x, y_p14t, label='P14T')
    plt.plot(x, y_how2sign, label='P2S(ASL)')
    plt.xlabel('lambda')
    plt.ylabel('BLEU-4')
    plt.legend()

    # 保存图像
    save_path = os.path.join(output_dir, "plot_ll.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved {save_path}")
    plt.close()


def plot_nl_pre():
    # 数据
    x = np.array([1, 2, 3, 4, 5])
    y_p14t = np.array([11.05, 12.73, 11.67, 11.56, 10.94])
    y_how2sign = np.array([20.11, 19.97, 20.52, 19.74, 19.92])

    width = 0.3  # 柱状图宽度

    # 适合科研论文的颜色（无边框）
    color_p14t = '#FFA500'  # 深蓝
    color_how2sign = '#C21E56'  # 深橙

    plt.figure(figsize=(6, 4))  # 控制图表大小
    plt.bar(x - width / 2, y_p14t, width=width, label='PH14T', color=color_p14t)
    plt.bar(x + width / 2, y_how2sign, width=width, label='P2S(ASL)', color=color_how2sign)

    # 添加标签
    plt.xlabel('Number of Layers', fontsize=12)
    plt.ylabel('BLEU-4', fontsize=12)
    plt.xticks(x, fontsize=10)  # 确保横坐标对齐
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10, frameon=False)

    # 调整边距
    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(output_dir, "plot_nl.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved {save_path}")
    plt.close()

def plot_nl():
    # 数据
    x = np.array([1, 2, 3, 4, 5])
    y_p14t = np.array([11.05, 12.73, 11.67, 11.56, 10.94])
    y_how2sign = np.array([20.11, 19.97, 20.52, 19.74, 19.92])

    width = 0.3  # 柱状图宽度

    # 适合科研论文的颜色（无边框）
    color_p14t = '#FFA500'  # 橙色
    color_how2sign = '#C21E56'  # 深红色

    plt.figure(figsize=(6, 4))  # 控制图表大小
    plt.bar(x - width / 2, y_p14t, width=width, label='PH14T', color=color_p14t)
    plt.bar(x + width / 2, y_how2sign, width=width, label='P2S(ASL)', color=color_how2sign)

    # 添加标签
    plt.xlabel('Number of Layers', fontsize=12)
    plt.ylabel('BLEU-4', fontsize=12)
    plt.xticks(x, fontsize=10)
    plt.yticks(fontsize=10)

    # 图例放外侧上方
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=10, frameon=False, ncol=2)

    # 调整边距
    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(output_dir, "plot_nl.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # bbox_inches 确保图例不会被裁剪
    plt.show()
    print(f"Saved {save_path}")
    plt.close()


def plot_lr():
    # 创建 DataFrame
    data = pd.DataFrame(
        [[21.64, 19.74, 19.19],  # P14T
         [20.79, 18.35, 18.02]],  # How2Sign
        index=["P14T", "How2Sign"],
        columns=["0.001", "0.0001", "1e-5"]
    )

    # 绘制热力图
    plt.figure(figsize=(10, 5))
    sns.heatmap(data, annot=True, cmap="YlOrRd", fmt=".2f", linewidths=0.5)

    plt.xlabel("Learning Rate")
    plt.ylabel("Dataset")

    # 调整图像的布局，使其居中
    plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.1)

    # 保存图像
    save_path = os.path.join(output_dir, "plot_lr.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved {save_path}")
    plt.close()


if __name__ == '__main__':
    # plot_ll()
    plot_nl()
    # plot_lr()
