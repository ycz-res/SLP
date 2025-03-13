import subprocess
from utils import log
import argparse

parser = argparse.ArgumentParser(description='参数验证')
parser.add_argument('--func', default='pn_test', help='执行函数')
args = vars(parser.parse_args())


def frange(start, stop, step):
    while round(start, 2) <= stop:
        yield round(start, 2)
        start += step


# PH14TDataset 数据集 不同p、n设置对实验的影响
def ph14t_pn_test():
    p_values = [round(x, 2) for x in frange(0.36, 0.52, 0.01)]
    n_values = [round(x, 2) for x in frange(0.05, 0.15, 0.01)]
    for p in p_values:
        for n in n_values:
            print(f"Running training with p/n: {p}/{n}")
            log('params_c', p=p, n=n)

            result = subprocess.run([
                "python", "train_c.py",
                "--dataset", "PH14TDataset",
                "--prob", str(p),
                "--noise_level", str(n),
            ])

            # 检查是否执行成功
            if result.returncode == 0:
                print(f"Finished training with p/n: {p}/{n}")
                log(phase='pl_c')
            else:
                print(f"Error occurred while training with p/n: {p}/{n}")
                log(phase='params_c', p=p, n=n, msg='error')
                return  # 如果出错，终止后续的训练


# P2SASLDataset


if __name__ == "__main__":
    eval(args['func'])()
