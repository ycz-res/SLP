import subprocess
from utils import log
import argparse

parser = argparse.ArgumentParser(description='参数验证')
parser.add_argument('--func', default='ph14t_test', help='执行函数')
parser.add_argument('--dataset', default='P2SASLDataset', type=str,
                      choices=['P2SASLDataset', 'PH14TDataset'])
parser.add_argument('--p', type=float, default=0.28, metavar='RATE')
parser.add_argument('--n', type=float, default=0.05, metavar='RATE')
args = vars(parser.parse_args())


def p_range(start, stop, step):
    while round(start, 2) >= stop:
        yield round(start, 2)
        start -= step


def n_range(start, stop, step):
    while round(start, 2) <= stop:
        yield round(start, 2)
        start += step


# P2SASLDataset 数据集 不同p、n设置对实验的影响
def p2s_test():
    p_values = [round(x, 2) for x in p_range(0.58, 0.52, 0.01)]
    n_values = [round(x, 2) for x in n_range(0.05, 0.18, 0.01)]
    for p in p_values:
        for n in n_values:
            print(f"Running training with p/n: {p}/{n}")
            log('params_c', p=p, n=n, dataset='P2SASLDataset')

            result = subprocess.run([
                "python", "train_c.py",
                "--dataset", "P2SASLDataset",
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


# PH14TDataset 数据集 不同p、n设置对实验的影响
def ph14t_test():
    p_values = [round(x, 2) for x in p_range(0.19, 0.15, 0.01)]
    n_values = [round(x, 2) for x in n_range(0.05, 0.15, 0.01)]
    for p in p_values:
        for n in n_values:
            print(f"Running training with p/n: {p}/{n}")
            log('params_c', p=p, n=n, dataset='PH14TDataset')

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


# 测试情感注意力模块设置对实验的影响
def nl_test():
    num_layers = [1, 2, 3, 4, 5]

    for nl in num_layers:
        print(f"Running training with nl: {nl}")
        log('params_c', nl=nl, dataset=args['dataset'], p=args['p'], n=args['n'])

        result = subprocess.run([
            "python", "train_c.py",
            "--num_layers", str(nl),
            "--dataset", str(args['dataset']),
            "--prob", str(args['p']),
            "--noise_level", str(args['n']),
        ])

        # 检查是否执行成功
        if result.returncode == 0:
            print(f"Finished training with num_layers: {nl}")
            log(phase='pl_c')
        else:
            print(f"Error occurred while training with num_layers: {nl}")
            log(phase='params_c', msg='error')
            return  # 如果出错，终止后续的训练


if __name__ == "__main__":
    eval(args['func'])()

    # p_values = [round(x, 2) for x in p_range(0.52, 0.36, 0.01)]
    # n_values = [round(x, 2) for x in n_range(0.05, 0.15, 0.01)]
    # print(p_values)
    # print(n_values)
