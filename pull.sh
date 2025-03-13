#!/bin/bash

# 定义GitHub仓库的URL
REPO_URL="https://github.com/SoftpaseFar/SLP.git"  # 替换为你的实际仓库路径

# 定义用户名和token
USERNAME="SoftpaseFar"
TOKEN="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"

# 使用curl命令生成一个带有token的克隆链接
CLONE_URL=$(echo $REPO_URL | sed "s#https://#https://$USERNAME:$TOKEN@#")

# 执行git pull命令
git pull $CLONE_URL