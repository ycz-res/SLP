#!/bin/bash
# chmod +x pull.sh
REPO_URL="https://github.com/SoftpaseFar/SLP.git"

USERNAME="SoftpaseFar"
TOKEN="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"

CLONE_URL=$(echo $REPO_URL | sed "s#https://#https://$USERNAME:$TOKEN@#")

git pull $CLONE_URL