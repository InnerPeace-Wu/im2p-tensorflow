#!/usr/bin/env bash

# Run with:
#       bash scripts/dense_cap_demo.sh [ckpt_path] [vocab_path]

set -x
set -e

ckpt=$1
vocab=$2

# For my own experiment usage, just ignore it.
if [ -d '/home/joe' ]; then
    ckpt='/home/joe/git/im2p/output/im2p_fixed/vg_1.2_train'
    vocab='/home/joe/git/visual_genome/im2p/vocabulary.txt'
fi

time python ./tools/demo.py \
    --ckpt ${ckpt} \
    --cfg  scripts/dense_cap_config.yml \
    --vocab ${vocab} \
    --set TEST.USE_BEAM_SEARCH False EMBED_DIM 512
