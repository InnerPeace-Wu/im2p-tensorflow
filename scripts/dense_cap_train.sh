#!/usr/bin/env bash

# Run with:
#       bash scripts/dense_cap_train.sh [dataset] [net] [ckpt_to_init] [data_dir] [step]

set -x
set -e

export PYTHONUNBUFFERED='True'

DATASET=$1
NET=$2
ckpt_path=$3
data_dir=$4
step=$5

# For my own experiment usage, just ignore it.
if [ -d '/home/joe' ]; then
    DATASET='visual_genome_1.2'
    NET='res50'
    ckpt_path="experiments/densecap"
    data_dir='/home/joe/git/visual_genome/im2p'
fi

case $DATASET in
  visual_genome_1.2)
    TRAIN_IMDB="vg_1.2_train"
    TEST_IMDB="vg_1.2_val"
    FINETUNE_AFTER=50000
    ITERS=100000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

# This is for valohai computing platform, one can just ignore it.
if [ -d '/valohai/outputs' ]; then
    ckpt_path='/valohai/inputs/resnet'
    data_dir='/valohai/inputs/visual_genome'
    LOG="/valohai/outputs/s${step}_${NET}_${TRAIN_IMDB}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
else
    LOG="logs/s${step}_${NET}_${TRAIN_IMDB}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
fi

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# First step, freeze conv nets weights
if [ ${step} -lt '2' ]
then
time python ./tools/train_net.py \
    --weights ${ckpt_path} \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters ${FINETUNE_AFTER}\
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set EXP_DIR im2p_fixed IM2P.FINETUNE False
fi

# Step2: Finetune convnets
NEW_WIGHTS=output/im2p_fixed/${TRAIN_IMDB}
if [ ${step} -lt '3' ]
then
time python ./tools/train_net.py \
    --weights ${NEW_WIGHTS} \
    --imdb ${TRAIN_IMDB} \
    --iters `expr ${ITERS} - ${FINETUNE_AFTER}` \
    --imdbval ${TEST_IMDB} \
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set EXP_DIR im2p_finetune RESNET.FIXED_BLOCKS 1 IM2P.FINETUNE True
fi
