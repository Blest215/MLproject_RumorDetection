#!/bin/bash

TENSORFLOW_TRAIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/train_convnet.py"
TENSORFLOW_EVAL="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/eval_convnet.py"

DATASET_DIR=/dataset
TRAIN_DIR=/summary
EVAL_DIR=${TRAIN_DIR}/eval

# Run TensorBoard
nohup tensorboard --logdir=${TRAIN_DIR} &> /dev/null &

# Run Tensorflow
if [ $1 = "train" ]; then
    python ${TENSORFLOW_TRAIN} \
    --train_dir=${TRAIN_DIR} \
    --num_clones=2 \
    --batch_size=32 \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --learning_rate=0.045 \

elif [ $1 = "eval" ]; then
    python ${TENSORFLOW_EVAL} \
    --alsologtostderr \
    --checkpoint_dir=${TRAIN_DIR} \
    --eval_dir=${EVAL_DIR} \
    --dataset_split_name=test \
    --dataset_dir=${DATASET_DIR} \
else
    echo "Usage: run_tensorflow.sh [train|eval]"
    exit 1
fi



