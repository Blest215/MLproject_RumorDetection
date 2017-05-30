#!/bin/bash

TENSORFLOW_TRAIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/train_convnet.py"
TENSORFLOW_EVAL="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/eval_convnet.py"

DATASET_DIR=/dataset
TRAIN_DIR=/summary
EVAL_DIR=${TRAIN_DIR}/eval

# Run TensorBoard
nohup tensorboard --logdir=${TRAIN_DIR} &> /dev/null &

# Run Tensorflow
python ${TENSORFLOW_TRAIN} \
--train_dir=${TRAIN_DIR} \
--batch_size=64 \
--dataset_split_name=train \
--dataset_dir=${DATASET_DIR} \
--save_summaries_secs=30 \
--save_interval_secs=30 \
--learning_rate=0.00001 \ &> /rumor/log.txt &

python ${TENSORFLOW_EVAL} \
--alsologtostderr \
--checkpoint_dir=${TRAIN_DIR} \
--eval_dir=${EVAL_DIR} \
--dataset_split_name=test \
--dataset_dir=${DATASET_DIR} \
--eval_interval_secs=30



