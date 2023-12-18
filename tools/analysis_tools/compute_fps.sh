#!/usr/bin/env bash
CONFIG=$1
CHECKPOINT=$2
SAVE_DIR=$3

python /media/ubuntu/nvidia/wlq/part1_tiny_detection/mmrotate-1.x/tools/analysis_tools/benchmark.py \
${CONFIG} --checkpoint ${CHECKPOINT} \
--task inference