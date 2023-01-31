#!/usr/bin/env bash

set -x

EXP_DIR=logs_run_001
PY_ARGS=${@:1}

python -u main.py \
    --pretrained params/detr-r50-pre-vaw.pth \
    --run_name ${EXP_DIR} \
    --mtl \
    --batch_size 8 \
    --update_obj_att \
    --epochs 90 \
    --lr_drop 30 \
    --mtl_data [\'hico\',\'vcoco\',\'vaw\'] \
    --num_obj_classes 81 \
    --num_verb_classes 117 \
    --backbone resnet50 \
    --output_dir checkpoints/mtl/ \
    ${PY_ARGS}