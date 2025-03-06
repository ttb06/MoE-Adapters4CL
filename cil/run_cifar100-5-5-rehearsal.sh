#!/bin/bash

# for CIFAR-100 dataset with rehearsal enabled
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_5-5-MoE-Adapters.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml" \
    +use_rehearsal=true \
    +memory_size=2000 \
    +memory_batch_size=32 \
    +rehearsal_ratio=0.3 \
    +perturb_factor=0.1 \
    +augmentation_enabled=true
