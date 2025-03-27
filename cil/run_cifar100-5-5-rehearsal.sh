#!/bin/bash

# for CIFAR-100 dataset with MRFA-style rehearsal enabled
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_5-5-MoE-Adapters.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml" \
    +use_rehearsal=true \
    +memory_size=2000 \
    +memory_batch_size=32 \
    +rehearsal_ratio=0.2 \
    +augmentation_enabled=true \
    +perturb_factor=0.1 \
    +num_augmem=2 \
    "+perturb_p=[0.0001,0.0001,0.0001,0.0001]"
