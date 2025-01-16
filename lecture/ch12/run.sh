#!/bin/bash

model_list=(
    "tf_efficientnet_b0.ns_jft_in1k"
    "tf_efficientnet_b5.ns_jft_in1k"
    "tf_efficientnet_b7.ns_jft_in1k"
    "resnet18"
    "resnet26"
    "resnet34"
    "resnet50"
    "resnet101"
    "resnet152"
    "vgg11"
    "vgg13"
    "vgg16"
    "vgg19"
    "swin_tiny_patch4_window7_224"
    "swin_small_patch4_window7_224"
    "swin_base_patch4_window7_224"
    "swin_large_patch4_window7_224"
    "swin_s3_tiny_224"
    "swin_s3_small_224"
    "swin_s3_base_224"
    "vit_tiny_patch16_224"
    "vit_small_patch16_18x2_224"
    "vit_large_patch16_224"
)

for model in "${model_list[@]}"; do
    echo "Running script for model: $model"
    if [[ $model == *"efficientnet"* ]]; then
        python3 ch12-baseline.py --model "$model" --img_size 450 640
    elif [[ $model == *"resnet"* ]]; then
        python3 ch12-baseline.py --model "$model" --img_size 450 640
    elif [[ $model == *"vgg"* ]]; then
        python3 ch12-baseline.py --model "$model" --img_size 450 640
    else
        python3 ch12-baseline.py --model "$model" --img_size 224 224
    fi
done