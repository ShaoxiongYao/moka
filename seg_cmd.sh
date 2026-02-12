#!/bin/bash

data_dir=/home/ydu/haowen/moka/outputs/0211_obstacle_moka_2
object_names=(
    # "metal watch"
    # "white ultrasound cleaner"
    # "red button"
    # "white coconut milk bottle"
    "coffee beans"
    "purple tape cage"
)

# Build text prompt from object_names array
object_prompt=$(printf "%s. " "${object_names[@]}")
object_prompt="${object_prompt% }"  # Remove trailing space

python ../real2sim/get_stream.py --snapshot --data_dir $data_dir

# echo "Data directory: $data_dir"
echo "Object prompt: $object_prompt"

# python ../real2sim/run_gsam2.py --text-prompt "$object_prompt" --img-path /home/ydu/haowen/moka/example/camera1_rgb_cropped.png --output-dir ${data_dir}

# python ../real2sim/mask_extraction.py --json /home/ydu/haowen/moka/example/outputs/camera1_rgb_cropped_gsam2.json --output ${data_dir} --png 
