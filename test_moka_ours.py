# Basic dependencies

import os 
import os.path
import io 
import traceback
from absl import app
from absl import flags

import time
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import yaml
from PIL import Image
from easydict import EasyDict as edict
from pathlib import Path

# MOKA utilities

import open3d as o3d
from string import ascii_lowercase

from openai import OpenAI
client = OpenAI()

# from moka.gpt_utils import request_gpt
# from moka.vision.segmentation import get_scene_object_bboxes
# from moka.vision.segmentation import get_segmentation_masks
# from moka.vision.keypoint import get_keypoints_from_segmentation
# from moka.planners.planner import Planner
from moka.planners.visual_prompt_utils import *
from pag_utils import process_objects, annotate_keypoints, \
            visualize_point_cloud_with_keypoints, create_point_array_from_rgbd,\
            get_3d_points_from_backprojected, plot_2d_points
from moka.vision.segmentation import mask_to_polygon
from moka.planners.visual_prompt_utils import request_motion

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from real2sim.estimate_scale import load_intrinsics_from_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/moka.yaml')
    parser.add_argument('--color_img_path', type=str, 
                        default='/home/ydu/haowen/real2sim/data/1107_pivot_0/camera1_rgb.png')
    parser.add_argument('--depth_img_path', type=str, 
                        default='/home/ydu/haowen/real2sim/data/1107_pivot_0/camera1_depth.npy')
    parser.add_argument('--task_instruction', type=str, 
                        # default='Push the white coconut milk bottle to align with the Pringles.')
                        default='Make the red pocky box lean vertically against the brown box.')
    parser.add_argument('--intrinsics_path', type=str, 
                        default="/home/ydu/haowen/real2sim/cam_utils/cam1_intrinsics.txt")
    parser.add_argument('--output_path', type=str, default=f'output/test_moka_{int(time.time())}')
    args = parser.parse_args()

    import os
    config_filename = './config/moka.yaml'
    with open(config_filename, 'r') as fh:
        config = yaml.load(fh, Loader=yaml.SafeLoader)
        config = edict(config)
    
    Path(args.output_path).mkdir(exist_ok=True)

    def load_prompts():
        """Load prompts from files.
        """
        prompts = dict()
        prompt_dir = os.path.join(
            config.prompt_root_dir, config.prompt_name)
        for filename in os.listdir(prompt_dir):
            path = os.path.join(prompt_dir, filename)
            if os.path.isfile(path) and path[-4:] == '.txt':
                with open(path, 'r') as f:
                    value = f.read()
                key = filename[:-4]
                prompts[key] = value
        return prompts
        
    prompts = load_prompts()

    obs_image = Image.open(args.color_img_path).convert('RGB')
    depth_image = np.load(args.depth_img_path)

    rgb1_intrinsics, depth1_intrinsics = load_intrinsics_from_file(args.intrinsics_path)

    # Back-project ALL points first using original intrinsics
    points = create_point_array_from_rgbd(obs_image, depth_image, rgb1_intrinsics)

    # Now crop the image and select corresponding points
    width, height = obs_image.size

    # left = max(0, (width - 512) // 2)
    # top = max(0, (height - 512) // 2)
    # right = min(width, left + 512)
    # bottom = min(height, top + 512)

    # Determine crop size (use smallest dimension if < 512, otherwise 512)
    crop_size = min(512, width, height)

    # Calculate center crop
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    print(f"Crop box: left={left}, top={top}, right={right}, bottom={bottom}")
    print(f"Crop size: {right-left} x {bottom-top}")

    # Crop images
    obs_image = obs_image.crop((left, top, right, bottom))
    depth_image_cropped = depth_image[top:bottom, left:right]
    points_cropped = points[top:bottom, left:right]

    print(f"Cropped image size: {obs_image.size}")
    print(f"Cropped points shape: {points_cropped.shape}")

    # Crop the point cloud to match
    # Assuming points is shaped (H, W, 3) or similar
    points_cropped = points[top:bottom, left:right]

    color_output_path = os.path.join(args.output_path, 'camera1_rgb_cropped.png')
    obs_image.save(color_output_path)
    depth_output_path = os.path.join(args.output_path, 'camera1_depth_cropped.npy')
    np.save(depth_output_path, depth_image_cropped)

    obs = {'image': obs_image}
    # plt.imshow(obs_image)
    # plt.axis('off')
    # plt.show()

    print('Task: ', args.task_instruction)

    plan = request_plan(
        args.task_instruction,
        obs_image, 
        plan_with_obs_image=True,
        prompts=prompts,
        debug=True)

    all_object_names = []
    for subtask in plan:
        if subtask['object_grasped'] != '' and subtask['object_grasped'] not in all_object_names:
            all_object_names.append(subtask['object_grasped'])

        if subtask['object_unattached'] != '' and subtask['object_unattached'] not in all_object_names:
            all_object_names.append(subtask['object_unattached'])

    print(all_object_names)
    process_objects(all_object_names, args.output_path, 
                    img_path=color_output_path)

    # Dynamically load masks and generate segmasks dictionary
    segmasks = {}
    dir_name = os.path.basename(os.path.normpath(args.output_path))
    output_prefix = os.path.join(args.output_path, dir_name)

    for object_name in all_object_names:
        # Construct the mask file path
        mask_path = f"{output_prefix}_{object_name}.npy"
        import pdb; pdb.set_trace()
        
        # Load the mask
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            segmasks[object_name] = {
                'mask': mask, 
                'vertices': mask_to_polygon(mask)
            }
            print(f"Loaded mask for: {object_name}")
        else:
            print(f"Warning: Mask not found at {mask_path}")

    subtask = plan[0]
    # import pdb; pdb.set_trace()
    candidate_keypoints = propose_candidate_keypoints(subtask, segmasks, 
                                                      num_samples=config.num_candidate_keypoints)

    annotation_size = next(iter(segmasks.values()))['mask'].shape[:2][::-1] 
    obs_image_reshaped = obs_image.resize(annotation_size, Image.LANCZOS)

    annotated_image = annotate_visual_prompts(obs_image, candidate_keypoints,
                                              waypoint_grid_size=config.waypoint_grid_size)

    context, _, _ = request_motion(
        subtask,
        obs_image,
        annotated_image,
        candidate_keypoints,
        waypoint_grid_size=config.waypoint_grid_size, 
        prompts=prompts, 
        debug=True
    )

    keypoints_img = annotate_keypoints(obs_image, context)
    keypoints_img.save(os.path.join(args.output_path, 'keypoints_visualization.png'))

    result = get_3d_points_from_backprojected(points_cropped, context, window_size=10)

    plot_2d_points(points_cropped, context)

    visualize_point_cloud_with_keypoints(points_cropped, obs_image, result)