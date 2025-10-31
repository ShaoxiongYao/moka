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

from string import ascii_lowercase

from openai import OpenAI
client = OpenAI()

# from moka.gpt_utils import request_gpt
# from moka.vision.segmentation import get_scene_object_bboxes
# from moka.vision.segmentation import get_segmentation_masks
# from moka.vision.keypoint import get_keypoints_from_segmentation
# from moka.planners.planner import Planner
from moka.planners.visual_prompt_utils import *
from pag_utils import process_objects, annotate_keypoints

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/moka.yaml')
    parser.add_argument('--color_img_path', type=str, default='/home/ydu/haowen/moka/example/camera1_rgb.png')
    parser.add_argument('--depth_img_path', type=str, default='/home/ydu/haowen/moka/example/camera1_depth.png')
    parser.add_argument('--task_instruction', type=str, default='Push the white coconut milk bottle to align with the Pringles.')
    parser.add_argument('--output_path', type=str, default=f'output/test_moka_{time.time()}')
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

    obs_image = Image.open('/home/ydu/haowen/moka/example/camera1_rgb.png').convert('RGB')
    depth_image = Image.open('/home/ydu/haowen/moka/example/camera1_depth.png')

    # Center crop to 512x512
    width, height = obs_image.size

    # Calculate crop box for center 512x512
    left = (width - 512) // 2
    top = (height - 512) // 2
    right = left + 512
    bottom = top + 512

    # Crop from center
    obs_image = obs_image.crop((left, top, right, bottom))

    output_path = '/home/ydu/haowen/moka/example/camera1_rgb_cropped.png'
    obs_image.save(output_path)

    obs = {'image': obs_image}
    # plt.imshow(obs_image)
    # plt.axis('off')
    # plt.show()

    task_instruction = 'Push the white coconut milk bottle to align with the Pringles.'
    print('Task: ', task_instruction)

    plan = request_plan(
        task_instruction,
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

    # print(all_object_names)
    # input()
    all_object_names = ['metal watch', 'white ultrasound cleaner', 'red button']


    from moka.vision.segmentation import get_scene_object_bboxes, get_segmentation_masks_sam2, mask_to_polygon

    # # get bounding boxes
    # boxes, logits, phrases = get_scene_object_bboxes(obs_image, all_object_names,
    #                                                  visualize=True, logdir='output')

    # # Get segmentation masks
    # segmasks = get_segmentation_masks_sam2(obs_image, all_object_names, boxes, logits, phrases, 
    #                                        visualize=True, logdir='output', 
    #                                        sam2_checkpoint='../Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt')

    mask1 = np.load('/home/ydu/haowen/moka/example/outputs_white coconut milk bottle.npy')

    segmasks = {
        'white coconut milk bottle': {'mask': mask1, 'vertices': mask_to_polygon(mask1)}, 
    }

    subtask = plan[0]
    # import pdb; pdb.set_trace()
    candidate_keypoints = propose_candidate_keypoints(subtask, segmasks, 
                                                    num_samples=config.num_candidate_keypoints)

    annotation_size = next(iter(segmasks.values()))['mask'].shape[:2][::-1] 
    obs_image_reshaped = obs_image.resize(annotation_size, Image.LANCZOS)

    annotated_image = annotate_visual_prompts(obs_image, candidate_keypoints,
                                            waypoint_grid_size=config.waypoint_grid_size)

    from moka.planners.visual_prompt_utils import request_motion

    context, _, _ = request_motion(
        subtask,
        obs_image,
        annotated_image,
        candidate_keypoints,
        waypoint_grid_size=config.waypoint_grid_size, 
        prompts=prompts, 
        debug=True
    )

    import pdb; pdb.set_trace()
