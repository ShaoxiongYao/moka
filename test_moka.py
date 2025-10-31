# Basic dependencies

import os 
import os.path
import io 
import traceback
from absl import app
from absl import flags

import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import yaml
from PIL import Image
from easydict import EasyDict as edict

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

import os
config_filename = './config/moka.yaml'
with open(config_filename, 'r') as fh:
    config = yaml.load(fh, Loader=yaml.SafeLoader)
    config = edict(config)

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

obs_image = Image.open('example/obs_image.jpg').convert('RGB')
obs_image = obs_image.resize([512, 512], Image.LANCZOS)

obs = {'image': obs_image}
# plt.imshow(obs_image)
# plt.axis('off')
# plt.show()

task_instruction = 'Use the white ultrasound cleaner to clean the metal watch. The unstrasound cleaner has no lid and can be turned on by pressing the red button.'
print('Task: ', task_instruction)

plan = request_plan(
    task_instruction,
    obs_image, 
    plan_with_obs_image=True,
    prompts=prompts,
    debug=True)

# all_object_names = []
# for subtask in plan:
#     if subtask['object_grasped'] != '' and subtask['object_grasped'] not in all_object_names:
#         all_object_names.append(subtask['object_grasped'])

#     if subtask['object_unattached'] != '' and subtask['object_unattached'] not in all_object_names:
#         all_object_names.append(subtask['object_unattached'])

# print(all_object_names)

all_object_names = ['metal watch', 'white ultrasound cleaner', 'red button']

from moka.vision.segmentation import get_scene_object_bboxes, get_segmentation_masks_sam2, mask_to_polygon

# # get bounding boxes
# boxes, logits, phrases = get_scene_object_bboxes(obs_image, all_object_names,
#                                                  visualize=True, logdir='output')

# # Get segmentation masks
# segmasks = get_segmentation_masks_sam2(obs_image, all_object_names, boxes, logits, phrases, 
#                                        visualize=True, logdir='output', 
#                                        sam2_checkpoint='../Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt')

mask1 = np.load('/home/ydu/haowen/real2sim/test_moka/test_moka_metal watch.npy')
mask2 = np.load('/home/ydu/haowen/real2sim/test_moka/test_moka_white ultrasound cleaner.npy')
mask3 = np.load('/home/ydu/haowen/real2sim/test_moka/test_moka_red button.npy')

segmasks = {
    'metal watch': {'mask': mask1, 'vertices': mask_to_polygon(mask1)}, 
    'ultrasound cleaner': {'mask': mask2, 'vertices': mask_to_polygon(mask2)}, 
    'red button': {'mask': mask3, 'vertices': mask_to_polygon(mask3)}
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
