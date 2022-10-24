"""
detectron2.py
Using detectron2 to generate custom visual embeddings
author: @alexiskaldany, @justjoshtings
created: 10/24/22
"""

from copy import deepcopy
import json
import os
from loguru import logger
import sys
import random
import pandas as pd

# get current directory
path = os.getcwd()
# parent_path = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
# add src to executable path to allow imports from src
sys.path.insert(0, path)
from src.utils.prepare_and_download import get_data_objects, create_dataframe
# print(DATA_DIRECTORY)
# from src.utils.answer_filtering import has_only_one_word_answers
from src.utils.configs import DATA_JSON, DATA_CSV, DATA_DIRECTORY, ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER, ANNOTATED_IMAGES_FOLDER
# """
# Set logger
# """
# logger.remove()
# logger.add(
#     # "./logs/training_log.txt",
#     sys.stdout,
#     format="{time:YYYY-MM-DD HH:mm:ss}|{level}| {message}|{function}: {line}",
#     level="DEBUG",
#     backtrace=True,
#     colorize=True,
# )
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import json
import detectron2
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.config import get_cfg
from PIL import Image



combined_list = get_data_objects(ANNOTATION_FOLDER, IMAGES_FOLDER,QUESTIONS_FOLDER )
data_df = create_dataframe(combined_list)

""" 
Loading pretrained model
"""
cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

def load_config_and_model_weights(cfg_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

    # ROI HEADS SCORE THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Comment the next line if you're using 'cuda'
    cfg['MODEL']['DEVICE']='cpu'

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

    return cfg

cfg = load_config_and_model_weights(cfg_path)

""" 
Build Model
"""
def get_model(cfg):
    # build model
    model = build_model(cfg)

    # load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # eval mode
    model.eval()
    return model

model = get_model(cfg)

""" 
Load and preprocess image
"""

def load_image_resize_convert(image_path):
    preprocess = transforms.Compose([
    transforms.ToTensor(),transforms.Resize(size=(224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = preprocess(Image.open(image_path))
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

""" 
Get features
"""

def get_features(model, images):
    features = model.backbone(images.tensor)
    return features

features = get_features(model, load_image_resize_convert(data_df['image_path'][0]))
print(features['p2'].shape)