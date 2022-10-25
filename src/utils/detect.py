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
from pathlib import Path
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
import torch.nn as nn
from PIL import Image
from torchvision.models.resnet import resnet18 as _resnet18



# combined_list = get_data_objects(ANNOTATION_FOLDER, IMAGES_FOLDER,QUESTIONS_FOLDER )
# data_df = create_dataframe(combined_list)

""" 
Needs to be (1, 512)
"""
# model = _resnet18(weights='DEFAULT')
# modules=list(model.children())[:-1]
# model=nn.Sequential(*modules)

def get_visual_embeddings(image_path:str)-> torch.Tensor:
    image_path = image_path[0]
    image_id = Path(image_path).name.split(".")[0]
    preprocess = transforms.Compose([
    transforms.ToTensor(),transforms.Resize(size=(224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = preprocess(Image.open(image_path))
    input_batch = input_tensor.unsqueeze(0)
    model = _resnet18(weights='DEFAULT')
    modules=list(model.children())[:-1]
    model=nn.Sequential(*modules)
    embeddings = model(input_batch)
    embeddings = embeddings[:,:,0,0]
    embedding_dict = {image_id:embeddings}
    return embedding_dict


# image_paths = data_df['image_path']
# torch.cuda.empty_cache()
# model.eval()

# embeddings = get_visual_embeddings(image_paths[443])
# print(type(embeddings))
# print(embeddings.shape)

""" 
Get features
"""

# def get_features(model, images):
#     features = model.backbone(images.tensor)
#     return features

# features = get_features(model, load_image_resize_convert(data_df['image_path'][0]))
# print(features['p2'].shape)