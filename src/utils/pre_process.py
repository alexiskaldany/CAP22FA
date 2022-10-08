"""
pre_process.py
author: @alexiskaldany, @justjoshtings
created: 9/23/22
"""
from pathlib import Path
import glob
import json
from PIL import Image
from transformers import (
    DeiTFeatureExtractor,
)
import torch
from src.utils.configs import *
import random
from loguru import logger

""" 
Data
"""

DATA_JSON = DATA_DIRECTORY / "question_and_answers_dict_id_key.json"
""" 
Loading all the file paths
"""

# def get_data_objects(ANNOTATION_FOLDER,IMAGES_FOLDER,QUESTIONS_FOLDER):
#     image_glob = glob.glob(str(IMAGES_FOLDER / "*.png"))
#     image_ids = [int(Path(image_path).stem) for image_path in image_glob]
#     image_path_dict = {k:v for k,v in zip(image_ids, image_glob)}
#     annotation_glob = glob.glob(str(ANNOTATION_FOLDER / "*.png.json"))
#     annotation_ids = [
#         int(annotation_path.split(".")[-3].split("/")[-1])
#         for annotation_path in annotation_glob
#     ]
#     annotation_path_dict = {k:v for k,v in zip(annotation_ids, annotation_glob)}
#     questions_glob = glob.glob(str(QUESTIONS_FOLDER / "*.png.json"))
#     question_ids = [
#         int(question_path.split(".")[-3].split("/")[-1]) for question_path in questions_glob
#     ]
#     question_path_dict = {k:v for k,v in zip(question_ids, questions_glob)}
#     id_list = set(image_ids + annotation_ids + question_ids)
#     id_index = [(id, index) for index, id in enumerate(id_list)]
#     print(f"Number of images: {len(image_ids)}")
#     combined_list = []
#     for id in id_list:
#         image_dict = {"image_path": image_path_dict[id]}
#         with open(annotation_path_dict[id], "r") as f:
#             annotation_dict = json.load(f)
#         with open(question_path_dict[id], "r") as f:
#             questions_dict = json.load(f)
#         temp_list = [image_dict, annotation_dict, questions_dict]
#         print(temp_list)
#         combined_list.append(temp_list)   
#     return combined_list

""" 
Processing the images
"""
image_dimensions = (620, 480)
def load_image_resize_convert(image_path):
    image = Image.open(image_path)
    image = image.resize(image_dimensions)
    image = image.convert("RGB")
    return image


""" 
Image Feature Extraction
"""
# print(f"Average image dimensions: {average_dims}")
# print(f"Image dimensions: {image_dimensions[:10]}")
visual_feature_extractor = DeiTFeatureExtractor.from_pretrained(
    "facebook/deit-base-distilled-patch16-224"
)
def get_image_features(image_path,visual_feature_extractor):
    image = load_image_resize_convert(image_path)
    image_feature = visual_feature_extractor(
        images=image, return_tensors="pt", do_normalize=True
    )
    # pixel_values — Pixel values to be fed to a model, of shape (batch_size, num_channels, height, width).
    # image_features_shape = [*image_feature.pixel_values.shape]
    return image_feature
"""
Loading Questions and Answers
"""
# def generate_question_id_path_dict(question_folder_path):
#     questions_glob = glob.glob(str(QUESTIONS_FOLDER / "*.png.json"))
#     question_ids = [
#         int(question_path.split(".")[-3].split("/")[-1]) for question_path in questions_glob
#     ]
#     question_path_dict = {k:v for k,v in zip(question_ids, questions_glob)}
#     return question_path_dict
# def get_question_and_answers(id,path):
#     with open(path, "r") as f:
#         questions_dict = json.load(f)
#     question_list = []
#     for key, value in questions_dict["questions"].items():
#         question_dict = {
#             "image_id": id,
#             "image_path": image_path_dict[id],
#             "question_id": value["questionId"],
#             "question": key,
#             "answer": value["answerTexts"][value["correctAnswer"]],
#             "correct_answer_id": value["correctAnswer"],
#             "answer_type": value["abcLabel"],
#             "answer_choices": value["answerTexts"],
#         }
#         question_list.append(question_dict)
#     return question_list

# def question_and_answers_dict(question_path_dict):
#     id_question_and_answers_list = []
#     for k,v in question_path_dict.items():
#         id_question_and_answers_list.extend(get_question_and_answers(k,v))
#     return id_question_and_answers_list

# TRAIN,VAL,TEST 

def create_train_val_test_split(data_df):
    '''
    Params:
        data_df (pandas dataframe): dataframe of full dataset to split 
    Returns:
        train_df (pandas dataframe): dataframe of split train dataset
        val_df (pandas dataframe): dataframe of split val dataset
        test_df (pandas dataframe): dataframe of split test dataset
    '''
    logger.info("Creating train, val, test split")
    train_index = int(0.7 * len(data_df))
    val_index = int(0.9 * len(data_df))
    data_df_shuffled = data_df.sample(frac = 1, random_state=RANDOM_STATE)

    train_df = data_df_shuffled.iloc[:train_index]
    val_df = data_df_shuffled.iloc[train_index:val_index]
    test_df = data_df_shuffled.iloc[val_index:]
    
    logger.info(f"Train: {len(train_df)}")
    logger.info(f"Val: {len(val_df)}")
    logger.info(f"Test: {len(test_df)}")

    return train_df, val_df, test_df

