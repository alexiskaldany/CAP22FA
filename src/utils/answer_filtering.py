""" 
answer_filtering.py
Analysis of Answers
author: @alexiskaldany, @justjoshtings
created: 10/15/22
"""
import os
import sys
import re
import pandas as pd
# get current directory
path = os.getcwd()
# parent_path = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
# add src to executable path to allow imports from src
sys.path.insert(0, path)
import pandas as pd 
from src.utils.configs import DATA_JSON, DATA_CSV, DATA_DIRECTORY, ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER, ANNOTATED_IMAGES_FOLDER, TEST_DIRECTORY, TEST_IMAGE_OUTPUT
from src.utils.prepare_and_download import get_data_objects, create_dataframe
from src.utils.applying_annotations import execute_full_set_annotation
from src.utils.visual_embeddings import get_multiple_embeddings
from src.utils.pre_process import create_train_val_test_split
from src.utils.configs import RANDOM_STATE
combined_list = get_data_objects(ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER)
data_df = create_dataframe(combined_list)
data_df['annotated_image_path'] = data_df['image_path'].str.replace('images','annotated_images')
with open(path + '/src/models/results/model_weights/visualbert_config_testing/vocab.txt','r') as f:
    vocab_list = f.read().split("\n")

def has_only_one_word_answers(list_of_answers:list)->bool:
    if any([re.match(r'^[a-zA-Z0-9_]+$',x) for x in list_of_answers]):
        return False 
    else:
        return True

def has_all_answers_in_token_list(list_of_answers:list,vocab_list)->bool:
    answer_is_token =[]
    token_match = []
    list_of_answers = [x for x in list_of_answers if x != None or list]
    for answer in list_of_answers:
        is_token =[]
        for index,token in enumerate(vocab_list):
            if answer == token:
                is_token.append(True)
                token_match.append(([index,token],answer))
            else:
                is_token.append(False)
        if any(is_token):
            answer_is_token.append(True)
    if any(answer_is_token):
        print(token_match)
        return True
    else:
        return False
            
data_df['one_word_answers'] = data_df['list_of_answers'].apply(lambda x: has_only_one_word_answers(x))

print(data_df.shape[0])
data_df = data_df[data_df['one_word_answers'] == True]
print(data_df.shape[0])
# data_df['has_all_answers_in_token_list'] = data_df['list_of_answers'].apply(lambda x: has_all_answers_in_token_list(x,vocab_list))
# print(data_df['one_word_answers'].value_counts())
# print(data_df['has_all_answers_in_token_list'].value_counts())
                                                                        

data_df.to_csv(DATA_CSV)