"""
test_custom_training.py
Testing: Testing training
author: @alexiskaldany, @justjoshtings
created: 9/28/22
"""

from loguru import logger
import os
import sys
import cv2
from transformers import BertTokenizer, VisualBertForQuestionAnswering, VisualBertForMultipleChoice
from transformers import VisualBertModel, VisualBertConfig
import torch, gc
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import time
import math
from test_custom_dataloader import CustomDataLoaderVisualBERT
from test_custom_trainer import Model_VisualBERT
from torch.utils.data import Dataset, DataLoader

# get current directory
path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
# add src to executable path to allow imports from src
sys.path.insert(0, parent_path)

from src.utils.configs import DATA_JSON, DATA_CSV, DATA_DIRECTORY, ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER, ANNOTATED_IMAGES_FOLDER, TEST_DIRECTORY, TEST_IMAGE_OUTPUT
from src.utils.prepare_and_download import get_data_objects, create_dataframe
from src.utils.applying_annotations import execute_full_set_annotation
from src.utils.visual_embeddings import get_multiple_embeddings
from src.utils.pre_process import create_train_val_test_split
from src.utils.configs import RANDOM_STATE

random_state = RANDOM_STATE

'''
Load data
'''
combined_list = get_data_objects(ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER)
data_df = create_dataframe(combined_list)
data_df['annotated_image_path'] = data_df['image_path'].str.replace('images','annotated_images')

logger.info(f"All data loaded, columns = {data_df.keys()} and {len(data_df)} samples")

'''
Train/Test Split
'''
train_df, val_df, test_df = create_train_val_test_split(data_df)

'''
Load tokenizer and visual embedders
'''
logger.info(f"Loading tokenizer and visual embedder")
visual_embedder = get_multiple_embeddings
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

'''
Prep for custom dataloader
'''
# Select only the first train_ind_to_run samples just for testing training loop purposes
train_ind_to_run = len(train_df)
val_ind_to_run = len(val_df)
test_ind_to_run = len(test_df)

logger.info(f"Preparing data for custom dataloader")

train_prompts = train_df["question"].to_list()[:train_ind_to_run]
train_answer_choices = train_df["list_of_answers"].to_list()[:train_ind_to_run]
train_answers = train_df["answer"].to_list()[:train_ind_to_run]
train_image_paths = train_df["image_path"].to_list()[:train_ind_to_run]

val_prompts = val_df["question"].to_list()[:val_ind_to_run]
val_answer_choices = val_df["list_of_answers"].to_list()[:val_ind_to_run]
val_answers = val_df["answer"].to_list()[:val_ind_to_run]
val_image_paths = val_df["image_path"].to_list()[:val_ind_to_run]

test_prompts = test_df["question"].to_list()[:test_ind_to_run]
test_answer_choices = test_df["list_of_answers"].to_list()[:test_ind_to_run]
test_answers = test_df["answer"].to_list()[:test_ind_to_run]
test_image_paths = test_df["image_path"].to_list()[:test_ind_to_run]

'''
VisualBERT Model Training
'''
# Set up dataloader

logger.info(f"Setting up custom dataloaders")
# Train
visualbert_train_data = CustomDataLoaderVisualBERT(train_prompts, train_answer_choices, train_answers, train_image_paths, tokenizer, visual_embedder)
visualbert_train_data_loader = DataLoader(visualbert_train_data, batch_size=1, shuffle=True)

# Validate
visualbert_valid_data = CustomDataLoaderVisualBERT(val_prompts, val_answer_choices, val_answers, val_image_paths, tokenizer, visual_embedder)
visualbert_valid_data_loader = DataLoader(visualbert_valid_data, batch_size=1, shuffle=True)

# Test
visualbert_test_data = CustomDataLoaderVisualBERT(test_prompts, test_answer_choices, test_answers, test_image_paths, tokenizer, visual_embedder)
visualbert_test_data_loader = DataLoader(visualbert_test_data, batch_size=1, shuffle=True)

# model_input_ids, model_token_type_ids, model_attention_mask, model_labels, model_visual_embeds, model_visual_attention_mask, model_visual_token_type_ids = training_dataloader[0]

logger.info(f'Num Train:  {len(visualbert_train_data_loader)}, \
	Num Validation:  {len(visualbert_train_data_loader)}, \
	Num Test:  {len(visualbert_train_data_loader)}, \
	Total Num:  {len(visualbert_train_data_loader)+len(visualbert_train_data_loader)+len(visualbert_train_data_loader)}')

model_visualbert = Model_VisualBERT(random_state=random_state, 
								train_data_loader=visualbert_train_data_loader,
								valid_data_loader=visualbert_valid_data_loader,
								test_data_loader=visualbert_test_data_loader,
								model_type='visualbert',
                                log_file=logger)

model_visualbert.train(num_epochs=3, model_weights_dir='./results/model_weights/visualbert_3epochs_test01/')
model_visualbert.get_training_stats(model_weights_dir='./results/model_weights/visualbert_3epochs_test01/training_stats.csv')

'''
Inference Test
'''
# logger.info(f"Performing inference test")
# # outputs = model(**inputs_dict)
# # loss = outputs.loss
# # logits = outputs.logits
# # print(logits)
# # print(logits.argmax(-1))
# # print(answer_choices[logits.argmax(-1)])