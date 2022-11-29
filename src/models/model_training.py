"""
model_training.py
Perform model training
author: @alexiskaldany, @justjoshtings
created: 9/28/22
"""

from loguru import logger
import os
import sys
from transformers import (
    BertTokenizer,
    VisualBertForQuestionAnswering,
    VisualBertForMultipleChoice,
)
from transformers import VisualBertModel, VisualBertConfig
import torch, gc
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import time
from datetime import datetime
import math
from custom_dataloader import CustomDataLoaderVisualBERT
from custom_trainer import Model_VisualBERT
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# get current directory
# add src to executable path to allow imports from src
current_operating_system = sys.platform
if current_operating_system == "linux" or current_operating_system == "linux2":
    path = os.getcwd()
    parent_path = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
    sys.path.insert(0, parent_path)
if current_operating_system == "darwin":
    path = os.getcwd()
    sys.path.insert(0, path)

from src.utils.configs import (
    ANNOTATION_FOLDER,
    IMAGES_FOLDER,
    QUESTIONS_FOLDER,
    MODEL_FOLDER,
    RESULTS_FOLDER,
)
from src.utils.prepare_and_download import get_data_objects, create_dataframe
from src.utils.applying_annotations import execute_full_set_annotation

# from src.utils.visual_embeddings import get_multiple_embeddings
from src.utils.detect import get_visual_embeddings
from src.utils.pre_process import create_train_val_test_split
from src.utils.configs import RANDOM_STATE
from src.utils.annotation_to_string import get_relationship_strings

# from src.utils.answer_filtering import has_only_one_word_answers

random_state = RANDOM_STATE

"""
Set logger
"""
logger.remove()
logger.add(
    "./logs/training_log.txt",
    # sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss}|{level}| {message}|{function}: {line}",
    level="INFO",
    backtrace=True,
    colorize=True,
)

logger.info(f"\n\n\n[Training Script - {datetime.now()}] Running training script....")

"""
Load data
"""
combined_list = get_data_objects(ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER)
data_df = create_dataframe(combined_list)
# data_df['one_word_answers'] = data_df['list_of_answers'].apply(lambda x: has_only_one_word_answers(x))
# data_df = data_df[data_df['one_word_answers'] == True]

"""
Different preprocessing setups to try with model
"""
setup = "setup1"

if setup == "setup1":
    ## Preprocessing Setup 1: Regular Diagrams, no Annotations
    data_df["annotated_image_path"] = data_df["image_path"]
elif setup == "setup2":
    ## Preprocessing Setup 2: Annotations on Diagrams
    data_df["annotated_image_path"] = data_df["image_path"].str.replace(
        "images", "annotated_images"
    )
else:
    # Preprocessing Setup 3: Testing questions combined with annotations
    data_df = get_relationship_strings(data_df)
    data_df["annotated_image_path"] = data_df["image_path"]
    data_df["question"] = data_df["question"] + " " + data_df["relationship_string"]
    question = data_df["question"].to_list()

    # Question must be less than 450 characters
    question = [
        question[i] if len(question[i]) < 450 else question[i][:450]
        for i in range(len(question))
    ]
    data_df["question"] = question
    logger.info(
        f"All data loaded, columns = {data_df.keys()} and {len(data_df)} samples"
    )

"""
Train/Test Split
"""
train_df, val_df, test_df = create_train_val_test_split(
    data_df, MODEL_FOLDER / "test_ids.txt"
)

"""
Load tokenizer and visual embedders
"""
logger.info(f"Loading tokenizer and visual embedder")
visual_embedder = get_visual_embeddings
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    bos_token="<|startoftext|>",
    eos_token="<|endoftext|>",
    pad_token="<|pad|>",
)
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

"""
Prep for custom dataloader
"""
# Select only the first train_ind_to_run samples just for testing training loop purposes
train_ind_to_run = len(train_df)
val_ind_to_run = len(val_df)
test_ind_to_run = len(test_df)
# train_ind_to_run = 50
# val_ind_to_run = 50
# test_ind_to_run = 50

logger.info(f"Preparing data for custom dataloader")

train_prompts = train_df["question"].to_list()[:train_ind_to_run]
train_answer_choices = train_df["list_of_answers"].to_list()[:train_ind_to_run]
train_answers = train_df["answer"].to_list()[:train_ind_to_run]
train_image_paths = train_df["annotated_image_path"].to_list()[:train_ind_to_run]

val_prompts = val_df["question"].to_list()[:val_ind_to_run]
val_answer_choices = val_df["list_of_answers"].to_list()[:val_ind_to_run]
val_answers = val_df["answer"].to_list()[:val_ind_to_run]
val_image_paths = val_df["annotated_image_path"].to_list()[:val_ind_to_run]

test_prompts = test_df["question"].to_list()[:test_ind_to_run]
test_answer_choices = test_df["list_of_answers"].to_list()[:test_ind_to_run]
test_answers = test_df["answer"].to_list()[:test_ind_to_run]
test_image_paths = test_df["annotated_image_path"].to_list()[:test_ind_to_run]

"""
Checking for Class Imbalance
"""
class_imba_train = []

for i in range(len(train_answers)):
    answer = train_answers[i]
    class_imba_train.append(train_answer_choices[i].index(answer))

print(
    "\n\nTrain Class Imbalance Check: ",
    class_imba_train.count(0),
    class_imba_train.count(1),
    class_imba_train.count(2),
    class_imba_train.count(3),
)

class_imba_val = []

for i in range(len(val_answers)):
    answer = val_answers[i]
    class_imba_val.append(val_answer_choices[i].index(answer))

print(
    "val Class Imbalance Check: ",
    class_imba_val.count(0),
    class_imba_val.count(1),
    class_imba_val.count(2),
    class_imba_val.count(3),
)

class_imba_test = []

for i in range(len(test_answers)):
    answer = test_answers[i]
    class_imba_test.append(test_answer_choices[i].index(answer))

print(
    "test Class Imbalance Check: ",
    class_imba_test.count(0),
    class_imba_test.count(1),
    class_imba_test.count(2),
    class_imba_test.count(3),
)
"""
VisualBERT Model Training
"""
# Set up dataloader

logger.info(f"Setting up custom dataloaders")

# def collate_fn(data):
# print(len(data), data[0]['input_ids'].shape, data[1]['input_ids'].shape)
# print(len(data), data[0]['token_type_ids'].shape, data[1]['token_type_ids'].shape)
# print(len(data), data[0]['attention_mask'].shape, data[1]['attention_mask'].shape)
# print(len(data), data[0]['labels'].shape, data[1]['labels'].shape)
# print(len(data), data[0]['visual_embeds'].shape, data[1]['visual_embeds'].shape)
# print(len(data), data[0]['visual_attention_mask'].shape, data[1]['visual_attention_mask'].shape)
# print(len(data), data[0]['visual_token_type_ids'].shape, data[1]['visual_token_type_ids'].shape)
# print( data[0]['input_ids'])

# max_pad = 0
# for i in range(len(data)):
#     if data[0]['input_ids'].shape[2] > max_pad:
#         max_pad = data[0]['input_ids'].shape[2]

# result = F.pad(input=data[0]['input_ids'], pad=(0, max_pad), mode='constant', value=0)

# print('result',data[0]['input_ids'].shape,result.shape)
# print(data[0]['input_ids'])
# print(result)
# print(result[0][0])
# zipped = zip(data)
# return list(zipped)

# Train
visualbert_train_data = CustomDataLoaderVisualBERT(
    train_prompts,
    train_answer_choices,
    train_answers,
    train_image_paths,
    tokenizer,
    visual_embedder,
)
visualbert_train_data_loader = DataLoader(
    visualbert_train_data, batch_size=1, shuffle=True
)

# Validate
visualbert_valid_data = CustomDataLoaderVisualBERT(
    val_prompts,
    val_answer_choices,
    val_answers,
    val_image_paths,
    tokenizer,
    visual_embedder,
)
visualbert_valid_data_loader = DataLoader(
    visualbert_valid_data, batch_size=1, shuffle=True
)

# Test
visualbert_test_data = CustomDataLoaderVisualBERT(
    test_prompts,
    test_answer_choices,
    test_answers,
    test_image_paths,
    tokenizer,
    visual_embedder,
)
visualbert_test_data_loader = DataLoader(
    visualbert_test_data, batch_size=1, shuffle=True
)

# model_input_ids, model_token_type_ids, model_attention_mask, model_labels, model_visual_embeds, model_visual_attention_mask, model_visual_token_type_ids = training_dataloader[0]

logger.info(
    f"Num Train:  {len(visualbert_train_data_loader)}, \
	Num Validation:  {len(visualbert_train_data_loader)}, \
	Num Test:  {len(visualbert_train_data_loader)}, \
	Total Num:  {len(visualbert_train_data_loader)+len(visualbert_train_data_loader)+len(visualbert_train_data_loader)}"
)

model_visualbert = Model_VisualBERT(
    random_state=random_state,
    train_data_loader=visualbert_train_data_loader,
    valid_data_loader=visualbert_valid_data_loader,
    test_data_loader=visualbert_test_data_loader,
    model_type="visualbert",
    log_file=logger,
)

training_experiment_name = f"visualbert_{setup}_12epochs/"

model_visualbert.set_train_parameters(num_epochs=4, lr=5e-5, previous_num_epoch=0)

# model_visualbert.train(model_weights_dir=f'{os.getcwd()}/results/model_weights/visualbert_{training_experiment_name}/')
# model_visualbert.get_training_stats(model_weights_dir=f'{os.getcwd()}/results/model_weights/visualbert_{training_experiment_name}/training_stats.csv')

"""
Load from checkpoint to continue training
"""

model_checkpint_directory = Path(f"./results/model_weights/{training_experiment_name}/").absolute()
model_checkpint_directory_checkpoint =  model_checkpint_directory / "checkpoint" 
if not model_checkpint_directory_checkpoint.exists():
    logger.info(f"Checkpoint directory does not exist. Exiting")
    exit()

(
    model_from_checkpoint,
    optimizer_from_checkpoint,
    previous_num_epoch,
    criterion_from_checkpoint,
    tokenizer_from_checkpoint,
) = model_visualbert.load_from_checkpoint(
    model_checkpoint_dir=model_checkpint_directory,
)


model_visualbert_checkpoint = Model_VisualBERT(
    random_state=random_state,
    train_data_loader=visualbert_train_data_loader,
    valid_data_loader=visualbert_valid_data_loader,
    test_data_loader=visualbert_test_data_loader,
    model_type="visualbert",
    log_file=logger,
    criterion=criterion_from_checkpoint,
    model=model_from_checkpoint,
    tokenizer=tokenizer_from_checkpoint,
)

model_visualbert_checkpoint.set_train_parameters(
    num_epochs=4,
    lr=5e-5,
    optimizer=optimizer_from_checkpoint,
    previous_num_epoch=previous_num_epoch,
)
model_visualbert_checkpoint.train(
    model_weights_dir=model_checkpint_directory
)
model_visualbert_checkpoint.get_training_stats(
    model_weights_dir=Path(model_checkpint_directory) /"training_stats.csv"
)


"""
Inference Test
"""

logger.info(f"Performing inference test")
model_visualbert_checkpoint.test(
    model_weights_dir= Path(model_checkpint_directory) / "testing_stats.csv"
)
