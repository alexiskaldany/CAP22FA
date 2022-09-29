from transformers import ViltProcessor, ViltForQuestionAnswering,Trainer
from pathlib import Path
from src.utils.pre_process import *
from src.utils.prepare_and_download import get_data_objects, create_dataframe
from src.utils.applying_annotations import execute_full_set_annotation
from src.utils.visual_embeddings import get_multiple_embeddings
from transformers import BertTokenizer, VisualBertForMultipleChoice
from transformers import Trainer, TrainingArguments
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from src.utils.configs import *
import pandas as pd

# VisualBertConfig()
# JSON_PATH = DATA_DIRECTORY /"question_and_answers_dict_id_key.json"
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

## Loading the data
data_list = get_data_objects(ANNOTATION_FOLDER,IMAGES_FOLDER,QUESTIONS_FOLDER)
dataframe = create_dataframe(data_list)
dataframe = dataframe.stack().reset_index()
print(dataframe.head())
## Applying the annotations

execute_full_set_annotation(DATA_JSON, ANNOTATED_IMAGES_FOLDER)

id_list = list(dataframe["image_id"])
annotated_image_path = [
    str(ANNOTATED_IMAGES_FOLDER / f"{id}.png") for id in id_list
]
dataframe["annotated_image_path"] = annotated_image_path
## Getting Visual Embeddings
# logger.info("Getting annotated images embeddings")
# annotated_list = []
# raw_list = []
# annotated_images_embeddings = get_multiple_embeddings(list(dataframe["annotated_image_path"]))
# raw_images_embeddings = get_multiple_embeddings(list(dataframe["image_path"]))
## Adding the visual embeddings to the dataframe
# for index, row in dataframe.iterrows():
#     if index % 100:
#         logger.info(f"At index: {index}")
#     img_id = str(row["image_id"])
#     if img_id in annotated_images_embeddings:
#         annotated_list.append(annotated_images_embeddings[img_id])
#     if img_id not in annotated_images_embeddings:
#         annotated_list.append([])
#     if img_id in raw_images_embeddings:
#         raw_list.append(raw_images_embeddings[img_id])
#     if img_id not in raw_images_embeddings:
#         raw_list.append([])
# if dataframe.shape[0] == len(annotated_list):
#     dataframe["annotated_images_embeddings"] = annotated_list
# if dataframe.shape[0] == len(raw_list):
#     dataframe["raw_image_embeddings"] = raw_list
## Creating the text embeddings
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# text_embeddings = []
# for index, row in dataframe.iterrows():
#     if index % 100:
#         logger.info(f"At index: {index}")
#     tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors="pt", padding=True)
## Saving the dataframe
dataframe.to_csv(DATA_CSV, index=False)

## Loading the dataframe
dataframe = pd.read_csv(DATA_CSV)

## Loading the model
model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vqa")


training_args = TrainingArguments(
    output_dir = SAVED_MODELS_FOLDER/"visualbert",          # output directory
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)
