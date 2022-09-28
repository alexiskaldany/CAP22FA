# from pathlib import Path
# from src.utils.pre_process import *
from transformers import BertTokenizer, VisualBertForMultipleChoice
from transformers import Trainer, TrainingArguments
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from src.utils.configs import *

# VisualBertConfig()
# JSON_PATH = DATA_DIRECTORY /"question_and_answers_dict_id_key.json"
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vqa")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir = SAVED_MODELS_FOLDER/"visualbert",          # output directory
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)
