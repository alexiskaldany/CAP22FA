# from pathlib import Path
# from src.utils.pre_process import *
from transformers import BertTokenizer,DeiTFeatureExtractor,VisualBertForMultipleChoice
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from pathlib import Path

image = "/Users/alexiskaldany/school/CAP22FA/example_data/0.png"
question = "What is A in the diagram?"
actual_answer = "face"
# JSON_PATH = DATA_DIRECTORY /"question_and_answers_dict_id_key.json"
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vqa")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")