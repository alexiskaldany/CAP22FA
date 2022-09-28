"""
test_model_load.py
Testing: testing if models can be loaded into EC2 and trainable with our inputs without running into GPU overflow
author: @alexiskaldany, @justjoshtings
created: 9/26/22
"""
from loguru import logger
import os
import sys
import cv2

# get current directory
path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
# add src to executable path to allow imports from src
sys.path.insert(0, parent_path)

from src.utils.configs import DATA_JSON, DATA_CSV, DATA_DIRECTORY, ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER, ANNOTATED_IMAGES_FOLDER, TEST_DIRECTORY, TEST_IMAGE_OUTPUT
from src.utils.prepare_and_download import get_data_objects, create_dataframe
from src.utils.applying_annotations import execute_full_set_annotation
from src.utils.visual_embeddings import get_multiple_embeddings

'''
Load data
'''
combined_list = get_data_objects(ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER)
data_df = create_dataframe(combined_list)

'''
1. Test VisualBERT
'''
# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
from transformers import BertTokenizer, VisualBertForQuestionAnswering
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

test_case_index = 88

text = data_df["question"][test_case_index] + ' ' +  data_df["list_of_answers"][test_case_index][0] + ' ' + data_df["list_of_answers"][test_case_index][1] + ' ' + data_df["list_of_answers"][test_case_index][2] + ' ' + data_df["list_of_answers"][test_case_index][3]
inputs = tokenizer(text, return_tensors="pt")
visual_embeds = get_multiple_embeddings([data_df["image_path"][test_case_index],])
# print(next(iter(visual_embeds)))
# list(test_dict.keys())[0]
visual_embeds = visual_embeds[str(next(iter(visual_embeds)))]
print(visual_embeds, len(visual_embeds), type(visual_embeds))
visual_embeds = torch.Tensor(visual_embeds).unsqueeze(0)
print(visual_embeds.shape)
# visual_embeds = get_visual_embeddings(image).unsqueeze(0)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)

labels = torch.tensor([[0.0, 1.0]]).unsqueeze(0)  # Batch size 1, Num labels 2
print(labels.shape)

# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# scores = outputs.logits

inputs = bert_tokenizer(
        test_question,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

output_vqa = visualbert_vqa(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    visual_embeds=features,
    visual_attention_mask=torch.ones(features.shape[:-1]),
    token_type_ids=inputs.token_type_ids,
    output_attentions=False,
)
# get prediction
pred_vqa = output_vqa["logits"].argmax(-1)
print("Question:", test_question)
print("prediction from VisualBert VQA:", vqa_answers[pred_vqa])