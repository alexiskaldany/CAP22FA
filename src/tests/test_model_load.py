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
from transformers import BertTokenizer, VisualBertForQuestionAnswering, VisualBertForMultipleChoice
from transformers import VisualBertModel, VisualBertConfig
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vcr")
# model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

# configuration = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa")
configuration = VisualBertConfig.from_pretrained("uclanlp/visualbert-vcr")
configuration.__dict__['visual_embedding_dim'] = 512
print(configuration)
model = VisualBertForMultipleChoice(configuration)

configuration = model.config
print(configuration)

test_case_index = 88

text = data_df["question"][test_case_index] + ' ' +  data_df["list_of_answers"][test_case_index][0] + ' ' + data_df["list_of_answers"][test_case_index][1] + ' ' + data_df["list_of_answers"][test_case_index][2] + ' ' + data_df["list_of_answers"][test_case_index][3]
inputs = tokenizer(text, return_tensors="pt")
visual_embeds = get_multiple_embeddings([data_df["image_path"][test_case_index],])
# print(next(iter(visual_embeds)))
# list(test_dict.keys())[0]
visual_embeds = visual_embeds[str(next(iter(visual_embeds)))]
# print(visual_embeds, len(visual_embeds), type(visual_embeds))
visual_embeds = torch.Tensor(visual_embeds).unsqueeze(0).unsqueeze(0)
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

print(inputs.input_ids.shape)
print(inputs.token_type_ids.shape)
print(inputs.attention_mask.shape)
print(inputs.visual_embeds.shape)
print(inputs.visual_token_type_ids.shape)
print(inputs.visual_attention_mask.shape)

# labels = torch.tensor([[0.0, 1.0, 2.0, 3.0]]).unsqueeze(0)  # Batch size 1, Num labels 2
# print(labels.shape)

# outputs = model(**inputs)
# loss = outputs.loss
# scores = outputs.logits
# print(scores)


print(text)
prompt = data_df["question"][test_case_index]
answer_choices = data_df["list_of_answers"][test_case_index]
choice0 = answer_choices[0]
choice1 = answer_choices[1]
choice2 = answer_choices[2]
choice3 = answer_choices[3]

answer_ind = answer_choices.index(data_df['answer'][test_case_index])
print(answer_ind)


# # (batch_size, num_choices, visual_seq_length, visual_embedding_dim)
visual_embeds = get_multiple_embeddings([data_df["image_path"][test_case_index],])
visual_embeds = visual_embeds[str(next(iter(visual_embeds)))]
visual_embeds = torch.Tensor(visual_embeds).unsqueeze(0)
print(visual_embeds.shape)
visual_embeds = visual_embeds.expand(1, len(answer_choices), *visual_embeds.shape)
print(visual_embeds.shape)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

labels = torch.tensor(answer_ind).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1
print(labels)

prompts = [[prompt]*len(answer_choices)]
prompts = sum(prompts, [])
print(prompts)

choices = [[choice0, choice1, choice2, choice3]]
choices = sum(choices, [])
print(choices)

encoding = tokenizer(prompts, choices, return_tensors="pt", padding=True)
print(encoding['input_ids'])
print(encoding['input_ids'].shape)

# # batch size is 1
inputs_dict = {k: v.unsqueeze(0) for k, v in encoding.items()}
# inputs_dict = {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in encoding.items()}

print(inputs_dict['input_ids'][0][0])
print(inputs_dict['input_ids'][0][1])
print(inputs_dict['input_ids'][0][2])
print(inputs_dict['input_ids'][0][3])
print(tokenizer.decode(inputs_dict['input_ids'][0][0]))
print(tokenizer.decode(inputs_dict['input_ids'][0][1]))
print(tokenizer.decode(inputs_dict['input_ids'][0][2]))
print(tokenizer.decode(inputs_dict['input_ids'][0][3]))
print(inputs_dict['input_ids'])
print(inputs_dict['input_ids'][0].shape)

# Check sizes of all of these inputs!!

inputs_dict.update(
{
        "visual_embeds": visual_embeds,
        "visual_attention_mask": visual_attention_mask,
        "visual_token_type_ids": visual_token_type_ids,
        "labels": labels,
    }
)

print(inputs_dict)
outputs = model(**inputs_dict)
loss = outputs.loss
logits = outputs.logits
print(logits.argmax(-1))
print(answer_choices[logits.argmax(-1)])