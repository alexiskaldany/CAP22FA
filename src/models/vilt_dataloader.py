""" 
vilt_dataloader.py
Custom data loader for Visual Question Answering
author: @alexiskaldany, @justjoshtings
created: 10/13/22
"""

## This model uses the ViLTProcessor to embed both the text and the image.
## As the 
## https://huggingface.co/docs/transformers/v4.23.1/en/model_doc/vilt#transformers.ViltForQuestionAnswering

from torch.utils.data import Dataset
import torch
from PIL import Image
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import spacy
import numpy as np
nlp = spacy.load("en_core_web_sm")

answers = ["cars", "trucks", "buses", "motorcycles"]
answer_tokens = [nlp(x) for x in answers]
print(answer_tokens)
array = torch.tensor([answers]).unsqueeze(0)
print(array)
# array = np.array([x for x in answer_tokens])
# print(array)
# labels = OneHotEncoder().fit_transform().toarray()
# print(labels)

class CustomDataLoader(Dataset):
    '''
	CustomDataLoader object
	'''
    def __init__(self, prompts, answer_choices, answers, image_paths, processor):
        '''
        Params:
            self: instance of object
            prompts (list of str): list of prompts in text
            answer_choices (list of list of str): 4 answer choices as lists per sample
            answers (list of str): the correct answer of the 4 answer choices for the prompt
            image_paths (list of str): image paths
            processor (processor object): processor function
        '''
        self.prompts = prompts
        self.answer_choices = answer_choices
        self.answers = answers
        self.image_paths = image_paths
        # self.tokenizer = tokenizer
        # self.visual_embedder = visual_embedder
        self.processor = processor

    def __len__(self):
        '''
        Params:
            self: instance of object
        Returns:
            number of samples
        '''
        return len(self.image_paths)
    
class CustomDataLoaderViLT(CustomDataLoader):
    '''
	CustomDataLoader object for ViLT
	'''
    def __init__(self, prompts, answer_choices, answers, image_paths,processor, return_tensors_type="pt", max_length=512):
        '''
        Params:
            self: instance of object
            prompts (list of str): list of prompts in text
            answer_choices (list of list of str): 4 answer choices as lists per sample
            answers (list of str): the correct answer of the 4 answer choices for the prompt
            image_paths (list of str): image paths
            tokenizer (tokenizer object): tokenizer function
            # visual_embedder (visual embedding object): visual embedding function
            # return_tensors_type (str): ['pt', 'tf'], default='pt'
            processor (processor object): processor function
            max_length (int): max length for tokenizer input, berttokenizer = 512
        '''
        CustomDataLoader.__init__(self, prompts, answer_choices, answers, image_paths, processor)

        self.max_length = max_length
        self.return_tensors_type = return_tensors_type
    
    def __getitem__(self, idx):
        '''
        Builds embeddings from data

        Params:
            self: instance of object
            idx (int): index of iteration
        Returns:
            input_ids (pt tensors): encoded text as tensors
            attn_masks (pt tensors): attention masks as tensors
            model_input_ids
            model_token_type_ids
            model_attention_mask
            model_labels
            model_visual_embeds
            model_visual_attention_mask
            model_visual_token_type_ids
        '''
        # Text 
        choice0 = self.answer_choices[idx][0]
        choice1 = self.answer_choices[idx][1]
        choice2 = self.answer_choices[idx][2]
        choice3 = self.answer_choices[idx][3]

        # prompts = [[self.prompts[idx]]*len(self.answer_choices[idx])]
        # prompts = sum(prompts, [])

        # choices = [[choice0, choice1, choice2, choice3]]
        # choices = sum(choices, [])

        text = [self.prompts[idx]]

        # Image
        
        image = Image.open(self.image_paths[idx])
        # text_encoding = self.tokenizer(prompts, choices, 
        #                             return_tensors=self.return_tensors_type, 
        #                             padding=True, max_length=self.max_length, 
        #                             truncation=True, add_special_tokens=True
        #                             )

        # Visual Embeddings
        # # (batch_size, num_choices, visual_seq_length, visual_embedding_dim)
        # visual_embeds = self.visual_embedder([self.image_paths[idx],])
        # visual_embeds = visual_embeds[str(next(iter(visual_embeds)))]
        # visual_embeds = torch.Tensor(visual_embeds).unsqueeze(0)
        # visual_embeds = visual_embeds.expand(1, len(self.answer_choices[idx]), *visual_embeds.shape)
        # visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        # visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        # Labels and Correct Answer
        answer = self.answer_choices[idx].index(self.answers[idx])
        tokenizer = get_tokenizer("basic_english")
        answer_token = tokenizer(answer)
        voc = build_vocab_from_iterator(answer_token)
        labels = F.one_hot(torch.tensor(voc(answer)), num_classes=1)
        encoding = self.processor(image, text, return_tensors=self.return_tensors_type, padding=True, truncation=True, max_length=self.max_length)
        # Inputs Dict
        inputs_dict = {k: v.unsqueeze(0) for k, v in text_encoding.items()}
        inputs_dict.update(
                    {
                            "visual_embeds": visual_embeds,
                            "visual_attention_mask": visual_attention_mask,
                            "visual_token_type_ids": visual_token_type_ids,
                            "labels": labels,
                        }
                    )

        return inputs_dict
    
    
# from transformers import ViltProcessor, ViltForQuestionAnswering
# import requests
# from PIL import Image

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open("/Users/alexiskaldany/school/CAP22FA/example_data/0.png")
# text = "What is A in the diagram?"

# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# # prepare inputs
# encoding = processor(image, text, return_tensors="pt")

# # forward pass
# outputs = model(**encoding)
# logits = outputs.logits
# idx = logits.argmax(-1).item()

# print(idx)
# print("Predicted answer:", model.config.id2label[idx])
# print(logits.argmax(-1))