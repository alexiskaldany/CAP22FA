"""
test_custom_dataloader.py
Testing: testing custom dataloaders
author: @alexiskaldany, @justjoshtings
created: 9/28/22
"""

# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
from transformers import BertTokenizer, VisualBertForMultipleChoice
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys

# # get current directory
# path = os.getcwd()
# parent_path = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
# # add src to executable path to allow imports from src
# sys.path.insert(0, parent_path)

# from src.utils.visual_embeddings import get_multiple_embeddings

class CustomDataLoader(Dataset):
    '''
	CustomDataLoader object
	'''
    def __init__(self, prompts, answer_choices, answers, image_paths, tokenizer, visual_embedder):
        '''
        Params:
            self: instance of object
            prompts (list of str): list of prompts in text
            answer_choices (list of list of str): 4 answer choices as lists per sample
            answers (list of str): the correct answer of the 4 answer choices for the prompt
            image_paths (list of str): image paths
            tokenizer (tokenizer object): tokenizer function
            visual_embedder (visual embedding object): visual embedding function
        '''
        self.prompts = prompts
        self.answer_choices = answer_choices
        self.answers = answers
        self.image_paths = image_paths
        self.tokenizer = tokenizer
        self.visual_embedder = visual_embedder

    def __len__(self):
        '''
        Params:
            self: instance of object
        Returns:
            number of samples
        '''
        return len(self.image_paths)
    
class CustomDataLoaderVisualBERT(CustomDataLoader):
    '''
	CustomDataLoader object for VisualBERT
	'''
    def __init__(self, prompts, answer_choices, answers, image_paths, tokenizer, visual_embedder, return_tensors_type="pt", max_length=512):
        '''
        Params:
            self: instance of object
            prompts (list of str): list of prompts in text
            answer_choices (list of list of str): 4 answer choices as lists per sample
            answers (list of str): the correct answer of the 4 answer choices for the prompt
            image_paths (list of str): image paths
            tokenizer (tokenizer object): tokenizer function
            visual_embedder (visual embedding object): visual embedding function
            return_tensors_type (str): ['pt', 'tf'], default='pt'
            max_length (int): max length for tokenizer input, berttokenizer = 512
        '''
        CustomDataLoader.__init__(self, prompts, answer_choices, answers, image_paths, tokenizer, visual_embedder)

        self.max_length = max_length
        self.return_tensors_type = return_tensors_type

    def __getitem__(self, idx):
        '''
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
        # Text Embeddings
        choice0 = self.answer_choices[idx][0]
        choice1 = self.answer_choices[idx][1]
        choice2 = self.answer_choices[idx][2]
        choice3 = self.answer_choices[idx][3]

        prompts = [[self.prompts[idx]]*len(self.answer_choices[idx])]
        prompts = sum(prompts, [])

        choices = [[choice0, choice1, choice2, choice3]]
        choices = sum(choices, [])

        text_encoding = self.tokenizer(prompts, choices, 
                                    return_tensors=self.return_tensors_type, 
                                    padding=True, max_length=self.max_length, 
                                    truncation=True, add_special_tokens=True
                                    )

        # Visual Embeddings
        # # (batch_size, num_choices, visual_seq_length, visual_embedding_dim)
        visual_embeds = self.visual_embedder([self.image_paths[idx],])
        visual_embeds = visual_embeds[str(next(iter(visual_embeds)))]
        visual_embeds = torch.Tensor(visual_embeds).unsqueeze(0)
        visual_embeds = visual_embeds.expand(1, len(self.answer_choices[idx]), *visual_embeds.shape)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        # Labels and Correct Answer
        answer_ind = self.answer_choices[idx].index(self.answers[idx])
        labels = torch.tensor(answer_ind).unsqueeze(0)

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
