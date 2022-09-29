"""
test_visualbert.py
Testing: testing if models can be loaded into EC2 and trainable with our inputs without running into GPU overflow
author: @alexiskaldany, @justjoshtings
created: 9/28/22
"""

# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
from transformers import BertTokenizer, VisualBertForMultipleChoice
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataLoader(Dataset):
    '''
	CustomDataLoader object
	'''
    def __init__(self, sentences_list, tokenizer, image_embeddings_list):
        '''
        Params:
            self: instance of object
            sentences_list (list of str): list of sentences
            tokenizer (tokenizer object): tokenizer function
        '''
        self.sentences_list = sentences_list
        self.tokenizer = tokenizer
        self.image_embeddings_list = image_embeddings_list

    def __len__(self):
        '''
        Params:
            self: instance of object
        Returns:
            number of corpus texts
        '''
        return len(self.sentences_list)
    
class CustomDataLoaderVisualBERT(CustomDataLoader):
    '''
	CustomDataLoader object for VisualBERT
	'''
    def __init__(self, sentences_list, tokenizer, image_embeddings_list, return_tensors_type="pt", max_length=768):
        '''
        Params:
            self: instance of object
            sentences_list (list of str): list of sentences
            tokenizer (tokenizer object): tokenizer function
            image_embeddings_list (list of list of embeddings): list of image embeddings
            return_tensors_type (str): ['pt', 'tf'], default='pt'
            max_length (int): max length for tokenizer input, gpt2:default=768, gpt_neo: 2048
        '''
        CustomDataLoader.__init__(self, sentences_list, tokenizer, image_embeddings_list)

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
        '''
        # try:
        # 	with open(self.sentences_list[idx], 'r+') as f:
        # 		text = f.read()
        # except FileNotFoundError:
        # 	with open(self.sentences_list[0], 'r+') as f:
        # 		text = f.read()

        text = self.sentences_list[idx]
        encodings_dict = self.tokenizer('<|startoftext|>'+ text + '<|endoftext|>', truncation=True, max_length=self.max_length, padding="max_length")
        image_embeddings = self.image_embeddings_list[idx]
        input_ids = torch.tensor(encodings_dict['input_ids'])
        attn_masks = torch.tensor(encodings_dict['attention_mask'])

# Add to tokenizer
    #     inputs = bert_tokenizer(
    #     test_question,
    #     padding="max_length",
    #     max_length=20,
    #     truncation=True,
    #     return_token_type_ids=True,
    #     return_attention_mask=True,
    #     add_special_tokens=True,
    #     return_tensors="pt",
    # )
# Outputs Needed
        # visual_attention_mask=torch.ones(features.shape[:-1]),
        # token_type_ids=inputs.token_type_ids,
        # output_attentions=False,

        return input_ids, attn_masks

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vcr")

# prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
# choice0 = "It is eaten with a fork and a knife."
# choice1 = "It is eaten while held in the hand."

# visual_embeds = get_visual_embeddings(image)
# # (batch_size, num_choices, visual_seq_length, visual_embedding_dim)
# visual_embeds = visual_embeds.expand(1, 2, *visual_embeds.shape)
# visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
# visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

# labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

# encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors="pt", padding=True)
# # batch size is 1
# inputs_dict = {k: v.unsqueeze(0) for k, v in encoding.items()}
# inputs_dict.update(
#     {
#         "visual_embeds": visual_embeds,
#         "visual_attention_mask": visual_attention_mask,
#         "visual_token_type_ids": visual_token_type_ids,
#         "labels": labels,
#     }
# )
# outputs = model(**inputs_dict)

# loss = outputs.loss
# logits = outputs.logits