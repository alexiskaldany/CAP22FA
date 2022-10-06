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
from transformers import BertTokenizer, VisualBertForQuestionAnswering, VisualBertForMultipleChoice
from transformers import VisualBertModel, VisualBertConfig
import torch, gc
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import time
import math
from test_custom_dataloader import CustomDataLoaderVisualBERT

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
print(inputs_dict['input_ids'][0])
# print(inputs_dict['input_ids'][0].shape)


inputs_dict.update(
{
        "visual_embeds": visual_embeds,
        "visual_attention_mask": visual_attention_mask,
        "visual_token_type_ids": visual_token_type_ids,
        "labels": labels,
    }
)

# Check sizes of all of these inputs against the demo one!! + is it actually working?
# try unsqueexing input ids to diff dim does it still run through?
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/multiple-choice/run_swag.py
# https://github.com/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb
print(inputs_dict['input_ids'].shape)
print(inputs_dict['visual_embeds'].shape)
print(inputs_dict['visual_attention_mask'].shape)
print(inputs_dict['visual_token_type_ids'].shape)
print(inputs_dict['labels'].shape)
print(inputs_dict.keys())

'''
Test Inference
'''
# Inference
outputs = model(**inputs_dict)
loss = outputs.loss
logits = outputs.logits
print(logits)
print(logits.argmax(-1))
print(answer_choices[logits.argmax(-1)])

'''
Test Training Loop
'''
# Training Iteration
# Optimizer and Learning Rate Scheduler
lr=5e-5
num_epochs = 1
sample_every = 1

optimizer = AdamW(model.parameters(), lr=lr)

num_epochs = num_epochs
# num_training_steps = num_epochs * len(train_data_loader)
num_training_steps = num_epochs * 1
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print('Using device..', device)
# if LOG_FILENAME:
#     MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training] Using device {device}...")	

model.to(device)

progress_bar = tqdm(range(num_training_steps))

total_t0 = time.time()
training_stats = []
sample_every = sample_every

gc.collect()
torch.cuda.empty_cache()

for epoch in range(num_epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
    print(f'Training {model._get_name()}...')
    # if LOG_FILENAME:
    #     MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training] \n======== Epoch {epoch + 1} / {num_epochs} ========")	
    #     MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training] Training... {gpt_model_type}")	
    
    total_train_loss = 0
    total_train_accuracy = 0

    t0 = time.time()
    model.train()

    for step, batch in enumerate([inputs_dict]):
        print(batch)
        # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'visual_embeds', 'visual_attention_mask', 'visual_token_type_ids', 'labels'])
        b_input_ids = batch['input_ids'].to(device)
        b_token_type_ids = batch['token_type_ids'].to(device)
        b_attention_mask = batch['attention_mask'].to(device)

        b_labels = batch['labels'].to(device)

        b_visual_embeds = batch['visual_embeds'].to(device)
        b_visual_attention_mask = batch['visual_attention_mask'].to(device)
        b_visual_token_type_ids = batch['visual_token_type_ids'].to(device)

        print(b_input_ids.shape)
        print(b_token_type_ids.shape)
        print(b_attention_mask.shape)
        print(b_labels.shape)
        print(b_visual_embeds.shape)
        print(b_visual_attention_mask.shape)
        print(b_visual_token_type_ids.shape)

        model.zero_grad()        

        outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, token_type_ids=b_token_type_ids,
                        visual_embeds=b_visual_embeds, visual_attention_mask=b_visual_attention_mask, visual_token_type_ids=b_visual_token_type_ids,
                        labels = b_labels
        )

        loss = outputs[0]

        # outputs = model(**inputs_dict)
        # loss = outputs.loss
        # logits = outputs.logits

        batch_loss = loss.item()
        # batch_perplexity = math.exp(batch_loss)
        
        total_train_loss += batch_loss

        print('LOSS: ', loss, batch_loss, total_train_loss)
        # total_train_perplexity += batch_perplexity
            
        # # Get sample every x batches.
        # if step % sample_every == 0 and not step == 0:

        #     elapsed = format_time(time.time() - t0)
        #     print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_data_loader), batch_loss, elapsed))
        #     if LOG_FILENAME:
        #         MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training]   Batch {step}  of  {len(train_data_loader)}. Loss: {batch_loss}.   Elapsed: {elapsed}.")	

        #     if eval_during_training:
        #         model.eval()

        #         sample_outputs = model.generate(
        #                                 bos_token_id=random.randint(1,30000),
        #                                 do_sample=True,   
        #                                 top_k=50, 
        #                                 max_length = 200,
        #                                 top_p=0.95, 
        #                                 num_return_sequences=1,
        #                                 no_repeat_ngram_size=2,
        #                                 early_stopping=True
        #                             )
        #         for i, sample_output in enumerate(sample_outputs):
        #             print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
        #             if LOG_FILENAME:
        #                 MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training] {i}: {tokenizer.decode(sample_output, skip_special_tokens=True)}")	
                
                # model.train()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)

    # Calculate the average loss over all of the batches.
    # avg_train_loss = total_train_loss / len(train_data_loader)       
    # avg_train_perplexity = total_train_perplexity / len(train_data_loader)       

    # # Measure how long this epoch took.
    # training_time = format_time(time.time() - t0)

    # print("")
    # print("  Average training loss: {0:.2f}".format(avg_train_loss))
    # print("  Average training perplexity: {0:.2f}".format(avg_train_perplexity))
    # print("  Training epoch took: {:}".format(training_time))
    # if LOG_FILENAME:
    #     MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training]\n  Average training loss: {avg_train_loss}")	
    #     MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training]\n  Average training perplexity: {avg_train_perplexity}")	
    #     MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training]  Training epoch took: {training_time}")