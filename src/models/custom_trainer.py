"""
custom_trainer.py
Custom models for training on Visual Question Answering
author: @alexiskaldany, @justjoshtings
created: 10/10/22
"""
from transformers import BertTokenizer, VisualBertForQuestionAnswering, VisualBertForMultipleChoice
from transformers import VisualBertModel, VisualBertConfig
import torch, gc
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import time
import math
import sys
import os
from datetime import datetime
import datetime as dt
import random
import pandas as pd
from loguru import logger

class VQAModel:
    '''
	Object to handle Visual Question Answering models.
	'''
    def __init__(self, random_state, train_data_loader, valid_data_loader, test_data_loader, log_file=None):
        '''
        Params:
            self: instance of object
            random_state (int): random seed
            train_data_loader (torch.utils.data.DataLoader): train data loader
            valid_data_loader (torch.utils.data.DataLoader): validation data loader
            test_data_loader (torch.utils.data.DataLoader): test data loader
            log_file (str): default is None to not have logging, otherwise, specify logging path ../filepath/log.log
        '''
        self.random_state = random_state
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        self.LOGGING = log_file
        if self.LOGGING:
            logger.info(f"Setting up model VisualBERT")
	
    def train(self, num_epochs=1, lr=5e-5, sample_every=100, eval_during_training=False, save_weights=True, model_weights_dir='./results/model_weights/') :
        '''
        Method to perform training loop
        Params:
            self: instance of object
            num_epochs (int): number of epochs to train, default=1
            lr (float): learning rate, default=5e-5
            sample_every (int): number of training steps to print progress, default=100
            eval_during_training (Boolean): whether to evaluate during training every sample_every, default=False
            save_weights (Boolean): whether to save weights or not after training
            model_weights_dir (str): directory to save model weights after model training completion
        '''
        # Optimizer and Learning Rate Scheduler
        lr=5e-5

        optimizer = AdamW(self.model.parameters(), lr=lr)

        num_epochs = num_epochs
        num_training_steps = num_epochs * len(self.train_data_loader)
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        self.model.resize_token_embeddings(len(self.tokenizer))
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        print('Using device..', device)
        if self.LOGGING:
            logger.info(f"{datetime.now()} -- [Model Training] Using device {device}...")	

        self.model.to(device)

        progress_bar = tqdm(range(num_training_steps))

        total_t0 = time.time()
        self.training_stats = []
        sample_every = sample_every

        gc.collect()
        torch.cuda.empty_cache()

        for epoch in range(num_epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
            print(f'Training {self.model._get_name()}...')
            if self.LOGGING:
                logger.info(f"{datetime.now()} -- [Model Training] \n======== Epoch {epoch + 1} / {num_epochs} ========")	
                logger.info(f"{datetime.now()} -- [Model Training] Training... {self.model_type}")	
            
            total_train_loss = 0
            total_train_accuracy = 0
            total_train_f1 = 0
            total_train_precision = 0
            total_train_recall = 0

            # Count each result per iteration, then pass to confusion matrix fn to calculate macro metrics per epoch to print to logger
            # 
            total_predA_labelA = 0
            total_predA_labelB = 0
            total_predA_labelC = 0
            total_predA_labelD = 0
            total_predB_labelA = 0
            total_predB_labelB = 0
            total_predB_labelC = 0
            total_predB_labelD = 0
            total_predC_labelA = 0
            total_predC_labelB = 0
            total_predC_labelC = 0
            total_predC_labelD = 0
            total_predD_labelA = 0
            total_predD_labelB = 0
            total_predD_labelC = 0
            total_predD_labelD = 0

            t0 = time.time()
            self.model.train()

            for step, batch in enumerate(self.train_data_loader):
                # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'visual_embeds', 'visual_attention_mask', 'visual_token_type_ids', 'labels'])
                b_input_ids = batch['input_ids'][0].to(device)
                b_token_type_ids = batch['token_type_ids'][0].to(device)
                b_attention_mask = batch['attention_mask'][0].to(device)

                b_labels = batch['labels'][0].to(device)

                b_visual_embeds = batch['visual_embeds'][0].to(device)
                b_visual_attention_mask = batch['visual_attention_mask'][0].to(device)
                b_visual_token_type_ids = batch['visual_token_type_ids'][0].to(device)

                # logger.info(f"Shapes of - b_input_ids: {b_input_ids.shape}, \
                #                 b_token_type_ids: {b_token_type_ids.shape}, \
                #                 b_attention_mask: {b_attention_mask.shape}, \
                #                 b_labels: {b_labels.shape}, \
                #                 b_visual_embeds: {b_visual_embeds.shape}, \
                #                 b_visual_attention_mask: {b_visual_attention_mask.shape}, \
                #                 b_visual_token_type_ids: {b_visual_token_type_ids.shape}, \
                #                 ")

                self.model.zero_grad()        

                outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask, token_type_ids=b_token_type_ids, visual_embeds=b_visual_embeds, visual_attention_mask=b_visual_attention_mask, visual_token_type_ids=b_visual_token_type_ids, labels=b_labels)

                loss = outputs[0]
                batch_loss = loss.item()
                
                logits = outputs[1]
                y_pred = logits.argmax(-1)

                labels_ind = b_labels.argmax(-1)

                # print('\n\nTRYING SOFTMAX!!!',torch.sigmoid(logits))
                # print('TRYING SOFTMAX PRED!!!',torch.sigmoid(logits).argmax(-1))
                # print('SUM SOFTMAX PROBS',torch.sigmoid(logits).sum())
                # print('SUM LOGITS',torch.sigmoid(logits).sum())
                # print('ARGMAX PRED!!!',y_pred)
                # print('LABEL!!!',b_labels, b_labels.shape)
                # print('LOGITS!!!',logits, logits.shape)
                # print(outputs)

                train_acc = torch.sum(y_pred == labels_ind)

                total_train_loss += batch_loss
                logger.info(f'LOSS:  {loss}, {batch_loss}, {total_train_loss}')
                logger.info(f'Predicted: {y_pred}, Target: {b_labels}, Accuracy: {train_acc}')

                total_train_accuracy += train_acc

                # def F1_score(prob, label):
                #     prob = prob.bool()
                #     label = label.bool()
                #     epsilon = 1e-7
                #     TP = (prob & label).sum().float()
                #     TN = ((~prob) & (~label)).sum().float()
                #     FP = (prob & (~label)).sum().float()
                #     FN = ((~prob) & label).sum().float()
                #     #accuracy = (TP+TN)/(TP+TN+FP+FN)
                #     precision = torch.mean(TP / (TP + FP + epsilon))
                #     recall = torch.mean(TP / (TP + FN + epsilon))
                #     F2 = 2 * precision * recall / (precision + recall + epsilon)
                #     return precision, recall, F2

                # y_true = torch.tensor([[1,0,0,1]])
                # y_pred = torch.tensor([[1,1,0,0]])
                # print(F1_score(y_pred, y_true))
                    
                # # Get sample every x batches to evaluate.
                # if step % sample_every == 0 and not step == 0:

                #     elapsed = format_time(time.time() - t0)
                #     print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_data_loader), batch_loss, elapsed))
                #     if self.LOGGING:
                #         logger.info(f"{datetime.now()} -- [Model Training]   Batch {step}  of  {len(train_data_loader)}. Loss: {batch_loss}.   Elapsed: {elapsed}.")	

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
                #             if self.LOGGING:
                #                 logger.info(f"{datetime.now()} -- [Model Training] {i}: {tokenizer.decode(sample_output, skip_special_tokens=True)}")	
                        
                        # model.train()

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(self.train_data_loader)
            avg_train_accuracy = total_train_accuracy/len(self.train_data_loader)       

            # # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            # print("")
            # print("  Average training loss: {0:.2f}".format(avg_train_loss))
            # print("  Average training accuracy: {0:.2f}".format(avg_train_accuracy))
            # print("  Training epoch took: {:}".format(training_time))
            if self.LOGGING:
                logger.info(f"{datetime.now()} -- [Model Training]\n  Average training loss: {avg_train_loss}")	
                logger.info(f"{datetime.now()} -- [Model Training]\n  Average training accuracy: {avg_train_accuracy}")	
                logger.info(f"{datetime.now()} -- [Model Training]  Training epoch took: {training_time}")

            # ========================================
			#               Validation
			# ========================================

            print("")
            print("Running Validation...")
            if self.LOGGING:
                logger.info(f"{datetime.now()} -- [Model Validation] Running Validation...")	

            t0 = time.time()

            self.model.eval()

            total_eval_loss = 0
            total_eval_accuracy = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in self.valid_data_loader:
                
                b_input_ids = batch['input_ids'][0].to(device)
                b_token_type_ids = batch['token_type_ids'][0].to(device)
                b_attention_mask = batch['attention_mask'][0].to(device)

                b_labels = batch['labels'][0].to(device)

                b_visual_embeds = batch['visual_embeds'][0].to(device)
                b_visual_attention_mask = batch['visual_attention_mask'][0].to(device)
                b_visual_token_type_ids = batch['visual_token_type_ids'][0].to(device)

                with torch.no_grad():        

                    outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask, token_type_ids=b_token_type_ids, visual_embeds=b_visual_embeds, visual_attention_mask=b_visual_attention_mask, visual_token_type_ids=b_visual_token_type_ids, labels=b_labels)
                
                    loss = outputs[0]
                    logits = outputs[1]
                
                batch_loss = loss.item()

                y_pred = logits.argmax(-1)
                labels_ind = b_labels.argmax(-1)
                val_acc = torch.sum(y_pred == labels_ind)

                total_eval_loss += batch_loss
                total_eval_accuracy += val_acc
                logger.info(f'[Model Validation] - LOSS: {loss}, {batch_loss}, {total_eval_loss}')
                logger.info(f'[Model Validation] - Predicted: {y_pred}, Target: {b_labels}, Accuracy: {val_acc}')

            avg_val_loss = total_eval_loss / len(self.valid_data_loader)
            avg_val_accuracy = total_eval_accuracy / len(self.valid_data_loader)
            
            validation_time = self.format_time(time.time() - t0)    

        #     print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        #     print("  Validation Perplexity: {0:.2f}".format(avg_val_perplexity))
        #     print("  Validation took: {:}".format(validation_time))
            if self.LOGGING:
                logger.info(f"{datetime.now()} -- [Model Validation]   Validation Loss: {avg_val_loss}")	
                logger.info(f"{datetime.now()} -- [Model Validation]   Validation Accuracy: {avg_val_accuracy}")	
                logger.info(f"{datetime.now()} -- [Model Validation]     Validation took: {validation_time}")	

            # Record all statistics from this epoch.
            self.training_stats.append(
                {
                    'epoch': epoch + 1,
                    'Training Loss': avg_train_loss,
                    'Training Accuracy': avg_train_accuracy,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accuracy': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        # print("")
        # print("Training complete!")
        # print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))
        if self.LOGGING:
            logger.info(f"{datetime.now()} -- [Model Training] Training complete!\n Total training took {self.format_time(time.time()-total_t0)} (h:mm:ss)")

        if save_weights:
            self.model_weights_dir = model_weights_dir
            self.save_weights()

    def save_weights(self):
        '''
        Method to save model weights
        Params:
            self: instance of object
        '''
        # Create output directory if needed
        if not os.path.exists(self.model_weights_dir):
            os.makedirs(self.model_weights_dir)

        if self.LOGGING:
            logger.info(f"{datetime.now()} -- [Model] Saving model to {self.model_weights_dir}...")
        print("Saving model to %s" % self.model_weights_dir)

        # Save a trained model, configuration and tokenizer using save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(self.model_weights_dir)
        self.tokenizer.save_pretrained(self.model_weights_dir)

        # Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    
    def format_time(self, elapsed):
        '''
        Method to format time
        Params:
            self: instance of object
            elapsed (float): time elapsed
        
        Returns:
            elapsed time in str
        '''
        return str(dt.timedelta(seconds=int(round((elapsed)))))

    def get_training_stats(self, save_weights=True, model_weights_dir='./results/model_weights/training_stats.csv'):
        '''
        Method to get training stats
        
        Params:
            self: instance of object	
        Returns:
            df_stats (pandas df): training stats	
        '''
        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=self.training_stats)

        # Use the 'epoch' as the row index.
        df_stats = df_stats.set_index('epoch')

        if save_weights:
            df_stats.to_csv(model_weights_dir, index=False)

        # Display the table.
        print(df_stats.head(100))
        logger.info(f'{df_stats.head(100)}')

        return df_stats
    
    def calculate_confusion_matrix():
        '''
        Method to calculate the confusion matrix to calculate macro precision, macro recall, and macro F1 score
        '''

class Model_VisualBERT(VQAModel):
    '''
    VisualBERT Model
    https://huggingface.co/docs/transformers/v4.22.1/en/model_doc/visual_bert#transformers.VisualBertModel
    '''
    def __init__(self, random_state, train_data_loader, valid_data_loader, test_data_loader, model_type='visualbert', log_file=None):
        '''
        Params:
            self: instance of object
            random_state (int): random seed
            train_data_loader (torch.utils.data.DataLoader): train data loader
            valid_data_loader (torch.utils.data.DataLoader): validation data loader
            test_data_loader (torch.utils.data.DataLoader): test data loader
            model_type (str): ['visualbert', ]
            log_file (str): default is None to not have logging, otherwise, specify logging path ../filepath/log.log
        '''
        VQAModel.__init__(self, random_state, train_data_loader, valid_data_loader, test_data_loader, log_file=log_file)
        self.model_type = model_type

        configuration = VisualBertConfig.from_pretrained("uclanlp/visualbert-vcr")
        configuration.__dict__['visual_embedding_dim'] = 512
        self.model = VisualBertForMultipleChoice(configuration)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

    def load_weights(self, model_weights_dir):
        '''
        Method to save model weights
        
        Params:
            self: instance of object
            model_weights_dir (str): model_weights_file
        
        Returns:
            model (torch model): loaded model
        '''
        # Load a trained model and vocabulary that you have fine-tuned
        self.model = VisualBertForMultipleChoice.from_pretrained(model_weights_dir)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
