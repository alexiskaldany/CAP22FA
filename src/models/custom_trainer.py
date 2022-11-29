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
import numpy as np

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
	
    def set_train_parameters(self, num_epochs=1, lr=5e-5, optimizer=None, previous_num_epoch=0):
        '''
        Method to set training parameters
        Params:
            self: instance of object
            num_epochs (int): number of epochs to train, default=1
            lr (float): learning rate, default=5e-5
            optimizer (optimizer): model optimizer to use
            previous_num_epoch (int): previous number of epochs already trained, default 0
        '''
        # Optimizer and Learning Rate Scheduler
        if optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        self.num_epochs = num_epochs
        self.previous_num_epoch = previous_num_epoch

    def train(self, sample_every=100, eval_during_training=False, save_weights=True, model_weights_dir='./results/model_weights/'):
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
            optimizer (optimizer): model optimizer to use
            previous_num_epoch (int): previous number of epochs already trained, default 0
        '''
        num_training_steps = self.num_epochs * len(self.train_data_loader)
        lr_scheduler = get_scheduler(name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

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

        for epoch in range(self.num_epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.num_epochs))
            print(f'Training {self.model._get_name()}...')
            if self.LOGGING:
                logger.info(f"{datetime.now()} -- [Model Training] \n======== Epoch {epoch + 1} / {self.num_epochs} ========")	
                logger.info(f"{datetime.now()} -- [Model Training] Training... {self.model_type}")	
            
            total_train_loss = 0

            total_confusion = {0:[0,0,0,0], 1:[0,0,0,0], 2:[0,0,0,0], 3:[0,0,0,0]}

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

                logger.info(f"Shapes of - b_input_ids: {b_input_ids.shape}, \
                                b_token_type_ids: {b_token_type_ids.shape}, \
                                b_attention_mask: {b_attention_mask.shape}, \
                                b_labels: {b_labels.shape}, \
                                b_visual_embeds: {b_visual_embeds.shape}, \
                                b_visual_attention_mask: {b_visual_attention_mask.shape}, \
                                b_visual_token_type_ids: {b_visual_token_type_ids.shape}, \
                                ")

                self.model.zero_grad()        

                outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask, token_type_ids=b_token_type_ids, visual_embeds=b_visual_embeds, visual_attention_mask=b_visual_attention_mask, visual_token_type_ids=b_visual_token_type_ids, labels=b_labels)

                loss = outputs[0]
                batch_loss = loss.item()
                
                logits = outputs[1]
                m = torch.nn.Softmax(dim=1)
                y_pred = m(logits).argmax(-1)

                labels_ind = b_labels.argmax(-1)
                
                # print('\n\nTRYING ARGMAX!!!',b_labels.argmax(-1))
                # print('TRYING LOGITS!!!',logits)
                # print('TRYING SOFTMAX!!!',m(logits))
                # print('TRYING SOFTMAX!!!',m(logits).sum())
                # print('TRYING sigmoid!!!',torch.sigmoid(logits))
                # print('TRYING sigmoid!!!',torch.sigmoid(logits).sum())

                # print('\n\nTRYING SOFTMAX!!!',torch.sigmoid(logits))
                # print('TRYING SOFTMAX PRED!!!',torch.sigmoid(logits).argmax(-1))
                # print('SUM SOFTMAX PROBS',torch.sigmoid(logits).sum())
                # print('SUM LOGITS',torch.sigmoid(logits).sum())
                # print('ARGMAX PRED!!!',y_pred)
                # print('LABEL!!!',b_labels, b_labels.shape)
                # print('LOGITS!!!',logits, logits.shape)
                # print(outputs)

                # train_acc = torch.sum(y_pred == labels_ind)

                total_train_loss += batch_loss
                logger.info(f'LOSS:  {loss}, {batch_loss}, {total_train_loss}')
                logger.info(f'Predicted: {y_pred}, Target: {b_labels}')

                # total_train_accuracy += train_acc

                total_confusion[labels_ind.item()][y_pred.item()] += 1

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
                self.optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(self.train_data_loader)
            # avg_train_accuracy = total_train_accuracy/len(self.train_data_loader)

            total_confusion_matrix = np.vstack((np.array(total_confusion[0]), np.array(total_confusion[1]), np.array(total_confusion[2]), np.array(total_confusion[3])))
            precision, recall, specificity, accuracy, F1_score, avg_train_accuracy, total_precision, total_recall, total_specificity, total_F1_score = self.calculate_scores_from_confusion_matrix(total_confusion_matrix)

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
            nb_eval_steps = 0

            total_confusion_eval = {0:[0,0,0,0], 1:[0,0,0,0], 2:[0,0,0,0], 3:[0,0,0,0]}

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
                # val_acc = torch.sum(y_pred == labels_ind)

                total_confusion_eval[labels_ind.item()][y_pred.item()] += 1

                total_eval_loss += batch_loss
                # total_eval_accuracy += val_acc
                logger.info(f'[Model Validation] - LOSS: {loss}, {batch_loss}, {total_eval_loss}')
                logger.info(f'[Model Validation] - Predicted: {y_pred}, Target: {b_labels}')

            avg_val_loss = total_eval_loss / len(self.valid_data_loader)
            # avg_val_accuracy = total_eval_accuracy / len(self.valid_data_loader)

            total_confusion_matrix_eval = np.vstack((np.array(total_confusion_eval[0]), np.array(total_confusion_eval[1]), np.array(total_confusion_eval[2]), np.array(total_confusion_eval[3])))
            precision_eval, recall_eval, specificity_eval, accuracy_eval, F1_score_eval, avg_val_accuracy, total_precision_eval, total_recall_eval, total_specificity_eval, total_F1_score_eval = self.calculate_scores_from_confusion_matrix(total_confusion_matrix_eval)
            
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
                    'Training Precision': total_precision,
                    'Training Recall': total_recall,
                    'Training Specificity': total_specificity,
                    'Training F-1 Score': total_F1_score,
                    'Training Class Precision': precision,
                    'Training Class Recall': recall,
                    'Training Class Specificity': specificity,
                    'Training Class Accuracy': accuracy,
                    'Training Class F-1 Score': F1_score,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accuracy': avg_val_accuracy,
                    'Valid. Precision': total_precision_eval,
                    'Valid. Recall': total_recall_eval,
                    'Valid. Specificity': total_specificity_eval,
                    'Valid. F-1 Score': total_F1_score_eval,
                    'Valid. Class Precision': precision_eval,
                    'Valid. Class Recall': recall_eval,
                    'Valid. Class Specificity': specificity_eval,
                    'Valid. Class Accuracy': accuracy_eval,
                    'Valid. Class F-1 Score': F1_score_eval,
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

        # save model checkpoint
        if self.LOGGING:
            logger.info(f"{datetime.now()} -- [Model] Saving model checkpoint to {self.model_weights_dir}checkpoint/model.pth...")
        print(f"Saving model checkpoint to {self.model_weights_dir}checkpoint/model.pth...")

        total_num_epochs = self.previous_num_epoch + self.num_epochs

        if not os.path.exists(self.model_weights_dir+'checkpoint/'):
            os.makedirs(self.model_weights_dir+'checkpoint/')

        torch.save({
                    'epoch': total_num_epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.criterion,
                    }, f'{self.model_weights_dir}checkpoint/model.pth')

        # Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    
    def calculate_scores_from_confusion_matrix(self, cm):
        '''
        Method to calculate other metrics: TP, TN, FP, FN, F1, Recall, Precision, Specificity, Sensitivity

        Params:
            self: instance of object
            cm (np array): confusion matrix
        
        Returns:
        '''
        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP

        num_classes = cm.shape[0]
        TN = []
        for i in range(num_classes):
            temp = np.delete(cm, i, 0)    # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            TN.append(sum(sum(temp)))

        # l = 50
        # for i in range(num_classes):
        #     print(TP[i] + FP[i] + FN[i] + TN[i] == l)

        precision = TP/(TP+FP)
        total_precision = sum(precision)/num_classes
        recall = TP/(TP+FN)
        total_recall = sum(recall)/num_classes
        specificity = TN/(TN+FP)
        total_specificity = sum(specificity)/num_classes
        accuracy = (TP) / (TP+FP)
        total_accuracy = sum(TP) / sum((TP+FP))
        F1_score = 2 * (precision * recall)/(precision + recall)
        total_F1_score = sum(F1_score) / num_classes

        return precision, recall, specificity, accuracy, F1_score, total_accuracy, total_precision, total_recall, total_specificity, total_F1_score
    
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

    def test(self, model_weights_dir='./results/model_weights/testing_stats.csv'):

        self.model.resize_token_embeddings(len(self.tokenizer))
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        print('Using device..', device)
        if self.LOGGING:
            logger.info(f"{datetime.now()} -- [Model Testing] Using device {device}...")	

        self.model.to(device)

        progress_bar = tqdm(range(len(self.test_data_loader)))

        total_t0 = time.time()
        self.testing_stats = []

        gc.collect()
        torch.cuda.empty_cache()

        print("")
        print("Running Testing...")
        if self.LOGGING:
            logger.info(f"{datetime.now()} -- [Model Testing] Running Testing...")	

        t0 = time.time()

        self.model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        total_confusion_test = {0:[0,0,0,0], 1:[0,0,0,0], 2:[0,0,0,0], 3:[0,0,0,0]}

        # Evaluate data for one epoch
        for batch in self.test_data_loader:
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
            # val_acc = torch.sum(y_pred == labels_ind)

            total_confusion_test[labels_ind.item()][y_pred.item()] += 1

            total_eval_loss += batch_loss
            # total_eval_accuracy += val_acc
            logger.info(f'[Model Testing] - LOSS: {loss}, {batch_loss}, {total_eval_loss}')
            logger.info(f'[Model Testing] - Predicted: {y_pred}, Target: {b_labels}')

            progress_bar.update(1)

        avg_val_loss = total_eval_loss / len(self.valid_data_loader)
        # avg_val_accuracy = total_eval_accuracy / len(self.valid_data_loader)

        total_confusion_matrix_eval = np.vstack((np.array(total_confusion_test[0]), np.array(total_confusion_test[1]), np.array(total_confusion_test[2]), np.array(total_confusion_test[3])))
        precision_eval, recall_eval, specificity_eval, accuracy_eval, F1_score_eval, avg_val_accuracy, total_precision_eval, total_recall_eval, total_specificity_eval, total_F1_score_eval = self.calculate_scores_from_confusion_matrix(total_confusion_matrix_eval)
        
        testing_time = self.format_time(time.time() - t0)    

    #     print("  Testing Loss: {0:.2f}".format(avg_val_loss))
    #     print("  Testing Perplexity: {0:.2f}".format(avg_val_perplexity))
    #     print("  Testing took: {:}".format(testing_time))
        if self.LOGGING:
            logger.info(f"{datetime.now()} -- [Model Testing]   Testing Loss: {avg_val_loss}")	
            logger.info(f"{datetime.now()} -- [Model Testing]   Testing Accuracy: {avg_val_accuracy}")	
            logger.info(f"{datetime.now()} -- [Model Testing]     Testing took: {testing_time}")	

        # Record all statistics from this epoch.
        self.testing_stats.append(
            {
                'Index': 0,
                'Test Loss': avg_val_loss,
                'Test Accuracy': avg_val_accuracy,
                'Test Precision': total_precision_eval,
                'Test Recall': total_recall_eval,
                'Test Specificity': total_specificity_eval,
                'Test F-1 Score': total_F1_score_eval,
                'Test Class Precision': precision_eval,
                'Test Class Recall': recall_eval,
                'Test Class Specificity': specificity_eval,
                'Test Class Accuracy': accuracy_eval,
                'Test Class F-1 Score': F1_score_eval,
                'Test Confusion Matrix': total_confusion_matrix_eval,
                'Testing Time': testing_time
            }
        )

         # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=self.testing_stats)

        df_stats = df_stats.set_index('Index')

        df_stats.to_csv(model_weights_dir, index=False)

        # Display the table.
        print(df_stats.head(100))
        logger.info(f'{df_stats.head(100)}')


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
    def __init__(self, random_state, train_data_loader, valid_data_loader, test_data_loader, model_type='visualbert', log_file=None, criterion=torch.nn.CrossEntropyLoss(), model=None, tokenizer=None):
        '''
        Params:
            self: instance of object
            random_state (int): random seed
            train_data_loader (torch.utils.data.DataLoader): train data loader
            valid_data_loader (torch.utils.data.DataLoader): validation data loader
            test_data_loader (torch.utils.data.DataLoader): test data loader
            model_type (str): ['visualbert', ]
            log_file (str): default is None to not have logging, otherwise, specify logging path ../filepath/log.log
            criterion (loss function): default for VisualBERT is torch.nn.CrossEntropyLoss()
        '''
        VQAModel.__init__(self, random_state, train_data_loader, valid_data_loader, test_data_loader, log_file=log_file)
        self.model_type = model_type

        if model is None:
            configuration = VisualBertConfig.from_pretrained("uclanlp/visualbert-vcr")
            configuration.__dict__['visual_embedding_dim'] = 512
            self.model = VisualBertForMultipleChoice(configuration)
        else:
            self.model = model

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
        else:
            self.tokenizer = tokenizer

        self.criterion = criterion

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

    def load_from_checkpoint(self, model_checkpoint_dir):
        '''
        To load from a saved checkpoint

        Params:
            model_checkpoint_dir (str): path to model checkpoint

        Returns:
            self.model (model): previous model state
            self.optimizer (optimizer): previous optimizer state
            self.previous_num_epoch (int): number of epochs previously trained in checkpoint
            self.criterion (loss function state): previous loss function state
        '''
        # load the model checkpoint
        logger.info(f"{datetime.now()} -- [Model Checkpoint Loading] Loading model from checkpoint {model_checkpoint_dir}checkpoint/model.pth...")
        print(f"{datetime.now()} -- [Model Checkpoint Loading] Loading model from checkpoint {model_checkpoint_dir}checkpoint/model.pth...")
        checkpoint = torch.load(model_checkpoint_dir / 'checkpoint/model.pth')

        # load model weights state_dict
        self.model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vcr", state_dict = checkpoint['model_state_dict'], ignore_mismatched_sizes=True)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"{datetime.now()} -- [Model Checkpoint Loading] Previously trained model weights state_dict loaded...")
        print('Previously trained model weights state_dict loaded...')
        
        # load trained optimizer state_dict
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"{datetime.now()} -- [Model Checkpoint Loading] Previously trained optimizer state_dict loaded...")
        print('Previously trained optimizer state_dict loaded...')
        
        # load last number of epochs
        self.previous_num_epoch = checkpoint['epoch']
       
        # load the criterion
        self.criterion = checkpoint['loss']

        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_checkpoint_dir, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

        return self.model, self.optimizer, self.previous_num_epoch, self.criterion, self.tokenizer
    
    

    
