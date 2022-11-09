"""
plotter.py
Utility functions to plot.
author: @alexiskaldany, @justjoshtings
created: 10/25/22
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from loguru import logger
from pathlib import Path
import os

class Plotter:
    '''
    Object to handle plotting to support understanding of model results.
    '''
    def __init__(self, name='not set'):
        '''
        Params:
            self: instance of object
            name (str): name of plotter object
        '''
        self.name = name

    def plot_model_train_results(self, results_df, save_dir, plotname):
        '''
        Method to plot model training results and save
        Params:
            self: instance of object
            results_df (pandas df): df of results to plot
            save_dir (str): path to save plots to
            plotname (str): name of plot to use
        '''
        self.save_dir = save_dir
        self.plotname = plotname

        # Create output directory if needed
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        results_df['epochs'] = range(1, len(results_df) + 1)

        print(results_df.columns)

        # Loss Plot
        fig = plt.figure(figsize=(12,8))
        ax = plt.axes()
        ax.plot(results_df['epochs'], results_df['Training Loss'], color='blue')
        ax.plot(results_df['epochs'], results_df['Valid. Loss'], color='orange')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Cross Entropy Loss')
        ax.set_title(self.plotname+': Loss over Training Epochs')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.savefig(self.save_dir+'model_loss.png')
        plt.show()

        # Accuracy Plot
        fig = plt.figure(figsize=(12,8))
        ax = plt.axes()
        ax.plot(results_df['epochs'], results_df['Training Accuracy'], color='blue')
        ax.plot(results_df['epochs'], results_df['Valid. Accuracy'], color='orange')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.set_title(self.plotname+': Accuracy over Training Epochs')
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.savefig(self.save_dir+'model_accuracy.png')
        plt.show()

        # Plot Precision, Recall, Specificity, F1 Score
        fig, axs = plt.subplots(2, 2, figsize=(16,10))
        
        # Precision
        axs[0, 0].plot(results_df['epochs'], results_df['Training Precision'], color='blue')
        axs[0, 0].plot(results_df['epochs'], results_df['Valid. Precision'], color='orange')
        axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].set_ylabel('Precision')
        axs[0, 0].title.set_text('Precision')
        
        # Recall
        axs[0, 1].plot(results_df['epochs'], results_df['Training Recall'], color='blue')
        axs[0, 1].plot(results_df['epochs'], results_df['Valid. Recall'], color='orange')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Recall')
        axs[0, 1].title.set_text('Recall')

        # Specificity
        axs[1, 0].plot(results_df['epochs'], results_df['Training Specificity'], color='blue')
        axs[1, 0].plot(results_df['epochs'], results_df['Valid. Specificity'], color='orange')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Specificity')
        axs[1, 0].title.set_text('Specificity')
        
        # F1 Score
        axs[1, 1].plot(results_df['epochs'], results_df['Training F-1 Score'], color='blue')
        axs[1, 1].plot(results_df['epochs'], results_df['Valid. F-1 Score'], color='orange')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('F1 Score')
        axs[1, 1].title.set_text('F1 Score')

        
        plt.legend(['Training', 'Validation'], loc='lower right')
        fig.suptitle(self.plotname+': Other Metrics')
        plt.savefig(self.save_dir+'model_other_metrics.png')
        plt.show()

        # Plot Accuracy per Class
        train_acc_A = [float(i.strip('][').split()[0]) for i in results_df['Training Class Accuracy'].tolist()]
        train_acc_B = [float(i.strip('][').split()[1]) for i in results_df['Training Class Accuracy'].tolist()]
        train_acc_C = [float(i.strip('][').split()[2]) for i in results_df['Training Class Accuracy'].tolist()]
        train_acc_D = [float(i.strip('][').split()[3]) for i in results_df['Training Class Accuracy'].tolist()]

        valid_acc_A = [float(i.strip('][').split()[0]) for i in results_df['Valid. Class Accuracy'].tolist()]
        valid_acc_B = [float(i.strip('][').split()[1]) for i in results_df['Valid. Class Accuracy'].tolist()]
        valid_acc_C = [float(i.strip('][').split()[2]) for i in results_df['Valid. Class Accuracy'].tolist()]
        valid_acc_D = [float(i.strip('][').split()[3]) for i in results_df['Valid. Class Accuracy'].tolist()]

        fig, axs = plt.subplots(2, 2, figsize=(16,10))
        
        # Class A
        axs[0, 0].plot(results_df['epochs'], train_acc_A, color='blue')
        axs[0, 0].plot(results_df['epochs'], valid_acc_A, color='orange')
        axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].set_ylabel('Accuracy')
        axs[0, 0].title.set_text('Class A')
        axs[0, 0].set_ylim([0.2,0.3])

        # Class B
        axs[0, 1].plot(results_df['epochs'], train_acc_B, color='blue')
        axs[0, 1].plot(results_df['epochs'], valid_acc_B, color='orange')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].title.set_text('Class B')
        axs[0, 1].set_ylim([0.2,0.3])

        # Class C
        axs[1, 0].plot(results_df['epochs'], train_acc_C, color='blue')
        axs[1, 0].plot(results_df['epochs'], valid_acc_C, color='orange')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Accuracy')
        axs[1, 0].title.set_text('Class C')
        axs[1, 0].set_ylim([0.2,0.3])
        
        # Class D
        axs[1, 1].plot(results_df['epochs'], train_acc_D, color='blue')
        axs[1, 1].plot(results_df['epochs'], valid_acc_D, color='orange')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].title.set_text('Class D')
        axs[1, 1].set_ylim([0.2,0.3])

        plt.legend(['Training', 'Validation'], loc='lower right')
        fig.suptitle(self.plotname+': Class Accuracy')
        plt.savefig(self.save_dir+'model_class_accuracy.png')
        plt.show()

        # Plot Training and Validation Time
        fig = plt.figure(figsize=(12,8))
        ax = plt.axes()
        ax.plot(results_df['epochs'], results_df['Training Time'], color='blue')
        ax.plot(results_df['epochs'], results_df['Validation Time'], color='orange')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Execution Time')
        ax.set_title(self.plotname+': Execution Time over Training Epochs')
        plt.legend(['Training Time', 'Validation Time'])
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.savefig(self.save_dir+'model_time.png')
        plt.show()

    def plot_model_test_results(self, results_df, save_dir, plotname):
        '''
        Method to plot model testing results and save
        Params:
            self: instance of object
            results_df (pandas df): df of results to plot
            save_dir (str): path to save plots to
            plotname (str): name of plot to use
        '''
        self.save_dir = save_dir
        self.plotname = plotname

        # Create output directory if needed
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        print(results_df.head())
        print(results_df.columns)

        final_epoch_model = results_df.iloc[-1]

        # Plot Overall Metrics Barchart
        metrics_cat = ['Loss', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score']
        metrics = [final_epoch_model['Test Loss'],final_epoch_model['Test Accuracy'],final_epoch_model['Test Precision'],final_epoch_model['Test Recall'],final_epoch_model['Test Specificity'],final_epoch_model['Test F-1 Score']]
        
        fig = plt.figure(figsize=(12,8))
        ax = plt.axes()
        ax.bar(metrics_cat, metrics, color='orange')
        # ax.plot(results_df['epochs'], results_df['Training Loss'], color='blue')
        # ax.plot(results_df['epochs'], results_df['Valid. Loss'], color='orange')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Metric Value')
        ax.set_title(self.plotname+': Test Metrics')
        # plt.legend(['Training', 'Validation'], loc='lower right')
        plt.savefig(self.save_dir+'test_metrics.png')
        plt.show()

        # Plot Individual Class Metrics, 4 subplots precision accuracy recall f1 score

        # Plot Confusion Matrix

    def plot_test_results_comparison():

        # Plot comparison bar plot of all overall metrics

        # Plot comparison sub bar plots of the 4 classes

        # Plot 3 confusion matrix vertical

        pass


