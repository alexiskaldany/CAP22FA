"""
plot_model_results.py
Plot model results
author: @alexiskaldany, @justjoshtings
created: 10/25/22
"""

from loguru import logger
import os
import sys
from datetime import datetime
from sympy import re
import pandas as pd

# get current directory
path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
# add src to executable path to allow imports from src
sys.path.insert(0, parent_path)

from src.utils.configs import DATA_JSON, DATA_CSV, DATA_DIRECTORY, ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER, ANNOTATED_IMAGES_FOLDER, TEST_DIRECTORY, TEST_IMAGE_OUTPUT
from src.utils.configs import RANDOM_STATE
from src.utils.plotter import Plotter

random_state = RANDOM_STATE

'''
Set logger
'''
logger.remove()
logger.add(
    "./logs/training_log.txt",
    # sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss}|{level}| {message}|{function}: {line}",
    level="INFO",
    backtrace=True,
    colorize=True,
)

logger.info(f"\n\n\n[Plotting results - {datetime.now()}] Running plot model results....")

'''
Helper stitch results function
'''
def stitch_model_results(list_to_stitch, results_dir):
    '''
    Method to stitch different epoch's model results into one df
    Params:
        list_to_stitch (list): list of paths to stitch into one df
    Returns:
        train_results_df (pandas df): single dataframe of all model epoch's results for training and validation
        test_results_df (pandas df): single dataframe of model's testing results every run (4 epochs)
    '''
    train_frames = []
    test_frames = []

    for run in list_to_stitch:
        current_training_path = os.getcwd()+results_dir+run+'/training_stats.csv'
        current_testing_path = os.getcwd()+results_dir+run+'/testing_stats.csv'
        
        current_train_results = pd.read_csv(current_training_path)
        current_test_results = pd.read_csv(current_testing_path)

        train_frames.append(current_train_results)
        test_frames.append(current_test_results)

    train_results_df = pd.concat(train_frames)
    test_results_df = pd.concat(test_frames)

    return train_results_df, test_results_df

'''
Run stitching and call plotter
'''
plot_name = 'visualbert_annotations_full_run'
results_to_stitch = ['visualbert_RUN_2_4epochs', 'visualbert_RUN_2_8epochs', 'visualbert_RUN_2_12epochs', 'visualbert_RUN_2_16epochs','visualbert_RUN_2_20epochs']
results_dir = '/results/model_weights/'

final_model_name = ' e'.join(results_to_stitch[-1].split('_')[-1].split('e'))

train_results_df, test_results_df  = stitch_model_results(results_to_stitch, results_dir)
model_results_plotter = Plotter('results_plotter')
# model_results_plotter.plot_model_train_results(train_results_df, save_dir=f'./results/plot_results/train_{plot_name}/', plotname='VisualBERT w/ Visual Annotations')
model_results_plotter.plot_model_test_results(test_results_df, save_dir=f'./results/plot_results/test_{plot_name}/', plotname=f'VisualBERT w/ Visual Annotations - {final_model_name}')

    

