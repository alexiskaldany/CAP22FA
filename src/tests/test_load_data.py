"""
test_load_data.py
Testing: testing loading data, apply annotations, display images, create embeddings
author: @alexiskaldany, @justjoshtings
created: 9/26/22
"""

'''
Import and setup
'''
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

'''
Load data
'''
combined_list = get_data_objects(ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER)
data_df = create_dataframe(combined_list)

'''
Plot data
'''
if not os.path.exists(TEST_IMAGE_OUTPUT):
    os.makedirs(TEST_IMAGE_OUTPUT)

img = cv2.imread(data_df['image_path'][88], cv2.IMREAD_COLOR)
isWritten = cv2.imwrite(str(TEST_IMAGE_OUTPUT) + "/plot_test.png", img)

print(data_df['question'][88])
print(data_df['answer'][88])
print(data_df['list_of_answers'][88])
print(data_df['abcLabel'][88])

test_case_index = 88

text = data_df["question"][test_case_index] + ' ' +  data_df["list_of_answers"][test_case_index][0] + ' ' + data_df["list_of_answers"][test_case_index][1] + ' ' + data_df["list_of_answers"][test_case_index][2] + ' ' + data_df["list_of_answers"][test_case_index][3]
print(text)

'''
Draw Annotations
'''

'''
Visual Embeddings
'''


# print(data_df.head(3)['image_path'])
# print(data_df.head(3)['question'])
# print(data_df.head(3)['answer'])
# print(data_df.head(3)['list_of_answers'])
# print(data_df.head(3)['abcLabel'])
