# """
# test_load_data.py
# Testing: testing loading data, apply annotations, display images, create embeddings
# author: @alexiskaldany, @justjoshtings
# created: 9/26/22
# """

# '''
# Import and setup
# '''
# from loguru import logger
# import os
# import sys
# import cv2

# # get current directory
# path = os.getcwd()
# parent_path = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
# # add src to executable path to allow imports from src
# sys.path.insert(0, parent_path)

# from src.utils.configs import DATA_JSON, DATA_CSV, DATA_DIRECTORY, ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER, ANNOTATED_IMAGES_FOLDER, TEST_DIRECTORY, TEST_IMAGE_OUTPUT
# from src.utils.prepare_and_download import get_data_objects, create_dataframe
# from src.utils.applying_annotations import execute_full_set_annotation

# '''
# Load data
# '''
# combined_list = get_data_objects(ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER)
# data_df = create_dataframe(combined_list)

# '''
# Plot data
# '''
# if not os.path.exists(TEST_IMAGE_OUTPUT):
#     os.makedirs(TEST_IMAGE_OUTPUT)

# img = cv2.imread(data_df['image_path'][88], cv2.IMREAD_COLOR)
# isWritten = cv2.imwrite(str(TEST_IMAGE_OUTPUT) + "/plot_test.png", img)

# print(data_df['question'][88])
# print(data_df['answer'][88])
# print(data_df['list_of_answers'][88])
# print(data_df['abcLabel'][88])

# test_case_index = 88

# text = data_df["question"][test_case_index] + ' ' +  data_df["list_of_answers"][test_case_index][0] + ' ' + data_df["list_of_answers"][test_case_index][1] + ' ' + data_df["list_of_answers"][test_case_index][2] + ' ' + data_df["list_of_answers"][test_case_index][3]
# print(text)

# '''
# Draw Annotations
# '''

# '''
# Visual Embeddings
# '''


# # print(data_df.head(3)['image_path'])
# # print(data_df.head(3)['question'])
# # print(data_df.head(3)['answer'])
# # print(data_df.head(3)['list_of_answers'])
# # print(data_df.head(3)['abcLabel'])

import torch
import numpy as np

def ohe_labels(label):
    if label == 0:
        labels = torch.tensor([1, 0, 0, 0])
    elif label == 1:
        labels = torch.tensor([0, 1, 0, 0])
    elif label == 2:
        labels = torch.tensor([0, 0, 1, 0])
    elif label == 3:
        labels = torch.tensor([0, 0, 0, 1])

    return labels
    
labels = torch.tensor([2.6223, 1.8740, 4.0509, 3.9332])
print(labels)
print(labels.argmax(-1))
print(ohe_labels(torch.tensor(1)))

outputs_before_sigmoid = torch.randn(1, 4)
sigmoid_outputs = torch.sigmoid(outputs_before_sigmoid)
target_classes = torch.randint(0, 2, (1, 4))

print(outputs_before_sigmoid)
print(sigmoid_outputs)
print(target_classes)

'''
Testing loss = torch.nn.CrossEntropyLoss()

Recreating error to troubleshoot:
    return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
        RuntimeError: Expected floating point type for target with class probabilities, got Long

https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
'''
# Check this out!
# how to go from logits of nn.CrossEntropyLoss() to target labels?

# Example of target with class indices
print('crossentropy')
loss = torch.nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
print(input)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target)
output = loss(input, target)
print(output)
output.backward()
# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
print(input)
target = torch.randn(3, 5).softmax(dim=1)
print(target)
output = loss(input, target)
print(output)
output.backward()

test_logits = torch.tensor([[0.0911, 0.2703, 1.7605, 1.8468]],requires_grad=True)
# test_labels = torch.tensor([3])
test_labels = torch.tensor([[0., 0., 1., 0.]],requires_grad=True)

output = loss(test_logits,test_labels)
print(test_logits, test_logits.shape)
print(test_labels, test_labels.shape)
print(output)
output.backward()