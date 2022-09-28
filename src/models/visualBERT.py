from transformers import (
    BertTokenizer,
    VisualBertForMultipleChoice,
    DeiTFeatureExtractor,
)
import torch
import numpy as np
torch.__version__
torch.cuda.is_available()
from pathlib import Path
import glob
import json
from PIL import Image

""" 
Data
"""
DATA_DIRECTORY = Path(__file__).parent / "data"
ANNOTATION_FOLDER = DATA_DIRECTORY / "ai2d" / "annotations"
IMAGES_FOLDER = DATA_DIRECTORY / "ai2d" / "images"
QUESTIONS_FOLDER = DATA_DIRECTORY / "ai2d" / "questions"

number_of_images = len(glob.glob(str(IMAGES_FOLDER / "*.png")))
print(f"Number of images: {number_of_images}")
# annotation_list = [str(ANNOTATION_FOLDER)+"/"+str(i)+".png.json" for i in range(number_of_images)]

""" 
Loading all the file paths
"""
image_glob = glob.glob(str(IMAGES_FOLDER / "*.png"))
image_ids = [int(Path(image_path).stem) for image_path in image_glob]
image_path_dict = {k:v for k,v in zip(image_ids, image_glob)}
annotation_glob = glob.glob(str(ANNOTATION_FOLDER / "*.png.json"))
annotation_ids = [
    int(annotation_path.split(".")[-3].split("/")[-1])
    for annotation_path in annotation_glob
]
annotation_path_dict = {k:v for k,v in zip(annotation_ids, annotation_glob)}
questions_glob = glob.glob(str(QUESTIONS_FOLDER / "*.png.json"))
question_ids = [
    int(question_path.split(".")[-3].split("/")[-1]) for question_path in questions_glob
]
question_path_dict = {k:v for k,v in zip(question_ids, questions_glob)}
id_list = set(image_ids + annotation_ids + question_ids)
id_index = [(id, index) for index, id in enumerate(id_list)]
# unmatched_ids = set(image_ids) - set(annotation_ids) - set(question_ids)
# print(len(unmatched_ids))



# image_dimensions = [Image.open(image_glob[i]).size for i in range(number_of_images)]
# average_dims = [
#     sum([image_dimensions[i][j] for i in range(number_of_images)]) / number_of_images
#     for j in range(2)
# ]
image_dimensions = (620, 480)
""" 
Processing the images
"""


def load_image_resize_convert(image_path):
    image = Image.open(image_path)
    image = image.resize(image_dimensions)
    image = image.convert("RGB")
    return image


""" 
Image Feature Extraction
"""
# print(f"Average image dimensions: {average_dims}")
# print(f"Image dimensions: {image_dimensions[:10]}")
feature_extractor = DeiTFeatureExtractor.from_pretrained(
    "facebook/deit-base-distilled-patch16-224"
)
def get_image_features(image_path):
    image = load_image_resize_convert(image_path)
    image_feature = feature_extractor(
        images=image, return_tensors="pt", do_normalize=True
    ).pixel_values
    # pixel_values — Pixel values to be fed to a model, of shape (batch_size, num_channels, height, width).
    batch_size = 1
    num_choices = 4
    image_features_shape = [*image_feature.shape]
    print(image_features_shape)
    visual_embeds = image_feature.expand(*image_feature.shape)
    return visual_embeds
def compile_image_features(image_paths):
    image_features = []
    for image_path in image_paths:
        image_features.append(get_image_features(image_path))
    return image_features
def save_as_csv(image_paths):
    image_features = np.concatenate(compile_image_features(image_paths))
    np.savetxt(DATA_DIRECTORY / "image_features.csv", image_features, delimiter=",",fmt='%s')
    return image_features
"""
Loading Questions and Answers
"""
def get_question_and_answers(id,path):
    with open(path, "r") as f:
        questions_dict = json.load(f)
    question_list = []
    for key, value in questions_dict["questions"].items():
        question_dict = {
            "image_id": id,
            "image_path": image_path_dict[id],
            "question_id": value["questionId"],
            "question": key,
            "answer": value["answerTexts"][value["correctAnswer"]],
            "correct_answer_id": value["correctAnswer"],
            "answer_type": value["abcLabel"],
            "answer_choices": value["answerTexts"],
        }
        question_list.append(question_dict)
    return question_list
def question_and_answers_list(question_path_dict):
    question_and_answers_list = []
    for k,v in question_path_dict.items():
        question_and_answers_list.extend(get_question_and_answers(k,v))
    return question_and_answers_list
# with open(DATA_DIRECTORY / "question_and_answers_dict.json", "w") as f:
#     json.dump(question_and_answers_list(question_path_dict), f)
print(get_image_features(image_glob[0]).shape)

""" 
Loading All Answers
"""
from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax
""" 
Tokenization
"""
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vbmc_model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vcr")

