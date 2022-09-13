
from pathlib import Path
import glob
import json
from PIL import Image
from transformers import (
    DeiTFeatureExtractor,
)
import torch

""" 
Data
"""
DATA_DIRECTORY = Path(__file__).parent.parent / "data"
ANNOTATION_FOLDER = DATA_DIRECTORY / "ai2d" / "annotations"
IMAGES_FOLDER = DATA_DIRECTORY / "ai2d" / "images"
QUESTIONS_FOLDER = DATA_DIRECTORY / "ai2d" / "questions"
DATA_JSON = DATA_DIRECTORY / "question_and_answers_dict_id_key.json"
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

""" 
Processing the images
"""
image_dimensions = (620, 480)
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
visual_feature_extractor = DeiTFeatureExtractor.from_pretrained(
    "facebook/deit-base-distilled-patch16-224"
)
def get_image_features(image_path,visual_feature_extractor):
    image = load_image_resize_convert(image_path)
    image_feature = visual_feature_extractor(
        images=image, return_tensors="pt", do_normalize=True
    )
    # pixel_values — Pixel values to be fed to a model, of shape (batch_size, num_channels, height, width).
    # image_features_shape = [*image_feature.pixel_values.shape]
    return image_feature
"""
Loading Questions and Answers
"""
def generate_question_id_path_dict(question_folder_path):
    questions_glob = glob.glob(str(QUESTIONS_FOLDER / "*.png.json"))
    question_ids = [
        int(question_path.split(".")[-3].split("/")[-1]) for question_path in questions_glob
    ]
    question_path_dict = {k:v for k,v in zip(question_ids, questions_glob)}
    return question_path_dict
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

def question_and_answers_dict(question_path_dict):
    id_question_and_answers_dict = {}
    for k,v in question_path_dict.items():
        id_question_and_answers_dict[k] =(get_question_and_answers(k,v))
    return id_question_and_answers_dict

tensor = get_image_features(image_glob[0],visual_feature_extractor).pixel_values

print(tensor.shape)
visual_embeds = tensor.expand(size=(1,4,tensor.shape[1],tensor.shape[2],tensor.shape[2]))
print(visual_embeds.shape)
# batch_json_creation = question_and_answers_dict(question_path_dict)
# with open(DATA_DIRECTORY / "question_and_answers_dict_id_key.json", "w") as f:
#      json.dump(batch_json_creation, f)