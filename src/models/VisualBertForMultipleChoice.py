#%%
from pathlib import Path
from transformers import BertTokenizer, VisualBertForMultipleChoice
from transformers import Trainer, TrainingArguments
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd
import json
import sys
import os
import glob
import random
from loguru import logger
from typing import Tuple
import math

logger.remove()
logger.add(
    sys.stdout,
    format="<light-yellow>{time:YYYY-MM-DD HH:mm:ss}</light-yellow> | <light-blue>{level}</light-blue> | <cyan>{message}</cyan> | <light-red>{function}: {line}</light-red>",
    level="INFO",
    backtrace=True,
    colorize=True,
)

## Constants
#############################################
# Internal Directories and Folders

DATA_DIRECTORY = Path(__file__).parent.parent / "data"
AI2D_FOLDER = DATA_DIRECTORY / "ai2d"
ANNOTATION_FOLDER = DATA_DIRECTORY / "ai2d" / "annotations"
os.system(f"rm {DATA_DIRECTORY}")
if not DATA_DIRECTORY.exists(): 
    os.makedirs(DATA_DIRECTORY)
if not AI2D_FOLDER.exists():
    os.makedirs(AI2D_FOLDER)
IMAGES_FOLDER = DATA_DIRECTORY / "ai2d" / "images"
QUESTIONS_FOLDER = DATA_DIRECTORY / "ai2d" / "questions"
ANNOTATED_IMAGES_FOLDER = DATA_DIRECTORY / "ai2d" / "annotated_images"
if not ANNOTATED_IMAGES_FOLDER.exists():
    os.makedirs(ANNOTATED_IMAGES_FOLDER)
if ANNOTATED_IMAGES_FOLDER.exists() == False:
    os.mkdir(ANNOTATED_IMAGES_FOLDER)
RUNS_FOLDER = Path(__file__).parent.parent / "runs"
SAVED_MODELS_FOLDER = Path(__file__).parent.parent / "models/saved_models"
DATA_JSON = DATA_DIRECTORY / "data_set.json"
DATA_CSV = DATA_DIRECTORY / "data.csv"
FINISHED_DATA_CSV = DATA_DIRECTORY / "finished_data.csv"
FULL_INPUT_DICT_JSON = DATA_DIRECTORY / "full_input_dict.json"
TRAIN_JSON = DATA_DIRECTORY / "train_set.json"
VAL_JSON = DATA_DIRECTORY / "val_set.json"
TEST_JSON = DATA_DIRECTORY / "test_set.json"
IMAGE_DIMENSIONS = (620, 480)
ANNOTATION_THICKNESS = int(2)
## Loading the data
#############################################
def download_data(DATA_DIRECTORY:Path,ANNOTATED_IMAGES_FOLDER:Path):
    """
    Creates needed folders then downloads, unzips and deletes the zip file
    """
    try:
        logger.info("Downloading the data")
        install_aws = f"sudo apt-get install awscli"
        os.system(install_aws)
        download_command = f"aws s3 cp --no-sign-request s3://ai2-public-datasets/diagrams/ai2d-all.zip {DATA_DIRECTORY}"
        os.system(download_command)
        logger.info("Completed downloading the data")
        logger.info("Unzipping the data")
        os.system(f"unzip {DATA_DIRECTORY}/ai2d-all.zip -d {DATA_DIRECTORY}")
        logger.info("Completed unzipping the data")
        logger.info("Deleting the zip file")
        os.system(f"rm {DATA_DIRECTORY}/ai2d-all.zip && rm {DATA_DIRECTORY}/__MACOSX -rf")
        print("Download and cleanup complete")
        return True
    except Exception as e:
        logger.error(f"Error in downloading the data: {e}")
        return False

def get_data_objects(ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER) -> list:
    """
    Takes in the annotation, images and questions folder and returns a list of dictionaries
    1 dictionary per image ID
    """
    image_glob = glob.glob(str(IMAGES_FOLDER / "*.png"))
    image_ids = [int(Path(image_path).stem) for image_path in image_glob]
    image_path_dict = {k: v for k, v in zip(image_ids, image_glob)}
    annotation_glob = glob.glob(str(ANNOTATION_FOLDER / "*.png.json"))
    annotation_ids = [
        int(annotation_path.split(".")[-3].split("/")[-1])
        for annotation_path in annotation_glob
    ]
    annotation_path_dict = {k: v for k, v in zip(annotation_ids, annotation_glob)}
    questions_glob = glob.glob(str(QUESTIONS_FOLDER / "*.png.json"))
    question_ids = [
        int(question_path.split(".")[-3].split("/")[-1])
        for question_path in questions_glob
    ]
    question_path_dict = {k: v for k, v in zip(question_ids, questions_glob)}
    id_list = set(image_ids).intersection(annotation_ids).intersection(question_ids)
    img_number = len(image_ids)
    logger.info(f"Number of images: {img_number}")
    data_list = []
    for id in id_list:
        try:
            image_dict = {"image_path": image_path_dict[id]}
        except:
            image_dict = {"image_path": None}
            logger.exception(f"Image not found for id: {id}")
        try:
            annotation_dict = json.load(open(annotation_path_dict[id]))
        except:
            annotation_dict = {}
        try:
            question_dict = json.load(open(question_path_dict[id]))
        except:
            question_dict = {}
            logger.exception(
                f"Error in loading the question file: {question_path_dict[id]}"
            )
        temp_list = [image_dict, annotation_dict, question_dict]
        data_list.append(temp_list)
    return data_list

def create_row_per_question_dataframe(data_list: list) -> pd.DataFrame:
    """
    Input: list of dictionaries (the output of get_data_objects)
    Output: Pandas dataframe, 1 row per question
    """
    list_of_dicts = []
    for data in data_list:
        image_id = data[2]["imageName"].split(".")[0]
        image_path = data[0]["image_path"]
        img_dict = {"image_id": image_id, "image_path": image_path}
        for question_key in data[2]["questions"].keys():
            question = question_key
            list_of_answers = data[2]["questions"][question_key]["answerTexts"]
            answer = list_of_answers[
                data[2]["questions"][question_key]["correctAnswer"]
            ]
            abcLabel = data[2]["questions"][question_key]["abcLabel"]
            question_dict = {
                "question": question,
                "list_of_answers": list_of_answers,
                "answer": answer,
                "abcLabel": abcLabel,
            }
            temp_dict = {**img_dict, **question_dict}
            list_of_dicts.append(temp_dict)
    df = pd.DataFrame(list_of_dicts)
    return df


def create_train_val_test_split(
    dataframe: pd.DataFrame, TRAIN_SPLIT: float, VAL_SPLIT: float, TEST_SPLIT: float
) -> tuple:
    logger.info("Creating train, val, test split")
    train_index = int(TRAIN_SPLIT * dataframe.shape[0])
    val_index = int((TRAIN_SPLIT + VAL_SPLIT) * dataframe.shape[0])
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    train = dataframe.iloc[:train_index, :]
    val = dataframe.iloc[train_index:val_index, :]
    test = dataframe.iloc[val_index:, :]
    logger.info(f"Train: {len(train)}")
    logger.info(f"Val: {len(val)}")
    logger.info(f"Test: {len(test)}")
    return train, val, test


# data_list = get_data_objects(ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER)
# with open(DATA_JSON, "w") as f:
#     json.dump(data_list, f)
# data_set = json.load(open(DATA_JSON))[:250]
# dataframe = create_row_per_question_dataframe(data_list)
# dataframe.to_csv(DATA_CSV)
# dataframe = pd.read_csv(DATA_CSV)[:250]
## Applying the annotations
#############################################
import cv2
from cv2 import imread, imwrite, rectangle, putText, FONT_HERSHEY_SIMPLEX, LINE_AA


def drawing_labels(
    Img: cv2.imread, label_dict: dict, color: Tuple[int, int, int]
) -> cv2.imread:
    """
    Draws labels on an image
    """
    label_id = label_dict["id"]
    coordinates_list = label_dict["rectangle"]
    replacement_text = label_dict["replacementText"]
    value = label_dict["value"]
    Img = cv2.rectangle(
        img=Img,
        pt1=coordinates_list[0],
        pt2=coordinates_list[1],
        color=color,
        thickness=1,
    )
    Img = cv2.putText(
        img=Img,
        text=value,
        org=coordinates_list[0],
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=color,
        thickness=ANNOTATION_THICKNESS,
    )
    return Img


def drawing_arrows(Img, coordinate_list, color: Tuple[int, int, int]):
    """
    Draws an arrow on an image
    """
    Img = cv2.polylines(
        Img,
        pts=[np.array(coordinate_list)],
        isClosed=False,
        color=color,
        thickness=ANNOTATION_THICKNESS,
    )
    return Img


def drawing_arrow_heads(
    Img, coordinate_list: list, color: Tuple[int, int, int], angle: float
):
    """
    pt1 = (x1,y1)
    pt2 = (x2,y2)
    pt3 = (? , ?)
    angle1_2_3 == angle2_1_3 == orientation/2
    side1_2 = ((x1-x2)**2 + (y1-y2)**2)**0.5
    side2_3 = side1_2 * tan(angle1_2_3)
    side_1_3 = ((x1-x3)**2 + (y1-y3)**2)**0.5

    m1 = np.arctan()
    Bleh trig...
    Lets just draw a circle using the two points

    midpoint = [(x1+x2)/2,(y1+y2)/2]
    radius = pt1[0]-pt2[0]
    """
    Center_point = [
        int((coordinate_list[0][0] + coordinate_list[1][0]) / 2),
        int((coordinate_list[0][1] + coordinate_list[1][1]) / 2),
    ]
    Radius = int(
        math.sqrt(
            (coordinate_list[0][0] - coordinate_list[1][0]) ** 2
            + (coordinate_list[0][1] - coordinate_list[1][1]) ** 2
        )
    )
    logger.debug(f"{Center_point,Radius}")
    Img = cv2.circle(
        img=Img,
        center=Center_point,
        radius=Radius,
        color=color,
        thickness=ANNOTATION_THICKNESS,
    )
    return Img


def drawing_blobs(Img, coordinate_list: list, color: Tuple[int, int, int]):
    """
    Draws blobs on an image
    """
    Img = cv2.polylines(
        Img, pts=[np.array(coordinate_list)], isClosed=True, color=color, thickness=1
    )
    return Img


def generate_random_color_tupple() -> Tuple[int, int, int]:
    return (
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(0, 255),
    )


def full_annotation(annotation_dict: dict, image_path: str, annotated_image_path: str):
    """
    Iterates through the annotation dictionary and draws the annotations on the image

    """
    Img_list = [cv2.imread(image_path)]
    label_number = len([annotation_dict["text"].keys()])
    logger.debug(f"Drawing labels: {label_number} ")
    for key in list(annotation_dict["text"].keys()):
        logger.debug(f"Drawing label: {annotation_dict['text'][key]} ")
        Img_list.append(
            drawing_labels(
                Img_list[-1],
                annotation_dict["text"][key],
                generate_random_color_tupple(),
            )
        )
    arrow_number = len(list(annotation_dict["arrows"].keys()))
    logger.debug(f"Drawing arrows:{arrow_number}")
    for key in list(annotation_dict["arrows"].keys()):
        Img_list.append(
            drawing_arrows(
                Img_list[-1],
                annotation_dict["arrows"][key]["polygon"],
                generate_random_color_tupple(),
            )
        )
    arrowhead_number = len(list(annotation_dict["arrowHeads"].keys()))
    logger.debug(f"Drawing arrow heads:{arrowhead_number}")
    for key in list(annotation_dict["arrowHeads"].keys()):
        Img_list.append(
            drawing_arrow_heads(
                Img_list[-1],
                annotation_dict["arrowHeads"][key]["rectangle"],
                annotation_dict["arrowHeads"][key]["orientation"],
                generate_random_color_tupple(),
            )
        )
    logger.debug(f"Drawing blobs:{arrow_number}")
    for key in list(annotation_dict["blobs"].keys()):
        Img_list.append(
            drawing_blobs(
                Img_list[-1],
                annotation_dict["blobs"][key]["polygon"],
                generate_random_color_tupple(),
            )
        )
    cv2.imwrite(annotated_image_path, Img_list[-1])
    return annotated_image_path


def execute_full_set_annotation(DATA_LIST_PATH: Path, ANNOTATED_IMAGES_FOLDER: Path):
    """
    full_dict[x][1] = the annotation dictionary for the image
    full_dict[x][0]["image_path"] = the local image path
    {ANNOTATED_IMAGES_FOLDER}/{full_dict[x][0]['image_path'].split('/')[-1].split('.')[0]}.png" = The path for the newly created annotated image
    """
    if not ANNOTATED_IMAGES_FOLDER.exists():
        os.makedirs(ANNOTATED_IMAGES_FOLDER)
    full_dict = json.load(open(DATA_LIST_PATH))
    annotated_image_paths = [
        full_annotation(
            full_dict[x][1],
            full_dict[x][0]["image_path"],
            f"{ANNOTATED_IMAGES_FOLDER}/{full_dict[x][0]['image_path'].split('/')[-1].split('.')[0]}.png",
        )
        for x in range(len(full_dict))
    ]
    return annotated_image_paths

## Getting Visual Embeddings
#############################################
from torchvision.models.resnet import resnet18 as _resnet18


def copy_embeddings(m, i, o):
    """Copy embeddings from the penultimate layer."""
    o = o[:, :, 0, 0].detach().numpy().tolist()
    embeddings.append(o)
    return


def load_image_resize_convert(image_path):
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(Image.open(image_path))
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def get_multiple_embeddings(list_of_images: list):
    global embeddings
    embeddings = []
    id_embedding_dict = {}
    # image_embedding_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=models.ResNet18_Weights.DEFAULT')
    image_embedding_model = _resnet18(weights="DEFAULT")
    layer = image_embedding_model._modules.get("avgpool")
    _ = layer.register_forward_hook(copy_embeddings)
    image_embedding_model.eval()
    for idx, image in enumerate(set(list_of_images)):
        if idx % 100 == 0:
            logger.info(f"At index:{idx}")
        image_tensor = load_image_resize_convert(image)
        _ = image_embedding_model(image_tensor)
        embedding = embeddings.pop()[0]
        # logger.debug(f"Embedding shape:{embedding.shape}")
        tensor = torch.as_tensor(embedding)
        # logger.debug(f"Tensor shape:{tensor.shape}")
        id_embedding_dict[Path(image).name.split(".")[0]] = tensor
        # image.split("/")[-1].split(".")[0]
    return id_embedding_dict

#%%
### Starting scripts
download_data(DATA_DIRECTORY,ANNOTATED_IMAGES_FOLDER)
data_list = get_data_objects(ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER)
with open(DATA_JSON, "w") as f:
    json.dump(data_list, f)
# data_set = json.load(open(DATA_JSON))
dataframe = create_row_per_question_dataframe(data_list)
dataframe.to_csv(DATA_CSV)
# dataframe = pd.read_csv(DATA_CSV)
execute_full_set_annotation(DATA_JSON, ANNOTATED_IMAGES_FOLDER)
logger.info("Getting annotated images embeddings")
id_list = list(dataframe["image_id"])
annotated_image_path = [str(ANNOTATED_IMAGES_FOLDER / f"{id}.png") for id in id_list]
dataframe["annotated_image_path"] = annotated_image_path
annotated_list = []
raw_list = []
annotated_images_embeddings = get_multiple_embeddings(
    dataframe["annotated_image_path"].to_list()
)
# logger.debug(f"Annotated images embeddings:{annotated_images_embeddings.get('235')}")
annotated_images_embeddings = [
    v.expand(1, 4, *v.shape) for v in annotated_images_embeddings.values()
]
raw_images_embeddings = get_multiple_embeddings(dataframe["image_path"].to_list())
raw_images_embeddings = [
    x.expand(1, 4, *x.shape) for x in raw_images_embeddings.values()
]
logger.debug(f"Raw images embeddings len:{len(raw_images_embeddings)}")
logger.debug(f"Annotated images embeddings len:{len(annotated_images_embeddings)}")
first_raw = raw_images_embeddings[0].shape
## Adding the visual embeddings to the dataframe
#############################################
for index, row in dataframe.iterrows():
    if index % 100 == 0:
        logger.info(f"At index: {index}")
    img_id = str(row["image_id"])
    if img_id in annotated_images_embeddings:
        annotated_list.append(annotated_images_embeddings[img_id])
    if img_id not in annotated_images_embeddings:
        annotated_list.append([])
    if img_id in raw_images_embeddings:
        raw_list.append(raw_images_embeddings[img_id])
    if img_id not in raw_images_embeddings:
        raw_list.append([])
# if dataframe.shape[0] == len(annotated_list):
#     dataframe["annotated_images_embeddings"] = annotated_list
# if dataframe.shape[0] == len(raw_list):
#     dataframe["raw_image_embeddings"] = raw_list
visual_token_type_ids = [
    torch.ones(first_raw[:-1], dtype=torch.long) for i in range(dataframe.shape[0])
]
# dataframe["visual_token_type_ids"] = visual_token_type_ids
visual_attention_mask = [
    torch.ones(first_raw[:-1], dtype=torch.float) for i in range(dataframe.shape[0])
]
# dataframe["visual_attention_mask"] = visual_attention_mask
#%%
## Creating the text embeddings
#############################################
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_embeddings = []
for index, row in dataframe.iterrows():
    if index % 100 == 0:
        logger.info(f"At index: {index}")
    questions = [row["question"], row["question"], row["question"], row["question"]]
    answers = [
        row["list_of_answers"][0],
        row["list_of_answers"][1],
        row["list_of_answers"][2],
        row["list_of_answers"][3],
    ]
    embeds = tokenizer(
        questions,
        answers,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    #logger.debug(f"Embeds:{type(embeds)}")
    text_embeddings.append(embeds)


## Creating the labels
#############################################
list_of_answers = dataframe["list_of_answers"].to_list()
answers = dataframe["answer"].to_list()
answer_index = [
    0
    if answers[i] == list_of_answers[i][0]
    else 1
    if answers[i] == list_of_answers[i][1]
    else 2
    if answers[i] == list_of_answers[i][2]
    else 3
    for i in range(len(answers))
]
dataframe["answer_index"] = answer_index
dataframe["labels"] = dataframe["answer_index"].apply(
    lambda x: torch.tensor(0).unsqueeze(0)
    if x == "0"
    else torch.tensor(1).unsqueeze(0)
    if x == "1"
    else torch.tensor(2).unsqueeze(0)
    if x == "2"
    else torch.tensor(3).unsqueeze(0)
)
# dataframe = dataframe.drop(columns=["Unnamed: 0"])

## Creating full input dict, organized as a list of dicts

full_input_dict = []
for index in range(dataframe.shape[0]):
    if index % 100 == 0:
        logger.info(f"At index: {index}")
    full_input_dict.append(
        {
            "input_ids": text_embeddings[index]["input_ids"],
            "token_type_ids": text_embeddings[index]["token_type_ids"],
            "attention_mask": text_embeddings[index]["attention_mask"],
            "raw_images_embeddings": raw_images_embeddings[index],
            "annotated_images_embeddings": annotated_images_embeddings[index],
            "visual_token_type_ids": visual_token_type_ids[index],
            "visual_attention_mask": visual_attention_mask[index],
            "labels": dataframe["labels"][index],
        }
    )
# with open(FULL_INPUT_DICT_JSON, "w") as f:
#     json.dump(full_input_dict, f)

## Saving the dataframe
#############################################
dataframe.to_csv(FINISHED_DATA_CSV, index=False)

## Loading the dataframe
#############################################
dataframe = pd.read_csv(FINISHED_DATA_CSV)

## Creating train,val and test sets
#############################################
train, val, test = create_train_val_test_split(dataframe, 0.8, 0.1, 0.1)
## Creating input list
#############################################
training_input_dict = []
for index in range(train.shape[0]):
    training_input_dict.append(
        {
            "input_ids": full_input_dict[index]["input_ids"],
            "token_type_ids": full_input_dict[index]["token_type_ids"],
            "attention_mask": full_input_dict[index]["attention_mask"],
            "visual_embeds": full_input_dict[index]["annotated_images_embeddings"],
            "visual_token_type_ids": full_input_dict[index]["visual_token_type_ids"],
            "visual_attention_mask": full_input_dict[index]["visual_attention_mask"],
            "labels": full_input_dict[index]["labels"],
        }
    )
# for index in range(train.shape[0]):
#     row = train.iloc[index]
#     training_input_dict.append(
#         {
#             "input_ids": text_embeddings[index].input_ids,
#             "attention_mask": text_embeddings[index].attention_mask,
#             "token_type_ids": text_embeddings[index].token_type_ids,
#             "visual_embeddings": row["annotated_images_embeddings"][0].tolist(),
#             "visual_token_type_ids": row["visual_token_type_ids"][0].tolist(),
#             "visual_attention_mask": row["visual_attention_mask"][0].tolist(),
#             "labels": row["labels"][0].tolist(),
#         }
#     )
validation_input_dict = []
for index in range(val.shape[0]):
    index = index + train.shape[0]
    validation_input_dict.append(
        {
            "input_ids": full_input_dict[index]["input_ids"],
            "token_type_ids": full_input_dict[index]["token_type_ids"],
            "attention_mask": full_input_dict[index]["attention_mask"],
            "visual_embeds": full_input_dict[index]["annotated_images_embeddings"],
            "visual_token_type_ids": full_input_dict[index]["visual_token_type_ids"],
            "visual_attention_mask": full_input_dict[index]["visual_attention_mask"],
            "labels": full_input_dict[index]["labels"],
        }
    )
    # row = val.iloc[index]
    # validation_input_dict.append(
    #     {
    #        "input_ids": text_embeddings[(train.shape[0]+index)].input_ids,
    #         "attention_mask": text_embeddings[(train.shape[0]+index)].attention_mask,
    #         "token_type_ids": text_embeddings[(train.shape[0]+index)].token_type_ids,
    #         "visual_embeddings": row["annotated_images_embeddings"][0].tolist(),
    #         "visual_token_type_ids": row["visual_token_type_ids"][0].tolist(),
    #         "visual_attention_mask": row["visual_attention_mask"][0].tolist(),
    #         "labels": row["labels"][0].tolist(),
    #     }
    # )

test_input_dict = []
for index in range(test.shape[0]):
    index = index + train.shape[0] + val.shape[0]
    test_input_dict.append(
        {
            "input_ids": full_input_dict[index]["input_ids"],
            "token_type_ids": full_input_dict[index]["token_type_ids"],
            "attention_mask": full_input_dict[index]["attention_mask"],
            "visual_embeds": full_input_dict[index]["raw_images_embeddings"],
            "visual_token_type_ids": full_input_dict[index]["visual_token_type_ids"],
            "visual_attention_mask": full_input_dict[index]["visual_attention_mask"],
            "labels": full_input_dict[index]["labels"],
        }
    )
    # row = test.iloc[index]
    # test_input_dict.append(
    # {
    #    "input_ids": text_embeddings[(train.shape[0]+val.shape[0]+index)].input_ids,
    #     "attention_mask": text_embeddings[(train.shape[0]+val.shape[0]+index)].attention_mask,
    #     "token_type_ids": text_embeddings[(train.shape[0]+val.shape[0]+index)].token_type_ids,
    #     "visual_embeddings": row["raw_image_embeddings"][0].tolist(),
    #     "visual_token_type_ids": row["visual_token_type_ids"][0].tolist(),
    #     "visual_attention_mask": row["visual_attention_mask"][0].tolist(),
    #     "labels": row["labels"][0].tolist(),
    # }

## Saving input lists as json
#############################################
# with open(TRAIN_JSON, "w") as f:
#     json.dump(training_input_dict, f)
# with open(VAL_JSON, "w") as f:
#     json.dump(validation_input_dict, f)
# with open(TEST_JSON, "w") as f:
#     json.dump(test_input_dict, f)

# ## Opening the json files
# #############################################
# with open(TRAIN_JSON, "r") as f:
#     training_input_list = json.load(f)
# with open(VAL_JSON, "r") as f:
#     validation_input_list = json.load(f)
# with open(TEST_JSON, "r") as f:
#     test_input_list = json.load(f)
## Creating datasets
#############################################
train_dataset = torch.utils.data.DataLoader(
    training_input_dict,
    batch_size=1,
    shuffle=True,
).dataset
val_dataset = torch.utils.data.DataLoader(
    validation_input_dict, batch_size=1, shuffle=True
).dataset
test_dataset = torch.utils.data.DataLoader(
    test_input_dict, batch_size=1, shuffle=True
).dataset
print("Train dataset length: ", len(train_dataset))
# shapes = [x[0].shape for x in train_dataset[0].values() if isinstance(x, list)]
# print("Train dataset shapes: ", shapes)
first_level =[[(x,y.shape) for y in train_dataset[0][x]] for x in train_dataset[0].keys()]
with open(DATA_DIRECTORY/"first_level.txt", "w") as f:
    f.write(str(first_level))
#%%
## Loading the model
#############################################
model = VisualBertForMultipleChoice.from_pretrained(
    "uclanlp/visualbert-vqa", ignore_mismatched_sizes=True
)


training_args = TrainingArguments(
    output_dir=str(SAVED_MODELS_FOLDER) + "/visualbert",
    do_train=True,
    do_eval=True,
    do_predict=True,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=1e-5,
)
try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=True,
    )
    trainer.train()
    trainer.evaluate()
    trainer.predict(test_dataset)
except Exception as e:
    print(e)
