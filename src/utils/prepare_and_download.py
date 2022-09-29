import os 
import sys
from pathlib import Path
from loguru import logger
import glob
import json
import pandas as pd
from transformers import DataCollatorWithPadding
import random
def download_data(DATA_DIRECTORY:Path,ANNOTATED_IMAGES_FOLDER:Path):
    """
    Creates needed folders then downloads, unzips and deletes the zip file
    """
    try:
        logger.info("Downloading the data")
        if not ANNOTATED_IMAGES_FOLDER.exists() :
            os.makedirs(ANNOTATED_IMAGES_FOLDER)
        if not DATA_DIRECTORY.exists(): 
            os.makedirs(DATA_DIRECTORY)
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
    
def get_data_objects(ANNOTATION_FOLDER,IMAGES_FOLDER,QUESTIONS_FOLDER) -> list:
    """ 
    Takes in the annotation, images and questions folder and returns a list of dictionaries
    1 dictionary per image ID
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
            logger.exception(f"Error in loading the question file: {question_path_dict[id]}")
        temp_list = [image_dict, annotation_dict, question_dict]
        data_list.append(temp_list)   
    return data_list

def create_row_per_question_dataframe(data_list:list) -> pd.DataFrame:
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
            answer = list_of_answers[data[2]["questions"][question_key]["correctAnswer"]]
            abcLabel = data[2]["questions"][question_key]["abcLabel"]
            question_dict = {"question": question, "list_of_answers":list_of_answers,"answer": answer, "abcLabel": abcLabel}
            temp_dict = {**img_dict, **question_dict}
            list_of_dicts.append(temp_dict)
    df = pd.DataFrame(list_of_dicts)
    return df

def create_train_val_test_split(dataframe:pd.DataFrame,TRAIN_SPLIT:float,VAL_SPLIT:float,TEST_SPLIT:float) -> tuple:
    logger.info("Creating train, val, test split")
    train_index = int(0.7 * dataframe.shape[0])
    val_index = int(0.9 * dataframe.shape[0])
    random.shuffle(dataframe)
    train = dataframe[:train_index]
    val = dataframe[train_index:val_index]
    test = dataframe[val_index:]
    logger.info(f"Train: {len(train)}")
    logger.info(f"Val: {len(val)}")
    logger.info(f"Test: {len(test)}")
    return train,val,test

