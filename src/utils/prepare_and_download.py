import os 
import sys
from pathlib import Path
from loguru import logger
import glob
import json
import pandas as pd
# def checking_folders(path_dict:dict):
#     if not path_dict["DATA_DIRECTORY"].exists(): 
#         os.makedirs(DATA_DIRECTORY)
#     return True

def download_data(DATA_DIRECTORY:Path):
    try:
        logger.info("Downloading the data")
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
    
def get_data_objects(ANNOTATION_FOLDER,IMAGES_FOLDER,QUESTIONS_FOLDER):
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
    print(f"Number of images: {len(image_ids)}")
    combined_list = []
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
        combined_list.append(temp_list)   
    return combined_list

def create_dataframe(data_list:list):
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