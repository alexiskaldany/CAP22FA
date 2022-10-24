"""
prepare_and_download.py
Utility functions to download data and handle data preparation
author: @alexiskaldany, @justjoshtings
created: 9/23/22
"""

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

def download_data(DATA_DIRECTORY:Path,ANNOTATED_IMAGES_FOLDER:Path):
    '''
    Downloads data from https://registry.opendata.aws/allenai-diagrams/ into ./CAP22FA/src/data/ directory
    
    Dataset includes:
       1. README.txt: this file

        2. license.txt: the license file

        3. images/
        PNG image files

        4. annotations/
        Annotation files

        5. questions/
        Questions that are associated with images. Some images do not have a question.

        6. categories.json
        A category tag per image. For example: lifeCycles/moonPhaseEquinox/rockCycle/volcano/etc

    Params:
        DATA_DIRECTORY (Path): filepath to save data download
        ANNOTATED_IMAGES_FOLDER (Path): filepath to save annotated images
    '''
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
    
def get_data_objects(ANNOTATION_FOLDER,IMAGES_FOLDER,QUESTIONS_FOLDER):
    '''
    Creates a list where each element is an instance of data.

    List includes dictionaries for:
        image path: {'image_path': '/home/ubuntu/capstone/CAP22FA/src/data/ai2d/images/0.png'}
        annotations: {'arrowHeads': {}, 'arrows': {'A0': {'id': 'A0', 'polygon': [[167, 64], [190, 57], ...
        questions and ansers: {'abcLabel': True, 'answerTexts': ['ears', 'nose', 'mouth', 'face'], 'correctAnswer': 3, 'questionId': '0.png-0'},...

    Params:
        DATA_DIRECTORY (Path): filepath to save data download
        ANNOTATED_IMAGES_FOLDER (Path): filepath to save annotated images

    Returns:
        combined_list (List): list of list of dictionaries where each element is an instance of data
    '''
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
    combined_list = []
    for index,id in enumerate(id_list):
        if index % 1000 == 0:
            logger.info(f"Processed {index} IDs")
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
    '''
    Creates a DF of data with image path, annotations, questions and answers

    Params:
        data_list (list): list of list of dictionaries where each element is an instance of data

    Returns:
        df (Pandas df): dataframe of data with image path, annotations, questions and answers
    '''
    list_of_dicts = []
    for index,data in enumerate(data_list):
        if index % 1000 == 0:
            logger.info(f"Processing image number: {index}")
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


if __name__ == "__main__":
    print("Executing prepare_and_download.py")

    # get current directory
    path = os.getcwd()
    # parent_path = os.path.abspath(os.path.join(path, os.pardir, os.pardir))

    # add src to executable path to allow imports from src
    sys.path.insert(0, path)

    from src.utils.configs import DATA_JSON, DATA_CSV, DATA_DIRECTORY, ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER, ANNOTATED_IMAGES_FOLDER
    from src.utils.applying_annotations import execute_full_set_annotation
    from src.utils.visual_embeddings import get_multiple_embeddings

    # If folder empty then download otherwise already has data and don't need to duplicate/replace
    if DATA_DIRECTORY.exists() == False:
        download_data(DATA_DIRECTORY=DATA_DIRECTORY, ANNOTATED_IMAGES_FOLDER=ANNOTATION_FOLDER)
    else:
        # If directory exists, check if empty, if empty, download otherwise skip
        if not os.listdir(DATA_DIRECTORY):
            download_data(DATA_DIRECTORY=DATA_DIRECTORY, ANNOTATED_IMAGES_FOLDER=ANNOTATION_FOLDER)
    
    combined_list = get_data_objects(ANNOTATION_FOLDER=ANNOTATION_FOLDER,IMAGES_FOLDER=IMAGES_FOLDER,QUESTIONS_FOLDER=QUESTIONS_FOLDER)
    # print(combined_list[0])
    data_df = create_dataframe(combined_list)
    # print(data_df.head())
    # print(data_df.columns)
    print(data_df.head(3)['image_path'])
    print(data_df.head(3)['question'])
    print(data_df.head(3)['answer'])
    print(data_df.head(3)['list_of_answers'])
    print(data_df.head(3)['abcLabel'])
    print(data_df[data_df['image_path']=='/home/ubuntu/capstone/CAP22FA/src/data/ai2d/images/1.png'])

    logger.info(f"Creating data_json")
    data_list = get_data_objects(ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER)
    with (open(DATA_JSON, "w")) as f:
        json.dump(data_list, f)
    dataframe = create_dataframe(data_list)
    logger.info("Starting annotations")
    execute_full_set_annotation(DATA_JSON, ANNOTATED_IMAGES_FOLDER)
    id_list = list(dataframe["image_id"])
    annotated_image_path = [str(ANNOTATED_IMAGES_FOLDER / f"{id}.png") for id in id_list]
    dataframe["annotated_image_path"] = annotated_image_path
    dataframe.to_csv(DATA_CSV, index=False)
    logger.info("Getting annotated images embeddings")
    annotated_list = []
    raw_list = []
    try:
        annotated_images_embeddings = get_multiple_embeddings(list(dataframe["annotated_image_path"]))
    except:
        logger.exception("Error getting embeddings")
    # dataframe["annotated_image_embeds"] = dataframe.apply(
    #     lambda x: annotated_images_embeddings[x["image_id"]]
    #     if annotated_images_embeddings[x["image_id"]]
    #     else []
    # )

    try:
        raw_images_embeddings = get_multiple_embeddings(list(dataframe["image_path"]))
    except:
        logger.exception("Error getting embeddings")
    for index, row in dataframe.iterrows():
        if index % 100:
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
    if dataframe.shape[0] == len(annotated_list):
        dataframe["annotated_images_embeddings"] = annotated_list
    if dataframe.shape[0] == len(raw_list):
        dataframe["raw_image_embeddings"] = raw_list      
    dataframe.to_csv(DATA_CSV, index=False)