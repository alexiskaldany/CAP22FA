"""
Main.py
Main script to execute for project, executes all functions needed to perform project.
author: @alexiskaldany, @justjoshtings
created: 9/23/22
"""

'''
Import and setup
'''
from genericpath import exists
from loguru import logger
import sys
import os
import json
import torch
import click

# get current directory
path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
# add src to executable path to allow imports from src
sys.path.insert(0, parent_path)

from src.utils.configs import DATA_JSON, DATA_CSV, DATA_DIRECTORY, ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER, ANNOTATED_IMAGES_FOLDER
from src.utils.prepare_and_download import download_data, get_data_objects, create_dataframe
from src.utils.pre_process import load_image_resize_convert, get_image_features, create_train_val_test_split
from src.utils.applying_annotations import execute_full_set_annotation
from src.utils.visual_embeddings import get_multiple_embeddings

logger.remove()
logger.add(
    sys.stdout,
    format="<light-yellow>{time:YYYY-MM-DD HH:mm:ss}</light-yellow> | <light-blue>{level}</light-blue> | <cyan>{message}</cyan> | <light-red>{function}: {line}</light-red>",
    level="INFO",
    backtrace=True,
    colorize=True,
)

'''
Main Function
'''
@click.command()
@click.option("-d", "--download", is_flag=True, help="Download the data")
@click.option(
    "-data", "--create_data", is_flag=True, help="Creates the combined data files"
)
def main(download: bool, create_data: bool) -> None:
    if torch.cuda.is_available():
        logger.info("GPU Available")
    if download:
        # If folder empty then download otherwise already has data and don't need to duplicate/replace
        if DATA_DIRECTORY.exists() == False:
            download_data(DATA_DIRECTORY, ANNOTATED_IMAGES_FOLDER)
        else:
            # If directory exists, check if empty, if empty, download otherwise skip
            if not os.listdir(DATA_DIRECTORY):
                download_data(DATA_DIRECTORY, ANNOTATED_IMAGES_FOLDER)
        return
    if create_data:
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
        return


if __name__ == "__main__":
    print("Executing main.py, capstone")
    main(prog_name="capstone")
    
