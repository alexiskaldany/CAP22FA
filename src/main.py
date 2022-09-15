from loguru import logger
import sys
logger.remove()
logger.add(
    sys.stdout,
    format="<light-yellow>{time:YYYY-MM-DD HH:mm:ss}</light-yellow> | <light-blue>{level}</light-blue> | <cyan>{message}</cyan> | <light-red>{function}: {line}</light-red>",
    level="INFO",
    backtrace=True,
    colorize=True,
)

from src.utils.configs import DATA_DIRECTORY, ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER
from src.utils.prepare_and_download import *
from src.utils.pre_process import *
from src.utils.applying_annotations import execute_full_set_annotation
from src.utils.configs import DATA_JSON,DATA_CSV,DATA_DIRECTORY,ANNOTATION_FOLDER,IMAGES_FOLDER,QUESTIONS_FOLDER,ANNOTATED_IMAGES_FOLDER
from src.
import click

@click.command()
@click.option('-d','--download', is_flag=True, help='Download the data')
@click.option('-crd','--create_data', is_flag=True, help='Creates the combined data files')

def main(download:bool,create_data:bool)->None:
    if download:
        download_data(DATA_DIRECTORY)
        return
    if create_data:
        data_list = get_data_objects(ANNOTATION_FOLDER,IMAGES_FOLDER,QUESTIONS_FOLDER)
        with(open(DATA_JSON,"w")) as f:
            json.dump(data_list,f)
        dataframe = create_dataframe(data_list)
        execute_full_set_annotation(DATA_JSON,ANNOTATED_IMAGES_FOLDER)
        dataframe["annotated_image_paths"] = dataframe["image_path"].apply(lambda x: ANNOTATED_IMAGES_FOLDER / Path(x).name)
        annotated_images_embeddings = get_embeddings(dataframe["annotated_image_paths"])
        dataframe.to_csv(DATA_CSV,index=False)
        # train, val, test = create_train_val_test_split(data_list)
        return 
    
if __name__ == "__main__":
    main(prog_name="capstone")