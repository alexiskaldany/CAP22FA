from loguru import logger
import sys
logger.remove()
logger.add(
    sys.stdout,
    format="<light-yellow>{time:YYYY-MM-DD HH:mm:ss}</light-yellow> | <light-blue>{level}</light-blue> | <cyan>{message}</cyan> | <light-red>{function}: {line}</light-red>",
    level="DEBUG",
    backtrace=True,
    colorize=True,
)

from src.utils.configs import DATA_DIRECTORY, ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER
from src.utils.prepare_and_download import *
from src.utils.pre_process import *
import click

@click.command()
@click.option('-d','--download', is_flag=True, help='Download the data')
@click.option('-dbg','--debug', is_flag=True, help='Test the data')

def main(download,debug)->None:
    if download:
        download_data(DATA_DIRECTORY)
        return
    if debug:
        data_list = get_data_objects(ANNOTATION_FOLDER,IMAGES_FOLDER,QUESTIONS_FOLDER)
        with(open(DATA_DIRECTORY / "data_list.json","w")) as f:
            json.dump(data_list,f)
        df =create_data_dataframe(data_list)
        df.to_csv(DATA_DIRECTORY / "input_rows.csv",index=False)
        # train, val, test = create_train_val_test_split(data_list)
        return 
    
if __name__ == "__main__":
    main(prog_name="capstone")