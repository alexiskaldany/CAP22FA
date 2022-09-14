import os 
import sys
from pathlib import Path
from loguru import logger

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
        os.system(f"unzip {DATA_DIRECTORY}/ai2d-all.zip")
        logger.info("Completed unzipping the data")
        logger.info("Deleting the zip file")
        os.system(f"rm {DATA_DIRECTORY}/ai2d-all.zip")
        print("Download and cleanup complete")
        return True
    except Exception as e:
        logger.error(f"Error in downloading the data: {e}")
        return False
    
