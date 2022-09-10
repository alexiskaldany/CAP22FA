import os 
import sys
from pathlib import Path

DATA_DIRECTORY = Path(__file__).parent / "data" 
if not DATA_DIRECTORY.exists(): 
    os.makedirs(DATA_DIRECTORY)
download_command = f"aws s3 cp --no-sign-request s3://ai2-public-datasets/diagrams/ai2d-all.zip {DATA_DIRECTORY} \n unzip {DATA_DIRECTORY}/ai2d-all.zip -d {DATA_DIRECTORY} \n rm {DATA_DIRECTORY}/ai2d-all.zip"
os.system(download_command)