"""
configs.py
Configuration file for program such as:
    - Internal Directories and Folders
    - Data File Paths
    - Constants
author: @alexiskaldany, @justjoshtings
created: 9/23/22
"""

from pathlib import Path
import os

# Internal Directories and Folders
DATA_DIRECTORY = Path(__file__).parent.parent / "data"
ANNOTATION_FOLDER = DATA_DIRECTORY / "ai2d" / "annotations"
IMAGES_FOLDER = DATA_DIRECTORY / "ai2d" / "images"
QUESTIONS_FOLDER = DATA_DIRECTORY / "ai2d" / "questions"
ANNOTATED_IMAGES_FOLDER = DATA_DIRECTORY / "ai2d" / "annotated_images"
MODEL_FOLDER = DATA_DIRECTORY.parent / "models"
RESULTS_FOLDER = DATA_DIRECTORY.parent / "results" / "model_weights" 
# Don't think this is supposed to run here? commenting it out
# if ANNOTATED_IMAGES_FOLDER.exists() == False:
#     os.makedirs(ANNOTATED_IMAGES_FOLDER)
RUNS_FOLDER = Path(__file__).parent.parent / "runs"

TEST_DIRECTORY = Path(__file__).parent.parent / "tests"
TEST_IMAGE_OUTPUT = TEST_DIRECTORY / "image_tests"

# Data File Paths
DATA_JSON = DATA_DIRECTORY / "data_set.json"
DATA_CSV = DATA_DIRECTORY / "data.csv"

# Constants
IMAGE_DIMENSIONS = (620, 480)
ANNOTATION_THICKNESS = int(2)

# Seed
RANDOM_STATE = 42

