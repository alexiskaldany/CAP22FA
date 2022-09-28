from pathlib import Path
import os
# Internal Directories and Folders
DATA_DIRECTORY = Path(__file__).parent.parent / "data"
ANNOTATION_FOLDER = DATA_DIRECTORY / "ai2d" / "annotations"
IMAGES_FOLDER = DATA_DIRECTORY / "ai2d" / "images"
QUESTIONS_FOLDER = DATA_DIRECTORY / "ai2d" / "questions"
ANNOTATED_IMAGES_FOLDER = DATA_DIRECTORY / "ai2d" / "annotated_images"
if ANNOTATED_IMAGES_FOLDER.exists() == False:
    os.mkdir(ANNOTATED_IMAGES_FOLDER)
RUNS_FOLDER = Path(__file__).parent.parent / "runs"
SAVED_MODELS_FOLDER = Path(__file__).parent.parent / "models/saved_models"

# Data File Paths

DATA_JSON = DATA_DIRECTORY / "data_set.json"
DATA_CSV = DATA_DIRECTORY / "data.csv"
# Constants
IMAGE_DIMENSIONS = (620, 480)
ANNOTATION_THICKNESS = int(2)

