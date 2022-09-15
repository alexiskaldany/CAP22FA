from pathlib import Path

# Internal Directories and Folders
DATA_DIRECTORY = Path(__file__).parent.parent / "data"
ANNOTATION_FOLDER = DATA_DIRECTORY / "ai2d" / "annotations"
IMAGES_FOLDER = DATA_DIRECTORY / "ai2d" / "images"
QUESTIONS_FOLDER = DATA_DIRECTORY / "ai2d" / "questions"
ANNOTATED_IMAGES_FOLDER = DATA_DIRECTORY / "ai2d" / "annotated_images"
RUNS_FOLDER = Path(__file__).parent.parent / "runs"

# Data File Paths

DATA_JSON = DATA_DIRECTORY / "data_set.json"
DATA_CSV = DATA_DIRECTORY / "data.csv"
# Constants
IMAGE_DIMENSIONS = (620, 480)
ANNOTATION_THICKNESS = int(2)