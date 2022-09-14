from pathlib import Path

# Internal Paths
DATA_DIRECTORY = Path(__file__).parent.parent / "data"
ANNOTATION_FOLDER = DATA_DIRECTORY / "ai2d" / "annotations"
IMAGES_FOLDER = DATA_DIRECTORY / "ai2d" / "images"
QUESTIONS_FOLDER = DATA_DIRECTORY / "ai2d" / "questions"

# Constants

IMAGE_DIMENSIONS = (620, 480)