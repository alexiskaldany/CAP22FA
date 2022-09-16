from src.utils.configs import DATA_DIRECTORY, ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER
from src.utils.prepare_and_download import *
from src.utils.pre_process import *
from src.utils.applying_annotations import execute_full_set_annotation
from src.utils.configs import DATA_JSON,DATA_CSV,DATA_DIRECTORY,ANNOTATION_FOLDER,IMAGES_FOLDER,QUESTIONS_FOLDER,ANNOTATED_IMAGES_FOLDER
from src.utils.visual_embeddings import get_multiple_embeddings


data_list = get_data_objects(ANNOTATION_FOLDER,IMAGES_FOLDER,QUESTIONS_FOLDER)
dataframe = create_dataframe(data_list)
logger.info("Starting annotations")
# execute_full_set_annotation(DATA_JSON,ANNOTATED_IMAGES_FOLDER)
id_list = list(dataframe["image_id"])
annotated_image_path= [str(ANNOTATED_IMAGES_FOLDER / f"{id}.png") for id in id_list]

# print([type(Path(image).name) for image in id_list[:25]])
# print(annotated_image_path[:25])

# print(len(id_list))
output=get_multiple_embeddings(annotated_image_path[:25])
print(type(output['5']))