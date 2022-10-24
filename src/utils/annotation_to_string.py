from copy import deepcopy
import json
import os
from loguru import logger
import sys
import random

# get current directory
path = os.getcwd()
# parent_path = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
# add src to executable path to allow imports from src
sys.path.insert(0, path)

from src.utils.configs import (
    DATA_JSON,
    DATA_CSV,
    DATA_DIRECTORY,
    ANNOTATION_FOLDER,
    IMAGES_FOLDER,
    QUESTIONS_FOLDER,
    ANNOTATED_IMAGES_FOLDER,
    TEST_DIRECTORY,
    TEST_IMAGE_OUTPUT,
)
from src.utils.prepare_and_download import get_data_objects, create_dataframe
from src.utils.applying_annotations import execute_full_set_annotation
from src.utils.visual_embeddings import get_multiple_embeddings
from src.utils.pre_process import create_train_val_test_split
from src.utils.configs import RANDOM_STATE

# from src.utils.answer_filtering import has_only_one_word_answers

random_state = RANDOM_STATE
object_list = ["connector", "destination", "origin"]
category_list = [
    "arrowDescriptor",
    "intraObjectTextLinkage", # done
    "arrowHeadTail",
    "intraObjectLinkage", # done
    "intraObjectLabel", # done
    "imageTitle", # done
    "misc",
    "sectionTitle", # done
    "interObjectLinkage", # done
    "intraObjectRegionLabel", # 
    "imageCaption",
]

"""
Set logger
"""
logger.remove()
logger.add(
    "./logs/training_log.txt",
    # sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss}|{level}| {message}|{function}: {line}",
    level="INFO",
    backtrace=True,
    colorize=True,
)

""" 
1. Each relationship will be its own string.
2. Syntactic structure of that string will depoend on `category` of relationship.

"""


"""
Load data
"""
combined_list = get_data_objects(ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER)
data_df = create_dataframe(combined_list)
data_df["annotations_path"] = data_df["image_path"].str.replace("images", "annotations")
data_df["annotations_path"] = data_df["annotations_path"].str.replace(
    ".png", ".png.json"
)

""" 
Function which reads annotations path and returns dictionary
"""


def read_annotation_json(annotation_path: str) -> dict:
    with open(annotation_path) as f:
        data = json.load(f)
    return data


""" 
Function which generates a list of relationships 
"""


def get_relationships(dict: dict) -> list:
    relationships = dict["relationships"]
    relationship_list = []
    for k, v in relationships.items():
        relationship_list.append(v)
    return relationship_list


""" Create list of text labels"""


def get_text(dict: dict) -> dict:
    text_labels = {}
    for k, v in dict["text"].items():
        text_labels[k] = v["value"]
    return text_labels


""" 
Add text label to any object within a relationship
"""


def add_text_values_to_relationships(relationships: list, text: dict) -> dict:
    label_ids = text.keys()
    relationship_copy = relationships.copy()
    for index, relationship in enumerate(list(relationships.copy())):
        text_dict = {}
        for rk, rv in relationship.copy().items():
            if rv in label_ids:
                text_dict[rv] = text[rv]
                # relationship_copy[index][rv] = text[rv]
        relationship_copy[index]["text_dict"] = text_dict
    return relationships

""" 
Matching text_dict to relationship keys
"""

def match_text_to_relationship_keys(relationships: list) -> list:
    for relationship in relationships:
        if "origin" in relationship:
            if "text_dict" in relationship:
                if relationship["origin"] in relationship["text_dict"].keys():
                    relationship["origin_text"] = relationship["text_dict"][
                        relationship["origin"]
                    ]
                else:
                    relationship["origin_text"] = ""
        if "destination" in relationship:
            destination = relationship["destination"]
            if "text_dict" in relationship:
                if destination in relationship["text_dict"]:
                    relationship["destination_text"] = relationship["text_dict"][
                        destination
                    ]
                else:
                    relationship["destination_text"] = ""
        if "connector" in relationship:
            connector = relationship["connector"]
            if "text_dict" in relationship:
                if connector in relationship["text_dict"]:
                    relationship["connector_text"] = relationship["text_dict"][connector]
                else:
                    relationship["connector_text"] = ""
    return relationships

""" 
A function for inter-object relationships
"""


def interObjectLinkage_string(relationship: dict) -> str:
    """
    interObjectLinkage: Two objects related to one another via an arrow.
    Example, if origin_text exists: f"{origin} object ({origin_text}) links to {destination} ({destination_text}) by connector {connector} ({connector_text})"
    
    "interObjectLinkage"
    """
    if relationship['origin_text'] != '' and relationship['destination_text'] != '':
        return f"{relationship['origin_text']} object links to {relationship['destination_text']} "
    else:
        return None

    # if "connector" in relationship:
    #     return f"{relationship['origin']} object ({relationship['origin_text']}) links to {relationship['destination']} ({relationship['destination_text']}) by connector {relationship['connector']} ({relationship['connector_text']})"
    # else:
    #     return f"{relationship['origin']} object ({relationship['origin_text']}) links to {relationship['destination']} ({relationship['destination_text']})"


""" 
A function for intra-object relationships
"""

def intraObjectLinkage_string(relationship: dict) -> str:
    """
    Intra-Object Linkage : A text box referring to a region within an object via an arrow.
    Intra-Object Label: A text box naming the entire object.
    Intra-Object Region Label: A text box referring to a region within an object.
    
    "intraObjectRegionLabel"
    "intraObjectLinkage"
    "intraObjectLabel"
    """
    # if "connector" in relationship:
    #     return f"({relationship['origin_text']}) describes region. to {relationship['destination']} ({relationship['destination_text']}) by connector {relationship['connector']} ({relationship['connector_text']})"    
    # else:
    #     return f"({relationship['origin_text']}) describes region to {relationship['destination']} ({relationship['destination_text']})"    
    if relationship['origin_text'] != '':
        return f"{relationship['origin_text']} describes region."
    else:
        None

""" 
A function for intra-object labels
"""

# def intraObjectLabel_string(relationship: dict) -> str:
#     """
#     Intra-Object Label: A text box naming the entire object.
#     Intra-Object Region Label: A text box referring to a region within an object.
#     """
#     if relationship["category"] == "intraObjectLabel":
#         return f"object is labelled as {relationship['origin_text']}" if relationship["origin_text"]!= '' else None
#     else: # intraObjectRegionLabel
#         return f"({relationship['origin_text']}) describes region."
""" 
A function for intraObjectTextLinkage
"""

def intraObjectTextLinkage_string(relationship: dict) -> str:
    """
    Intra-Object Text Linkage: A text box referring to another text box.
    
    "intraObjectTextLinkage"
    """
    if relationship['origin_text'] != '' and relationship['destination_text'] != '':
        return f"{relationship['origin_text']} refers to {relationship['destination_text']}"
    else:
        return None

""" 
A function for imageTitle and sectionTitle
"""

def Title_string(relationship: dict) -> str:
    """
    Image Title : The title of the entire image.
    Image Section Title : Text box that serves as a title for a section of the image.
    
    "imageTitle"
    "sectionTitle"
    """
    if relationship['origin_text'] != '' and relationship['category'] == 'imageTitle':
        return f"The title of the image is {relationship['origin_text']}"
    elif relationship['origin_text'] != '' and relationship['category'] == 'sectionTitle':
        return f"The title of the section is {relationship['origin_text']}"
    else:
        return None
""" 
Master function which determines `category` of relationship and branches off depending on `category` value
"""

def category_to_string(relationships: list) -> list:
    relationships_copy = relationships.copy()
    for relationship in relationships:
        category = relationship["category"]
    pass


""" 
Create a per annotation_path master function
"""


def get_annotations_dict(annotation_path: str) -> str:
    constructed_dict = {}
    annotation_dict = read_annotation_json(annotation_path)
    relationships = get_relationships(annotation_dict)
    text = get_text(annotation_dict)
    relationships_with_text = add_text_values_to_relationships(relationships, text)
    relationships_with_text_and_keys = match_text_to_relationship_keys(
        relationships_with_text
    )
    pass


""" 
Test 
"""

data_df_test = data_df.sample(1000, random_state=random_state)
category = []
for index, row in data_df_test.iterrows():
    annotation_path = row["annotations_path"]
    relationships_with_text = get_annotations_dict(annotation_path)
    # [print(relationship) for relationship in relationships_with_text if relationship['category'] == 'sectionTitle' and relationship != None]
    # categories = [relationship["category"] for relationship in relationships_with_text]
    # category.extend(categories)
# from collections import Counter
# counts = Counter(category)
# print(counts)
# print(len(category))
# category = set(category)
# print(category)
