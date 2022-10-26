"""
annotation_to_string.py
Creates a string from each annotation json
author: @alexiskaldany, @justjoshtings
created: 10/23/22
"""

from copy import deepcopy
import json
import os
from loguru import logger
import sys
import random
import pandas as pd

# get current directory
path = os.getcwd()
# parent_path = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
# add src to executable path to allow imports from src
sys.path.insert(0, path)
from src.utils.configs import RANDOM_STATE, DATA_DIRECTORY

# print(DATA_DIRECTORY)
# from src.utils.answer_filtering import has_only_one_word_answers

random_state = RANDOM_STATE
object_list = ["connector", "destination", "origin"]
category_list = [
    "arrowDescriptor",  # done
    "intraObjectTextLinkage",  # done
    "arrowHeadTail",  # done
    "intraObjectLinkage",  # done
    "intraObjectLabel",  # done
    "imageTitle",  # done
    "misc",
    "sectionTitle",  # done
    "interObjectLinkage",  # done
    "intraObjectRegionLabel",  #
    "imageCaption",  # done
]

"""
Set logger
"""
# logger.remove()
# logger.add(
#     # "./logs/training_log.txt",
#     sys.stdout,
#     format="{time:YYYY-MM-DD HH:mm:ss}|{level}| {message}|{function}: {line}",
#     level="DEBUG",
#     backtrace=True,
#     colorize=True,
# )

""" 
TODO:
Post processing of the combined string:
1. Remove multiple periods/bad formatting
2. Ensure that the string is not too long (less than 450?)
"""
""" 
Function which reads annotations path and returns dictionary
"""


def read_annotation_json(annotation_path: str) -> dict:
    try:
        with open(annotation_path, "r") as f:
            data = json.load(f)
        return data
    except:
        logger.exception(f"Error reading annotation file: {annotation_path}")


""" 
Function which generates a list of relationships 
"""


def get_relationships(dict: dict) -> list:
    if "relationships" in dict.keys():
        relationship_list = []
        for k, v in dict["relationships"].items():
            relationship_list.append(v)
        return relationship_list
    else:
        return None


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
                    relationship["connector_text"] = relationship["text_dict"][
                        connector
                    ]
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
    if relationship["origin_text"] != "" and relationship["destination_text"] != "":
        return f"{relationship['origin_text']} object links to {relationship['destination_text']}."
    else:
        return None


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
    if relationship["origin_text"] != "":
        return f"{relationship['origin_text']} describes region."
    else:
        None


""" 
A function for intraObjectTextLinkage
"""


def intraObjectTextLinkage_string(relationship: dict) -> str:
    """
    Intra-Object Text Linkage: A text box referring to another text box.

    "intraObjectTextLinkage"
    """
    if relationship["origin_text"] != "" and relationship["destination_text"] != "":
        return f"{relationship['origin_text']} refers to {relationship['destination_text']}."
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
    if relationship["origin_text"] != "" and relationship["category"] == "imageTitle":
        return f"The title of the image is {relationship['origin_text']}."
    elif (
        relationship["origin_text"] != "" and relationship["category"] == "sectionTitle"
    ):
        return f"The title of the section is {relationship['origin_text']}."
    else:
        return None


"""
A function for imageCaption
"""


def imageCaption_string(relationship: dict) -> str:
    """
    Image Caption: A text box that adds information about the entire image, but does not serve as the image title.

    "imageCaption"
    """
    if relationship["origin_text"] != "":
        return f"The image caption is {relationship['origin_text']}."
    else:
        return None


""" 
A function for arrowDescriptor
"""


def arrowDescriptor_string(relationship: dict) -> str:
    """
    Arrow Descriptor: A text box describing a process that an arrow refers to.

    "arrowDescriptor"
    """
    if relationship["origin_text"] != "":
        return f"{relationship['origin_text']} is process described by arrow."
    else:
        return None


""" 
A function for arrowHeadTail
"""


def arrowHeadTail_string(relationship: dict) -> str:
    """
    Arrow Head Assignment: An arrow head associated to an arrow tail.

    "arrowHeadTail"
    """
    if relationship["origin_text"] != "":
        return f"{relationship['origin_text']} is arrow head associated to arrow tail."
    else:
        return None


""" 
A function for misc
"""


def misc_string(relationship: dict) -> str:
    """
    Image Misc: Decorative elements in the diagram.

    "misc"
    """
    if relationship["origin_text"] != "":
        return f"{relationship['origin_text']} is misc information."
    else:
        return None


""" 
Master function which determines `category` of relationship and branches off depending on `category` value
"""


def category_to_string(relationships: list) -> list:
    """
    Input: relationships list,
    updates each relationship dictionary with a string value
    Output: relationships list with string value for each relationship
    """

    for relationship in relationships:
        if relationship["category"] == "interObjectLinkage":
            relationship["string"] = interObjectLinkage_string(relationship)
        elif (
            relationship["category"] == "intraObjectLinkage"
            or relationship["category"] == "intraObjectLabel"
            or relationship["category"] == "intraObjectRegionLabel"
        ):
            relationship["string"] = intraObjectLinkage_string(relationship)
        elif relationship["category"] == "intraObjectTextLinkage":
            relationship["string"] = intraObjectTextLinkage_string(relationship)
        elif (
            relationship["category"] == "imageTitle"
            or relationship["category"] == "sectionTitle"
        ):
            relationship["string"] = Title_string(relationship)
        elif relationship["category"] == "imageCaption":
            relationship["string"] = imageCaption_string(relationship)
        elif relationship["category"] == "arrowDescriptor":
            relationship["string"] = arrowDescriptor_string(relationship)
        elif relationship["category"] == "arrowHeadTail":
            relationship["string"] = arrowHeadTail_string(relationship)
        elif relationship["category"] == "misc":
            relationship["string"] = misc_string(relationship)
        else:
            relationship["string"] = None
    return relationships


""" 
A function with concatenates all relationship strings into one string
"""


def combine_relationship_strings(relationships: list) -> str:
    """
    Input: relationships list,
    Output: string with all relationship strings concatenated
    """
    relationship_strings = [
        relationship["string"]
        for relationship in relationships
        if relationship["string"] != None
    ]
    return " ".join(relationship_strings)


""" 
Create a per annotation_path master function
"""


def get_relationship_string(annotation_path: str) -> str:
    index = int(annotation_path.split("/")[-1].split(".")[0])
    if index % 500 == 0:
        logger.info(index)
        logger.info(annotation_path)
    annotation_dict = read_annotation_json(annotation_path)
    if annotation_dict == None:
        return ""
    relationships = get_relationships(annotation_dict)
    if relationships == None:
        return ""
    text = get_text(annotation_dict)
    relationships_with_text = add_text_values_to_relationships(relationships, text)
    relationships_with_text_and_keys = match_text_to_relationship_keys(
        relationships_with_text
    )
    relationships_with_string = category_to_string(relationships_with_text_and_keys)
    combined_string = combine_relationship_strings(relationships_with_string)
    return combined_string


"""
Create model_training.py level master function
"""


def get_relationship_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: dataframe with annotation_path column
    Output: dataframe with relationship_string column
    """
    df["annotations_path"] = df["image_path"].str.replace("images", "annotations")
    df["annotations_path"] = df["annotations_path"].str.replace(".png", ".png.json")
    logger.debug(f"{df['annotations_path'][:5]}")
    df["relationship_string"] = df["annotations_path"].apply(get_relationship_string)
    return df


""" 
Test 
"""
# combined_list = get_data_objects(ANNOTATION_FOLDER, IMAGES_FOLDER, QUESTIONS_FOLDER)
# data_df = create_dataframe(combined_list)
# data_df['annotated_image_path'] = data_df['image_path'].str.replace('images','annotated_images')
# data_df = get_relationship_strings(data_df)
# data_df['string_length'] = data_df['relationship_string'].str.len()
# data_df.to_csv(str(DATA_DIRECTORY)+"/data_df_test.csv", index=False)
# data_df_test = data_df.sample(1000, random_state=random_state)
# strings = []
# for index, row in data_df_test.iterrows():
#     annotation_path = row["annotations_path"]
#     strings.append(get_relationship_string(annotation_path))
# data_df_test["relationship_string"] = strings
# data_df_test.to_csv(str(DATA_DIRECTORY)+"/data_df_test.csv", index=False)
# relationships_with_text = get_annotations_dict(annotation_path)
# [print(relationship) for relationship in relationships_with_text if relationship['category'] == 'misc']
# categories = [relationship["category"] for relationship in relationships_with_text]
# category.extend(categories)
# from collections import Counter
# counts = Counter(category)
# print(counts)
# print(len(category))
# category = set(category)
# print(category)
