"""
applying_annotations.py
Utility functions to apply annotations onto image data. Draws labels, arrows, arrow heads, and blobs.
Allows for drawing full set of annotations or just a part.
author: @alexiskaldany, @justjoshtings
created: 9/23/22
"""

import cv2
import matplotlib.pyplot as plt
import json
from typing import Tuple
import numpy as np
import sys
from loguru import logger
import math
from pathlib import Path
from src.utils.configs import ANNOTATION_THICKNESS
import os


def drawing_labels(Img, label_dict: dict, color: Tuple[int, int, int]):
    label_id = label_dict["id"]
    coordinates_list = label_dict["rectangle"]
    replacement_text = label_dict["replacementText"]
    value = label_dict["value"]
    Img = cv2.rectangle(
        img=Img,
        pt1=coordinates_list[0],
        pt2=coordinates_list[1],
        color=color,
        thickness=1,
    )
    Img = cv2.putText(
        img=Img,
        text=value,
        org=coordinates_list[0],
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=color,
        thickness=ANNOTATION_THICKNESS,
    )
    return Img


def drawing_arrows(Img, coordinate_list, color: Tuple[int, int, int]):
    Img = cv2.polylines(
        Img,
        pts=[np.array(coordinate_list)],
        isClosed=False,
        color=color,
        thickness=ANNOTATION_THICKNESS,
    )
    return Img


def drawing_arrow_heads(
    Img, coordinate_list: list, color: Tuple[int, int, int], angle: float
):
    """
    pt1 = (x1,y1)
    pt2 = (x2,y2)
    pt3 = (? , ?)
    angle1_2_3 == angle2_1_3 == orientation/2
    side1_2 = ((x1-x2)**2 + (y1-y2)**2)**0.5
    side2_3 = side1_2 * tan(angle1_2_3)
    side_1_3 = ((x1-x3)**2 + (y1-y3)**2)**0.5

    m1 = np.arctan()
    Bleh trig...
    Lets just draw a circle using the two points

    midpoint = [(x1+x2)/2,(y1+y2)/2]
    radius = pt1[0]-pt2[0]

    Look at 23.png for seeing how to evaluate arrows
    """
    Center_point = [
        int((coordinate_list[0][0] + coordinate_list[1][0]) / 2),
        int((coordinate_list[0][1] + coordinate_list[1][1]) / 2),
    ]
    Radius = int(
        math.sqrt(
            (coordinate_list[0][0] - coordinate_list[1][0]) ** 2
            + (coordinate_list[0][1] - coordinate_list[1][1]) ** 2
        )
    )
    logger.debug(f"{Center_point,Radius}")
    Img = cv2.circle(
        img=Img,
        center=Center_point,
        radius=Radius,
        color=color,
        thickness=ANNOTATION_THICKNESS,
    )
    return Img


def drawing_blobs(Img, coordinate_list: list, color: Tuple[int, int, int]):
    Img = cv2.polylines(
        Img, pts=[np.array(coordinate_list)], isClosed=True, color=color, thickness=1
    )
    return Img


def generate_random_color_tupple() -> Tuple[int, int, int]:
    return (
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(0, 255),
    )


def full_annotation(annotation_dict: dict, image_path: str, annotated_image_path: str):
    Img_list = [cv2.imread(image_path)]
    label_number = len([annotation_dict["text"].keys()])
    logger.debug(f"Drawing labels: {label_number} ")
    for key in list(annotation_dict["text"].keys()):
        logger.debug(f"Drawing label: {annotation_dict['text'][key]} ")
        Img_list.append(
            drawing_labels(
                Img_list[-1],
                annotation_dict["text"][key],
                generate_random_color_tupple(),
            )
        )
    arrow_number = len(list(annotation_dict["arrows"].keys()))
    logger.debug(f"Drawing arrows:{arrow_number}")
    for key in list(annotation_dict["arrows"].keys()):
        Img_list.append(
            drawing_arrows(
                Img_list[-1],
                annotation_dict["arrows"][key]["polygon"],
                generate_random_color_tupple(),
            )
        )
    arrowhead_number = len(list(annotation_dict["arrowHeads"].keys()))
    logger.debug(f"Drawing arrow heads:{arrowhead_number}")
    for key in list(annotation_dict["arrowHeads"].keys()):
        Img_list.append(
            drawing_arrow_heads(
                Img_list[-1],
                annotation_dict["arrowHeads"][key]["rectangle"],
                annotation_dict["arrowHeads"][key]["orientation"],
                generate_random_color_tupple(),
            )
        )
    logger.debug(f"Drawing blobs:{arrow_number}")
    for key in list(annotation_dict["blobs"].keys()):
        Img_list.append(
            drawing_blobs(
                Img_list[-1],
                annotation_dict["blobs"][key]["polygon"],
                generate_random_color_tupple(),
            )
        )
    cv2.imwrite(annotated_image_path, Img_list[-1])
    return annotated_image_path


def execute_full_set_annotation(DATA_LIST_PATH: Path, ANNOTATED_IMAGES_FOLDER: Path):
    if not ANNOTATED_IMAGES_FOLDER.exists():
        os.makedirs(ANNOTATED_IMAGES_FOLDER)
    full_dict = json.load(open(DATA_LIST_PATH))
    annotated_image_paths = [
        full_annotation(
            full_dict[x][1],
            full_dict[x][0]["image_path"],
            f"{ANNOTATED_IMAGES_FOLDER}/{full_dict[x][0]['image_path'].split('/')[-1].split('.')[0]}.png",
        )
        for x in range(len(full_dict))
    ]
    return annotated_image_paths
