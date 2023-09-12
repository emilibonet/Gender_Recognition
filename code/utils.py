import os
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

root = os.getcwd().split("cv-gender-age-recognition")[0]+"cv-gender-age-recognition/"

# Paths to datasets
data = root + "data/"
MMFashion = root + "data/mmfashion_sample_images/"

# Paths to model checkpoints
chkpts_face_recognition = root + "trained_models/face/face_recognition"
chkpts_deepface = root + "trained_models/face/deepface"
chkpts_FRCNN = root + "trained_models/body/torchvision_FRCNN_ResNet50/"


# Paths to results
results_face_recognition = root + "trained_models/face/results_face_recognition"
results_deepface = root + "trained_models/face/results_deepface"
results_FRCNN = root + "trained_models/body/results_torchvision_FRCNN_ResNet50/"


def path2idx(path):
    """Given the path of a frame, extracts its frame ID.

    ---
    Arguments:
        - path: string of the frame's path
    
    Returns:
        integer of the frame's ID.
    """
    return int(path.split("/")[-1].strip(".jpg"))


# Annotation tool

def add_text(img, text, font_scale, font_thickness, font_color, bg_color, pos=None):
    """Adds text to the input image.

    ---
    Arguments:
        - img: ndarray representing the image that needs to be annotated.
        - text: string with the text that needs to be drawn.
        - font_scale: integer or float representing the text's size.
        - font_thickness: integer or float representing the text's thickness.
        - font_color: integer or float representing the text's color.
        - bg_color: integer or float representing the background's color.
        - pos: tuple, list or ndarray of integers representing the top left position of the text.
    
    Returns:
        ndarray representing the input image with the text drawn on it.
    """
    font = cv2.FONT_HERSHEY_DUPLEX
    (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    if pos:    
        cv2.rectangle(img, (pos[0], pos[1] - h), (pos[0] + w + 1, pos[1] + 1), bg_color, -1)
        cv2.putText(img, text, (pos[0], pos[1] + font_scale - 1), font, font_scale, font_color, font_thickness)
    else:
        cv2.rectangle(img, (0, 0), (w, h), bg_color, -1)
        cv2.putText(img, text, (0, h + font_scale - 1), font, font_scale, font_color, font_thickness)
    return


def gender_color(ann_dict):
    """Returns the defined color for each gender.

    ---
    Arguments:
        - ann_dict: dictionary with the annotations for gender.
    
    Returns:
        tuple representing the color of the gender from the annotation in bgr format.
    """
    if ann_dict['gender'][0] == 'M':
        return (212, 161, 95)
    if ann_dict['gender'][0] == 'W':
        return (212, 95, 146)
    return (132, 252, 3)


def draw_single_annotation(cv2_img, ann_dict, color=None):
    if 'gender' in ann_dict.keys() and ann_dict['gender'] and 'age' in ann_dict.keys() and ann_dict['age']:
        if color is None:
            color = gender_color(ann_dict)
            add_text(cv2_img, f"{ann_dict['age']}", pos=(ann_dict['bbox'][0], ann_dict['bbox'][1]-1),
                    font_scale=1, font_thickness=1, font_color=(255, 255, 255), bg_color=color)
        else:
            add_text(cv2_img, f"{ann_dict['gender'][0]},{ann_dict['age']}", pos=(ann_dict['bbox'][0], ann_dict['bbox'][1]-1),
                    font_scale=1, font_thickness=1, font_color=(255, 255, 255), bg_color=color)
    cv2.rectangle(cv2_img, (ann_dict['bbox'][0], ann_dict['bbox'][1]), (ann_dict['bbox'][2], ann_dict['bbox'][3]), color, thickness=2, lineType=cv2.LINE_8)
    return cv2_img


def draw_annotation(img, ann, color=None):
    """Draws the face and body detections and gender and age labelling annotations on the image.

    ---
    Arguments:
        - img: ndarray representing the image that needs to be annotated, or string of the path leading to the image.
        - ann: dictionary or list of dictionaries consisting of bounding boxes
        - color: tuple of three integers representing the color of the annotations in bgr format, or string 'gendered' indicating that the color has to encode the gender (blue for men and purple for women); if None, random colors are given to each detection
    
    Returns:
        ndarray representing the input image with the annotations drawn on it.
    """
    cv2_img = cv2.imread(img) if type(img) is str else deepcopy(img)
    if ann is not None:
        if type(ann) is dict:
            ann = [ann]
        for ann_dict in ann:
            if 'bbox' in ann_dict.keys():
                cv2_img = draw_single_annotation(cv2_img, ann_dict, color)
            else:
                assert('bbb' in ann_dict.keys() and 'fbb' in ann_dict.keys())
                color = tuple([int(c) for c in np.random.choice(range(256), size=3)]) if color is None else color
                ag_already_drawn = False
                if ann_dict['bbb'] is not None:
                    tmp_color = None if color is not None and type(color) is str and color == "gendered" else color
                    cv2_img = draw_single_annotation(cv2_img, {'bbox':ann_dict['bbb'], 'age':ann_dict['age'], 'gender':ann_dict['gender']}, tmp_color)
                    ag_already_drawn = True
                if ann_dict['fbb'] is not None:
                    if ag_already_drawn:
                        color = gender_color(ann_dict) if color is not None and type(color) is str and color == "gendered" else color
                        cv2_img = draw_single_annotation(cv2_img, {'bbox':ann_dict['fbb']}, color)
                    else:
                        cv2_img = draw_single_annotation(cv2_img, {'bbox':ann_dict['fbb'], 'age':ann_dict['age'], 'gender':ann_dict['gender']}, color)
    else:
        add_text(cv2_img, "FAILED TO DETECT", font_scale=1, font_thickness=1, font_color=(255, 255, 255), bg_color=(0, 0, 255))
    return cv2_img

def mark_point(img, x, y = None, color=(255, 0, 0)):
    """Draws a horizontal and vertical line to indicate where the provided point is located in the input image.

    ---
    Arguments:
        - img: ndarray of the reference image.
        - x: integer representing the x (horizontal) coordinate of the point, or pair of integers (list, tuple, or ndarray) where the first represents the x (horizontal) coordinates and the later represents the y (vertical) coordinates.
        - y: if x is an integer, represents the y (vertical) coordinates of the point being represented.
        - cformat: string 'rgb' or 'bgr' that defines the color format of the image that needs to be represented.
    
    Returns:
        List consisting on the reformatted version of the dictionaries given as input.
    """
    if type(x) in [list, tuple, np.array]:
        y = x[1]
        x = x[0]
    if not (type(x) is int and y is not None and type(y) is int):
        raise ValueError(f"Horizontal coordinate {x} is of type {type(x)}; Vertical coordinate {y} is of type {type(y)}.")
    cv2_img = cv2.imread(img) if type(img) is str else deepcopy(img)
    width, height, _ = img.shape
    cv2_img[y, 0:height, :] = color
    cv2_img[0:width, x, :] = color
    return cv2_img


def show(img, size=None, cformat="rgb"):
    """Outputs the image with the desired resolution.

    ---
    Arguments:
        - size: defines the resolution of the outputted image.
        - cformat: string 'rgb' or 'bgr' that defines the color format of the image that needs to be represented.
    
    Returns:
        List consisting on the reformatted version of the dictionaries given as input.
    """
    if type(img) is str:
        img = cv2.imread(img)
        cformat = "bgr"
    if type(img) is torch.Tensor:
        img = img.permute(1,2,0)
    if cformat == "bgr":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif cformat != "rgb":
        raise ValueError(f"Unknown color format {cformat}.")
    if size:
        plt.figure(figsize = size)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()

def save(img, path:str, cformat:str="rgb"):
    """Saves image to the provided path.

    ---
    Arguments:
        - path: string with the filename and directory where the image needs to be saved.
        - cformat: string 'rgb' or 'bgr' that defines the color format of the image that is being saved.
    
    Returns:
        List consisting on the reformatted version of the dictionaries given as input.
    """
    if cformat not in ["rgb", "bgr"]:
        raise ValueError(f"Value color format {cformat} is not supported. Supported types are {['rgb', 'bgr']}")
    if cformat == "bgr":
        plt.imsave(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imsave(path, img)


def preds_reformat(preds):
    """Transforms the raw outputs of the model into the inference prediction format.

    ---
    Arguments:
        - preds: dictionary or list of dictionaries with fields 'bbb' (body bounding box), 'fbb' (face bounding box), 'age' (float), 'gender' (float between 0 and 1).
    
    Returns:
        List consisting on the reformatted version of the dictionaries given as input.
    """
    if type(preds) is not list:
        preds = [preds]
    res = []
    agedict = {0:"Young", 1:"Adult", 2:"Old"}
    genderdict = {0:"Woman", 1:"Man"}
    for p in preds:
        res.append({'bbb': p['bbb'], 'fbb': p['fbb'],
                    'age': agedict[min(max(round(p['age']), 0), 2)],
                    'gender': genderdict[min(max(round(p['gender']), 0), 1)]})
    return res