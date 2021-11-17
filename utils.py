import cv2
import pandas as pd
import xml.etree.ElementTree as ET
from ast import literal_eval  # CSV bbox as list

# Find project's root directory
import os 
root = os.getcwd().split("Gender_Recognition")[0]+"Gender_Recognition/"


def parse_xml(path_to_xml):
    root = ET.parse(path_to_xml).getroot()
    rows = []
    for object in root.findall("object"):
        gender = object.find("name").text
        xmin, ymin = int(object.find("bndbox").find("xmin").text), int(object.find("bndbox").find("ymin").text)
        xmax, ymax = int(object.find("bndbox").find("xmax").text), int(object.find("bndbox").find("ymax").text)
        rows.append({"gender":gender, "bbox":[xmin, ymin, xmax, ymax]})
    return rows


def read_annotations(path):
    try:
        return pd.read_csv(path, converters={"bbox": literal_eval})
    except:
        if path.split("/")[-1][0] != ".":  # if false -> hidden file; ignore
            print("File not found:", path)
        return None   
    

def read_image(path):
    try:
        return cv2.imread(path).copy()
    except:
        if path.split("/")[-1][0] != ".":  # if false -> hidden file; ignore it
            print("Image not found:", path)
        return None


def draw_annotation(img_path, annotations, save_path=None):
    if type(annotations) not in [str, pd.DataFrame]:
        print("Please provide either a path to a csv containing the annotations or a pandas dataframe.")
        return None
    if type(annotations) is str:
        ann = read_annotations(annotations)
    else:
        ann = annotations
    img = read_image(img_path)
    gender2color = {"man":(255, 0, 0), "woman":(0, 0, 255), "undef":(0, 255, 0)}
    if img is None or ann is None:
        return None
    for i in range(len(ann)):
        start_point = (ann["bbox"][i][0], ann["bbox"][i][1])
        end_point = (ann["bbox"][i][2], ann["bbox"][i][3])
        # draw the rectangle
        cv2.rectangle(img, start_point, end_point, gender2color[ann["gender"][i]], thickness=1, lineType=cv2.LINE_8) 
        cv2.putText(img, ann["gender"][i], (start_point[0], start_point[1]-5), color=gender2color[ann["gender"][i]], fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = .35)
        
        if save_path is not None:
            cv2.imwrite(save_path, img)
    return img