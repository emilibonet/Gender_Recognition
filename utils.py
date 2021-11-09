import pandas as pd
from pandas.core.frame import DataFrame
import cv2
from ast import literal_eval  # CSV bbox as list


def read_annotations(path):
    try:
        return pd.read_csv(path, converters={"bbox": literal_eval})
    except:
        if path.split("/")[-1][0] != ".":  # if false: hidden file; ignore
            print("File not found:", path)
        return None   
    

def read_image(path):
    try:
        return cv2.imread(path).copy()
    except:
        if path.split("/")[-1][0] != ".":  # if false -> hidden file; ignore it
            print("Image not found:", path)
        return None


def plot_annotation(img_path, save_path, annotations):
    if type(annotations) not in [str, pd.DataFrame]:
        print("Please provide either a path to a csv containing the annotations or a pandas dataframe.")
        return 1
    if type(annotations) is str:
        ann = read_annotations(annotations)
    else:
        ann = annotations
    img = read_image(img_path)
    gender2color = {"man":(255, 0, 0), "woman":(0, 0, 255), "undef":(0, 255, 0)}
    if img is None or ann is None:
        return 1
    for i in range(len(ann)):
        start_point = (ann["bbox"][i][0], ann["bbox"][i][1])
        end_point = (ann["bbox"][i][2], ann["bbox"][i][3])
        # draw the rectangle
        cv2.rectangle(img, start_point, end_point, gender2color[ann["gender"][i]], thickness=3, lineType=cv2.LINE_8) 
        cv2.putText(img, ann["gender"][i], (start_point[0], start_point[1]-5), color=gender2color[ann["gender"][i]], fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = .35)
        # display the output
        cv2.imwrite(save_path, img)
    return 0