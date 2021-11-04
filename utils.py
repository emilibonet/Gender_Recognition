import pandas as pd
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


def plot_annotation(img_path, gt_path, save_path):
    gender2color = {"man":(255, 0, 0), "woman":(0, 0, 255)}
    img, gt = read_image(img_path), read_annotations(gt_path)
    if img is None or gt is None:
        return 1
    for i in range(len(gt)):
        start_point = (gt["bbox"][i][0], gt["bbox"][i][1])
        end_point = (gt["bbox"][i][2], gt["bbox"][i][3])
        # draw the rectangle
        cv2.rectangle(img, start_point, end_point, gender2color[gt["gender"][i]], thickness=3, lineType=cv2.LINE_8) 
        cv2.putText(img, gt["gender"][i], (start_point[0], start_point[1]-5), color=gender2color[gt["gender"][i]], fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = .35)
        # display the output
        cv2.imwrite(save_path, img)
    return 0