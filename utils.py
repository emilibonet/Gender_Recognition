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


def IoU(bbox_a, bbox_b):
    # coordinates of the intersection rectangle
    x_min, y_min = max(bbox_a[0], bbox_b[0]), max(bbox_a[1], bbox_b[1])
    x_max, y_max = min(bbox_a[2], bbox_b[2]), min(bbox_a[3], bbox_b[3])

    # areas of bboxes a, b and intersection
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    area_inter = (x_max - x_min) * (y_max - y_min)
    return area_inter / float(area_a + area_b - area_inter)


def pred2gt_pairup(pred_df, gt_df, iou_thrs=.5):
    pred_sorted_idx = np.argsort([conf for conf in pred_df["conf"]])
    # Assign prediction bounding boxes to the ground truth bounding box that has the highest IoU and is over the threshold.
    pred_adjlist = [None]*len(pred_df)
    for pred_idx in pred_sorted_idx:
        gt_iousorted_idx = np.argsort([IoU(pred_df["bbox"][pred_idx], gt_bbox) for gt_bbox in gt_df["bbox"]])
        for gt_idx in gt_iousorted_idx:
            if not gt_idx in pred_adjlist:
                break  # we have found an unassigned gt bbox
        if IoU(pred_df["bbox"][pred_idx], gt_df["bbox"][gt_idx]) > iou_thrs:
            pred_adjlist[pred_idx] = gt_idx
    return pred_adjlist  # pred_df["gender"][4] == gt_df["gender"][pred_adjlist[4]]
