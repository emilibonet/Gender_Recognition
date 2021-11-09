import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

import utils
import cv2
import face_detection as fd

img_dir = "data/dataset_train_samples/"
gt_dir = "data/ground_truth_dataset_train_samples/"
ann_dir = "data/annotated_train_samples/"
pred_dir = "data/predicted_train_samples/"

def dets2df(detections):
    return pd.DataFrame([{"gender":"undef", "bbox":[int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))]} for xmin, ymin, xmax, ymax,_ in detections])
        

if __name__ == "__main__":
    print("Loading model...", end=" ")
    start = time()
    detector = fd.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    print(f"done ({round(time()-start, 2)}s)")

    print(f"Applying detection to images from directory {img_dir}")
    image_names = [name[:-4] for name in os.listdir(img_dir) if name[0] != "."]
    for i,img_name in tqdm(enumerate(image_names)):
        img_path, save_path = f'{img_dir}{img_name}.jpg', f'{ann_dir}{img_name}.jpg'
        img = cv2.imread(img_path)[:, :, ::-1]
        detections = detector.detect(img)
        annotations = dets2df(detections)
        print(i, annotations)
        utils.plot_annotation(img_path, save_path, annotations)
        

"""
import cv2
import face_detection
print(face_detection.available_detectors)
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
# BGR to RGB
im = cv2.imread("path_to_im.jpg")[:, :, ::-1]

detections = detector.detect(im)
"""