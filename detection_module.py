import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

import utils
import cv2
import face_detection as fd

raw_dir = utils.root + "data/images/raw/"
face_dir = utils.root + "data/images/faces/"
ann_img_dir = utils.root + "data/images/detections/"
pred_dir = utils.root + "data/annotations/predictions/"


def dets2df(detections):
    return pd.DataFrame([{"gender":"undef",
                          "bbox":[int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))],
                          "bbox_conf": conf} for xmin, ymin, xmax, ymax, conf in detections])
        

def detect(img, detector=None):
    if detector is None:
        detector = fd.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    return dets2df(detector.detect(img))


def crop_faces(img, ann):
    faces = []
    for bbox in ann["bbox"]:
        face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if 0 not in face.shape:
            faces.append(face)
    return faces


if __name__ == "__main__":
    print("Loading detector...", end=" ")
    start = time()
    detector = fd.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    print(f"done ({round(time()-start, 2)}s)")

    print(f"Applying detection to images from directory {raw_dir}")
    image_names = [name[:-4] for name in os.listdir(raw_dir) if name[0] != "."]
    for img_name in tqdm(image_names):
        img_path, save_path = f'{raw_dir}{img_name}.jpg', f'{ann_img_dir}{img_name}.jpg'
        img = cv2.imread(img_path)[:, :, ::-1]
        ann = detect(img, detector)
        ann.to_csv(f'{pred_dir}{img_name}.csv', index=False)
        utils.draw_annotation(img_path, ann, save_path)
        faces = crop_faces(img, ann)
        std_width, std_height = 64, 64    # width, height
        faces = [cv2.resize(face, (std_width, std_height), interpolation = cv2.INTER_CUBIC) for face in faces]
        [cv2.imwrite(f"{face_dir}{img_name}_face{i}.png", face[:, :, ::-1]) for i,face in enumerate(faces)]
