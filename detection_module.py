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
gt_dir = utils.root + "data/annotations/ground_truth/"


def dets2df(detections):
    return pd.DataFrame([{"gender":"undef",
                          "bbox":[int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))],
                          "bbox_conf": conf} for xmin, ymin, xmax, ymax, conf in detections])
        

def detect(img, detector=None):
    if detector is None:
        detector = fd.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    return dets2df(detector.detect(img))


def crop_faces(img, ann, padding=0, padding_mode="abs"):
    assert padding_mode in ['abs', 'rel'], f"Unknown padding mode '{padding_mode}'; it has to be either 'abs' or 'rel'."
    faces = []
    for bbox in ann["bbox"]:
        if padding_mode == "abs":
            pad_ymin, pad_ymax, pad_xmin, pad_xmax = bbox[1]-padding, bbox[3]+padding, bbox[0]-padding, bbox[2]+padding
        else:
            ypad, xpad = round(img.shape[0]*padding), round(img.shape[1]*padding)
            pad_ymin, pad_ymax, pad_xmin, pad_xmax = bbox[1]-ypad, bbox[3]+ypad, bbox[0]-xpad, bbox[2]+xpad
        ymin, ymax = max(0, pad_ymin), min(img.shape[0], pad_ymax)
        xmin, xmax = max(0, pad_xmin), min(img.shape[1], pad_xmax)
        face = img[ymin:ymax, xmin:xmax]
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
    for img_name in tqdm(image_names, colour="#81c934"):
        img_path = f'{raw_dir}{img_name}.jpg'
        save_gt, save_preds = f'{ann_img_dir}gt/{img_name}.jpg', f'{ann_img_dir}preds/{img_name}.jpg'
        gt = utils.read_annotations(f'{gt_dir}{img_name}.csv')
        img = cv2.imread(img_path)[:, :, ::-1]
        ann = detect(img, detector)
        ann.to_csv(f'{pred_dir}{img_name}.csv', index=False)
        pred_adjlist = utils.pred2gt_pairup(ann, gt)
        utils.draw_annotation(img_path, gt, save_gt)
        utils.draw_annotation(img_path, ann, save_preds)
        faces = crop_faces(img, ann, padding=0.06, padding_mode="rel")
        std_width, std_height = 64, 64
        faces = [cv2.resize(face, (std_width, std_height), interpolation = cv2.INTER_CUBIC) for face in faces]
        man_count = 0
        for i,face in enumerate(faces):
            count = man_count if gt['gender'][i]=="man" else i-man_count
            cv2.imwrite(f"{face_dir}{gt['gender'][i]}/{img_name}_{gt['gender'][i]}{count}.png", face[:, :, ::-1])
            man_count += gt['gender'][i]=="man"
        # [cv2.imwrite(f"{face_dir}{img_name}_face{i}.png", face[:, :, ::-1]) for i,face in enumerate(faces)]
