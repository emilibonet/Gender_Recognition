import os
import pandas as pd
from tqdm import tqdm
from time import time

import utils
import cv2
import face_detection as fd

img_dir = utils.root + "data/images/raw/"
ann_img_dir = utils.root + "data/images/detections/"
pred_dir = utils.root + "data/annotations/predictions/"


def dets2df(detections):
    return pd.DataFrame([{"gender":"undef", "bbox":[int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))]} for xmin, ymin, xmax, ymax,_ in detections])
        

def detect(img, detector=None):
    if detector is None:
        detector = fd.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    return dets2df(detector.detect(img))


if __name__ == "__main__":
    print("Loading model...", end=" ")
    start = time()
    detector = fd.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    print(f"done ({round(time()-start, 2)}s)")

    print(f"Applying detection to images from directory {img_dir}")
    image_names = [name[:-4] for name in os.listdir(img_dir) if name[0] != "."]
    for img_name in tqdm(image_names):
        img_path, save_path = f'{img_dir}{img_name}.jpg', f'{ann_img_dir}{img_name}.jpg'
        img = cv2.imread(img_path)[:, :, ::-1]
        annotations = detect(img, detector)
        annotations.to_csv(f'{pred_dir}{img_name}.csv', index=False)
        utils.draw_annotation(img_path, annotations, save_path)
        