import face_recognition

from deepface import DeepFace
from deepface.detectors import FaceDetector as DFFD

from facelib import FaceDetector as FLFD, AgeGenderEstimator

import torch
from torchvision import models, transforms

import cv2
from PIL import Image


# Face detection baselines

def facerecognition_baseline(imgpath):
    fr_img = face_recognition.load_image_file(imgpath)
    face_locations = face_recognition.face_locations(fr_img)
    if face_locations:
        (y0, x0, y1, x1) = face_locations[0]
        return {'bbox':(x0, y0, x1, y1)}
    else:
        print(f"No face has been found in image {imgpath}.")
        return None


# Gender and Age Recognition (+ face detection) baselines

def deepface_baseline(imgpath):
    try:
        predDict =  DeepFace.analyze(imgpath, actions=["age", "gender"], detector_backend="retinaface", prog_bar=False)
        x0, y0 = predDict['region']['x'], predDict['region']['y']
        x1, y1 = predDict['region']['x']+predDict['region']['w'], predDict['region']['y']+predDict['region']['h']
        return {'bbox':(x0, y0, x1, y1), 'gender':predDict['gender'], 'age':predDict['age']}
    except:
        print(f"No face has been found in image {imgpath}.")
        return None


def deepface_multi_baseline(img, margin=50):
    detector = DFFD.build_model("retinaface")
    img = cv2.imread(img) if type(img) is str else img
    preds = DFFD.detect_faces(detector, "retinaface", img)
    ann = []
    for _, bbox in preds:
        x0, y0 = bbox[0], bbox[1]
        x1, y1 = bbox[0]+bbox[2], bbox[1]+bbox[3]
        try:
            predDict =  DeepFace.analyze(img[y0-margin:y1+margin, x0-margin:x1+margin,:], actions=["age", "gender"], detector_backend="retinaface", prog_bar=False)
            ann.append({'bbox':(x0, y0, x1, y1), 'age':predDict['age'], 'gender':predDict['gender']})
        except:
            ann.append({'bbox':(x0, y0, x1, y1)})
    return ann


def facedet_gar_baseline(imgpath):
    faces, boxes, scores, _ = FLFD().detect_align(cv2.imread(imgpath))
    genders, ages = AgeGenderEstimator().detect(faces)
    return [{'bbox':tuple([round(n) for n in boxes[i].numpy()]), 'age':ages[i], 'gender':"Man" if genders[i]=="Male" else "Woman", 'conf':scores[i].numpy()[0]} for i in range(faces.shape[0])]


# Pedestrian detection baselines

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Options for models
# Faster R-CNN with a ResNet50 backbone: accurate but slow
# Faster R-CNN with a MobileNet backbone: faster but less accurate
# RetinaNet with a ResNet50 backbone: good tradeoff
frcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True).to(device).eval()

def frcnn_baseline(imgpath):
    tform = transforms.ToTensor()
    img = tform(Image.open(imgpath))
    pred = frcnn([img])[0]
    x0, y0, x1, y1 = pred['boxes'][0].detach().numpy()
    return {'bbox':(round(x0), round(y0), round(x1), round(y1)), 'conf':pred['scores'][0]}
