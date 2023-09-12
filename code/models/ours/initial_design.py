import os
import cv2
import copy
import json
import random
import numpy as np
from tqdm import tqdm
from skimage import transform
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from facelib.utils import download_weight
from facelib.Retinaface.utils.alignment import get_reference_facial_points, FaceWarpException
from facelib.Retinaface.utils.box_utils import decode, decode_landmark, prior_box, nms
from facelib.Retinaface.utils.config import cfg_mnet, cfg_re50
from facelib.Retinaface.models.retinaface import RetinaFace
from facelib.AgeGender.models.model import ShuffleneTiny, ShuffleneFull

import wandb

# Model tools

def smart_resize(det, target_size=(224, 224)):
    """Resizes the detection cropping such that the interpolation rate is minimum, thus minimizing the disturbance on the detection.

    ---
    Arguments:
        * det: ndarray with the cropped region of the image containing the detection.
        * target_size: desired output size of the cropped region.

    Returns: ndarray of the specified sizes.
    """
    if det.shape[0] > 0 and det.shape[1] > 0:
        factor_0 = target_size[0] / det.shape[0]
        factor_1 = target_size[1] / det.shape[1]
        factor = min(factor_0, factor_1)
        dsize = (int(det.shape[1] * factor), int(det.shape[0] * factor))
        det = cv2.resize(det, dsize)
        
        diff_0 = target_size[0] - det.shape[0]
        diff_1 = target_size[1] - det.shape[1]
        det = np.pad(det, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
        
    if det.shape[0:2] != target_size:
        return cv2.resize(det, target_size)
    return det


def age_binning(agepred, nbins=3):
    nages, prob_distr = len(agepred), [0]*nbins
    for i in range(nbins):
        prob_distr[i] = sum(agepred[i*nages//nbins:(i+1)*nages//nbins])
    return [pbin/sum(prob_distr) for pbin in prob_distr]


# Evaluation

def extract_field(frame_ann, field:str):
    """Given the annotations of a frame (with all the detections and labels), extact the specified field of all the detection instances into a single list.

    ---
    Arguments:
        * frame_ann: list of dictionaries representing the annotations of different detections present in a frame.
        * field: string with the name of the field that needs to be extacted.
    
    Returns: a list containing the field's values for all annotations of the frame.
    """
    if not frame_ann:
        return []
    if type(frame_ann) is dict:
        frame_ann = [frame_ann]
    return [ann_dict[field] for ann_dict in frame_ann if ann_dict[field] is not None]


def pred2ann(preds):
    if not len(preds):
        return None
    if "bbox" in preds[0].keys():
        return [{'bbox':p['bbox'], 'age':f"{np.argmax(p['age'])+1}({len(p['age'])})", 'gender':f"{'Man' if round(p['gender']) else 'Woman'}"} for p in preds]
    else:
        assert("bbb" in preds[0].keys() and "fbb" in preds[0].keys())
        return [{'bbb':p['bbb'], 'fbb':p['fbb'], 'age':f"{np.argmax(p['age'])+1}({len(p['age'])})", 'gender':f"{'Man' if round(p['gender']) else 'Woman'}"} for p in preds]



def iou(a, b):
    """Computes the Intersection over Union (IoU) between two bounding boxes.

    ---
    Arguments:
        * a: tuple, list or ndarray representing the bounding box with format (top-left y, top-left x, bottom-right y, bottom-right x)
        * b: tuple, list or ndarray representing the bounding box with format (top-left y, top-left x, bottom-right y, bottom-right x)
    
    Returns: Float representing the IoU between a and b.
    """
    i = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    i_area = max(0, i[2] - i[0] + 1) * max(0, i[3] - i[1] + 1)
    a_area = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    b_area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return i_area / float(a_area + b_area - i_area)

# Molt important per les mètriques: l'esquerra de la parella indica l'anotació, i la dreta el ground truth
def bbox_pairing(preds, gts, iou_thrs=.5):
    """Computes the bipartite bounding box matching using the Hungarian algorithm based on the IoU threshold.

    ---
    Arguments:
        * preds: list of tuples, lists or ndarrays representing the prediction bounding boxes with tlbr format.
        * gts: list of tuples, lists or ndarrays representing the ground truth bounding boxes with tlbr format.
        * iou_thrs: IoU threshold determining which is the minimum IoU required to pair two bounding boxes.
    
    Returns: List of integer pairs representing the indexes of the paired bounding boxes - to the left, indexes of the prediction bboxes, and to the right the indexes of the ground truth bboxes. \
    If of the boxes could not be paired, its index is paired with a None. 
    """
    # Si hi ha confidence, s'hauria de primer ordenar els bbox de pred per confidence
    if iou_thrs <= 0:
        raise ValueError("IoU threshold cannot be zero or negative.")
    free = [1]*len(gts)
    pairs = []
    for i in range(len(preds)):
        ious = [iou(preds[i], gts[j])*free[j] for j in range(len(gts))]
        opt_gt = np.argmax(ious)
        if ious[opt_gt] >= iou_thrs:
            pairs.append((i, opt_gt))
            free[opt_gt] = 0
        else:
            pairs.append((i, None))
    for j in range(len(gts)):
        if free[j]:
            pairs.append((None, j))
    return pairs

def metrics(pairs):
    tp = fp = fn = 0
    for annpair, gtpair in pairs:
        fn += int(annpair is None)
        fp += int(gtpair is None)
        tp += int(not (annpair is None or gtpair is None))
    if tp == 0:
        return {"recall":0, "precision":0, "f1":0}
    precision, recall = tp/(tp+fp), tp/(tp+fn)
    f1score = 2/(1/recall + 1/precision)
    return {"recall":recall, "precision":precision, "f1":f1score}


class CustomDataset(Dataset):
    def __init__(self, samples, transform=None, prob_tform:float=1.0, blurkernel:int=61):
        self.data = [s for s in samples if os.path.exists(s['path']) and s['bbb'] is not None and s['gender'] is not None and s['age'] is not None]
        self.transform = transform
        self.prob_tform = prob_tform
        self.blurkernel = blurkernel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx:int):
        rawsample = self.data[idx]
        crop = plt.imread(rawsample['path'])[rawsample['bbb'][1]:rawsample['bbb'][3], rawsample['bbb'][0]:rawsample['bbb'][2], :]
        bool_tform = self.transform is not None and random.uniform(0,1) < self.prob_tform
        if rawsample['fbb'] is not None and bool_tform:
            blurred_crop = cv2.GaussianBlur(crop, (self.blurkernel, self.blurkernel), 0)
            newfbb = (rawsample['fbb'][0] - rawsample['bbb'][0], rawsample['fbb'][1] - rawsample['bbb'][1],
                        rawsample['fbb'][2] - rawsample['bbb'][0], rawsample['fbb'][3] - rawsample['bbb'][1])
            mask = np.zeros((*crop.shape[:2], 3))
            mask[newfbb[1]:newfbb[3], newfbb[0]:newfbb[2], :] = 1
            crop = np.where(mask==np.array([1, 1, 1]), blurred_crop, crop)
        try:
            if (not bool_tform or type(transforms.ToTensor()) not in [type(t) for t in self.transform.transforms]):
                crop = transforms.ToTensor()(crop)
            if bool_tform:
                crop = self.transform(crop)
            if self.transform is None or type(transforms.Resize(size=(0,0))) not in [type(t) for t in self.transform.transforms]:
                crop = transforms.Resize(size=(600,200))(crop)
        except:
            print(rawsample)
        return crop, (rawsample['gender'], rawsample['age'])


######################### Architecture of the classes #############################
#                                                                                 #
#           Face Branch                                                           #
#           ................................................                      #
#           . ┌─────────────────┐      ┌──────────────────┐.                      #
#           . │                 │      │                  │.                      #
#      ┌─────►│  Face Detector  ├─────►│  Face Predictor  ├────────────┐          #
#      │    . │                 │      │                  │.           │          #
#      │    . └─────────────────┘      └──────────────────┘.           │          #
#      │    ................................................           ▼          #
#      │                                                     ┌──────────────────┐ #
#      │                                                     │                  │ #
# image path                                                 │  Ensemble Model  │ #
#      │                                ┌─ Baseline          │                  │ #
#      │    Body Branch                 ├─ Multihead         └──────────────────┘ #
#      │    ............................│...................           ▲          #
#      │    . ┌─────────────────┐      ┌┴─────────────────┐.           │          #
#      │    . │                 │      │                  │.           │          #
#      └─────►│  Body Detector  ├─────►│  Body Predictor  ├────────────┘          #
#           . │                 │      │                  │.                      #
#           . └─────────────────┘      └──────────────────┘.                      #
#           ................................................                      #
#                                                                                 #
###################################################################################


# Branch to manage faces

class FaceDetector():
    """The face detector is based on the implementation from the Facelib's github repo. It implements a face detector based on a pretrained model."""
    def __init__(self, backbone='mobilenet', weight_path=None, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), conf_thrs=0.99,
                 top_k=5000, nms_threshold=0.4, keep_top_k=750, face_size=(224, 224), crop_size=(96, 112), verbose=False):
        """Constructor for the face detector class.

        ---
        Arguments:
            - backbone: The backbone to use for the face detector. Can be either 'mobilenet' or 'resnet'. Default: 'mobilenet'.
            - weight_path: The path to the weights of the pretrained model. If None, the weights will be downloaded from the internet.
            - device: The device to use for the face detector. Can be either 'cuda' or 'cpu'.
            - conf_thrs: The confidence threshold for the detection scores to determine what detections are to be discarded.
            - top_k and keep_top_k: The top_k and keep_top_k parameters of the nms function (faster NMS implementation).
            - face_size: The output size of detect_align.
            - crop_size: Facelib parameter.
            - verbose: If True, the face detector will print out the progress of the face detector.
        
        Returns: A face detector object.
        """
        model, cfg = None, None
        if backbone == 'mobilenet':
            cfg = cfg_mnet
            model = RetinaFace(cfg=cfg, phase='test')
            file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mobilenet.pth')
            weight_path = os.path.join(os.path.dirname(file_name), 'weights/mobilenet.pth')
            if os.path.isfile(weight_path) == False:
                os.makedirs(os.path.split(weight_path)[0], exist_ok=True)
                download_weight(link='https://drive.google.com/uc?export=download&id=15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1',
                                file_name=file_name,
                                verbose=verbose)
                os.rename(file_name, weight_path)
        elif backbone == 'resnet':
            cfg = cfg_re50
            model = RetinaFace(cfg=cfg, phase='test')
            file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resnet50.pth')
            weight_path = os.path.join(os.path.dirname(file_name), 'weights/resnet50.pth')
            if os.path.isfile(weight_path) == False:
                os.makedirs(os.path.split(weight_path)[0], exist_ok=True)
                download_weight(link='https://www.dropbox.com/s/8sxkgc9voel6ost/resnet50.pth?dl=1',
                                file_name=file_name,
                                verbose=verbose)
                os.rename(file_name, weight_path)
        else:
            raise ValueError(f"Face detector backbone can be either 'mobilenet' or 'resnet'; passed {backbone}")             
        # settings for model
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device).eval()
        self.model = model
        self.device = device
        self.cfg = cfg
        # settings for face detection
        self.conf_thrs = conf_thrs
        self.top_k = top_k
        self.nms_thresh = nms_threshold
        self.keep_top_k = keep_top_k
        # settings for face alignment
        self.trans = transform.SimilarityTransform()
        self.out_size = face_size
        self.ref_pts = get_reference_facial_points(output_size=face_size, crop_size=crop_size)

    def preprocessor(self, img_raw):
        """Preprocesses the image for the face detector.
        
        ---
        Arguments:
            - img_raw: The raw image to be preprocessed.

        Returns: The preprocessed image and the scale.
        """
        img = torch.tensor(img_raw, dtype=torch.float32).to(self.device)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).to(self.device)
        img -= torch.tensor([104, 117, 123]).to(self.device)
        img = img.permute(2, 0, 1).unsqueeze(0)
        return img, scale

    def detect_faces(self, img_raw):
        """Detects faces in the inputted image.

        ---
        Arguments:
            - img_raw: ndarray representing the raw image where the faces need to be detected.
        
        Returns: A list of bounding boxes and two more list with the corresponding scores and landmarks.
        """
        img, scale = self.preprocessor(img_raw)
        # tic = time.time()
        with torch.no_grad():
            loc, conf, landmarks = self.model(img)  # forward pass
            # print('net forward time: {:.4f}'.format(time.time() - tic))

        priors = prior_box(self.cfg, image_size=img.shape[2:]).to(self.device)
        boxes = decode(loc.data.squeeze(0), priors, self.cfg['variance'])
        boxes = boxes * scale
        scores = conf.squeeze(0)[:, 1]
        landmarks = decode_landmark(landmarks.squeeze(0), priors, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]]).to(self.device)
        landmarks = landmarks * scale1

        # ignore low scores
        index = torch.where(scores > self.conf_thrs)[0]
        boxes = boxes[index]
        landmarks = landmarks[index]
        scores = scores[index]

        # keep top-K before NMS
        order = scores.argsort(dim=0, descending=True)[:self.top_k]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # Do NMS
        keep = nms(boxes, scores, self.nms_thresh)
        boxes = torch.abs(boxes[keep, :])
        scores = scores[:, None][keep, :]
        landmarks = landmarks[keep, :].reshape(-1, 5, 2)

        # keep top-K faster NMS
        landmarks = landmarks[:self.keep_top_k, :]
        scores = scores[:self.keep_top_k, :]
        boxes = boxes[:self.keep_top_k, :]
        return boxes, scores, landmarks
    

    def detect_align(self, img):
        """Detects faces in the inputted image and provides the resized and aligned cropped face.

        ---
        Arguments:
            - img: ndarray representing the raw image where the faces need to be detected.
        
        Returns: A torch.tensor with the resized and aligned cropped faces and two more list with the corresponding bounding boxes and detection scores.
        """
        boxes, scores, landmarks = self.detect_faces(img)

        warped = []
        for src_pts in landmarks:
            if max(src_pts.shape) < 3 or min(src_pts.shape) != 2:
                raise FaceWarpException('facial_pts.shape must be (K,2) or (2,K) and K>2')

            if src_pts.shape[0] == 2:
                src_pts = src_pts.T

            if src_pts.shape != self.ref_pts.shape:
                raise FaceWarpException('facial_pts and reference_pts must have the same shape')

            self.trans.estimate(src_pts.cpu().numpy(), self.ref_pts)
            face_img = cv2.warpAffine(img, self.trans.params[0:2, :], self.out_size)
            warped.append(face_img)

        faces = torch.tensor(np.array(warped)).to(self.device)
        return faces, boxes, scores

    def to(self, device):
        self.model.to(torch.device(device))
        self.device = torch.device(device)


class FacePredictor():
    """The face predictor is based on the implementation from the Facelib's github repo. It implements a face prediction model for age and gender based on a pretrained model."""
    def __init__(self, size='full', weight_path=None, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """Constructor for the face predictor class.

        ---
        Arguments:
            - size: The size of the backbone model. Can be either 'full' or 'small'.
            - weight_path: The path to the pretrained model weights. If None, the weights will be downloaded from the internet.
            - device: The device to use for the model. Can be either 'cpu' or 'cuda:0'.

        Returns: A face predictor object.
        """
        if size == 'tiny':
            model = ShuffleneTiny()
        elif size == 'full':
            model = ShuffleneFull()
        else:
            raise ValueError(f"Predictor parameter 'size' can only be 'tiny' or 'full', not {size}")

        # download the default weigth
        if weight_path is None:
            file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ShufflenetFull.pth')
            weight_path = os.path.join(os.path.dirname(file_name), 'weights/ShufflenetFull.pth')
            if os.path.isfile(weight_path) == False:
                print('from AgeGenderEstimator: download defualt weight started')
                os.makedirs(os.path.split(weight_path)[0], exist_ok=True)
                download_weight(link='https://drive.google.com/uc?export=download&id=1rnOZo46RjGZYrUb6Wup6sSOP37ol5I9E', file_name=file_name)
                os.rename(file_name, weight_path)
        
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device).eval()
        self.model = model
        self.device = device

    def predict(self, faces):
        """Predicts the age and gender from the cropped faces.
        
        ---
        Arguments:
            - faces: torch.tensor representing the cropped faces.
        
        Returns: Two lists of predictions for the age and gender of the inputted faces.
        """
        faces = faces.permute(0, 3, 1, 2)
        faces = faces.float().div(255).to(self.device)

        mu = torch.as_tensor([0.485, 0.456, 0.406], dtype=faces.dtype, device=faces.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=faces.dtype, device=faces.device)
        faces[:].sub_(mu[:, None, None]).div_(std[:, None, None])

        outputs = self.model(faces)
        genders = []
        ages = []
        for out in outputs:
            genders.append(out[:2])
            ages.append(out[-1])
        return genders, ages

    def to(self, device):
        """Moves the model to the indicated device.

        ---
        Arguments:
            - device: string with the device where to move the model. Can be either 'cpu' or 'cuda:0'.
        
        Returns: (void)
        """
        self.model.to(torch.device(device))
        self.device = torch.device(device)


class FaceBranch():
    """The face branch uses the face detector and the face predictor to first detect faces and then predict their age and gender."""
    def __init__(self, detector_backbone="resnet", det_thrs=0.99, predictor_size="full"):
        """Constructor for the face branch class. Initializes the detector and the predictor with the passed arguments.

        ---
        Arguments:
            - detector_backbone: The backbone of the face detector. Can be either 'resnet' or 'shufflenet'.
            - det_thrs: The threshold for the face detector.
            - predictor_size: The size of the face prediction model. Can be either 'full' or 'tiny'.
        
        Returns: A face branch object.
        """
        self.face_detector = FaceDetector(backbone=detector_backbone, conf_thrs=det_thrs)
        self.face_predictor = FacePredictor(size=predictor_size)

    def predict(self, image):
        """Performs the facial detection and prediction combining the face detector and the face predictor.

        ---
        Arguments:
            - image: ndarray representing the image where the faces need to be detected.
        
        Returns: A list of dictionaries with fields 'bbox' (tuple of the bounding box in tlbr), 'score' (float; detection score), 'age' (float; age in years), 'gender' (integer, '0' for woman and '1' for man).
        """
        # Prepare image for detection (necessita en format BGR; fem servir cv2)
        img = cv2.imread(image) if type(image) is str else image
        
        # Detection
        faces, boxes, scores = FaceDetector().detect_align(img)
        if faces.nelement() == 0:
            return None
        
        # Inference
        genders, ages = FacePredictor().predict(faces)
        return [{'bbox':tuple(bbox.numpy()),'score':score.numpy()[0],'gender':float(1-torch.argmax(gender).numpy()),'age':float(age.detach().numpy())} for bbox,score,gender,age in zip(boxes, scores, genders, ages)]

    def to(self, device):
        """Moves the model to the indicated device.

        ---
        Arguments:
            - device: string with the device where to move the model. Can be either 'cpu' or 'cuda:0'.
        
        Returns: (void)
        """
        self.face_detector.to(device)
        self.face_predictor.to(device)


# Branch to manage full bodies

class BodyDetector():
    """The body detector class is used to perform human (full-body) detection."""
    def __init__(self, detector_arch="faster_rcnn", confidence_thrs=None):
        """Constructor for the body detector class.
        
        ---
        Arguments:
            - detector_arch: The architecture of the detector. Can be either 'faster_rcnn' or 'retinanet'.
            - confidence_thrs: The confidence threshold for the detector. If None, the default value will be used (0.95 for faster_rcnn and 0.65 for retinanet).

        Returns: A body detector object.
        """
        if detector_arch == "faster_rcnn":
            self.detector = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True).eval()
            self.confidence_thrs = confidence_thrs if confidence_thrs else 0.95
        elif detector_arch == "retinanet":
            self.detector = models.detection.retinanet_resnet50_fpn(pretrained=True, pretrained_backbone=True).eval()
            self.confidence_thrs = confidence_thrs if confidence_thrs else 0.65
        else:
            raise ValueError(f"Detector '{detector_arch}' is not available. Choose one of {['faster_rcnn', 'retinanet']}.")

    def predict(self, img):
        """Detects the people inside the provided image with a bounding box.

        ---
        Arguments:
            - img: ndarray representing the image to be detected.

        Returns: A list of dictionaries with fields 'bbox' (tuple of the bounding box in tlbr), 'score' (float; detection score).
        """
        out = self.detector([img])[0]
        detections = []
        for i in range(len(out['boxes'])):
            bbox = tuple([round(n) for n in out['boxes'][i].detach().cpu().numpy()])
            label, score = int(out['labels'][i].detach().cpu().numpy()), float(out['scores'][i].detach().cpu().numpy())
            if label == 1 and score > self.confidence_thrs:
                detections.append({'bbox':bbox, 'score':score})
        return detections

    def to(self, device):
        """Moves the model to the indicated device.

        ---
        Arguments:
            - device: string with the device where to move the model. Can be either 'cpu' or 'cuda:0'.
        
        Returns: (void)
        """
        self.detector.to(device)


class BodyPredictor(nn.Module):
    """The body predictor class is used to estimate the age and gender based on body detections. It is the generalization of the subclasses Baseline and Multihead. \
    Implements shared attributes such as the loading of the backbones and the definition of the MLP heads."""
    def __init__(self):
        """Constructor for the body prediction class.

        ---
        Returns: A body predictor object.
        """
        super(BodyPredictor,self).__init__()

    def save_state_dict(self, path):
        """Saves the current state of the model's parameters.
        
        ---
        Arguments:
            - path: string with the path where to save the state.
        
        Returns: (void)
        """
        torch.save(self.state_dict(), path)

    def num_params(self, trainable=True):
        """Returns the number of parameters of the model.
        
        ---
        Arguments:
            - trainable: boolean indicating whether to count the trainable parameters or not.
        
        Returns: The number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable)
    
    def _load_backbone(self, model, size, pretrained):
        """Loads the backbone of the model from the torchvision library.
        
        ---
        Arguments:
            - model: string with the name of the model. Can be either 'convnext', 'resnet', or 'efficientnet'.
            - size: string with the size of the backbone. Can be either 'tiny', 'small', 'base', or 'large'.
            - pretrained: boolean indicating whether to load the pretrained weights or not.
        
        Returns: The backbone model.
        """
        possible_models, possible_sizes = ["convnext", "resnet", "efficientnet"], ["tiny", "small", "base", "large"]
        if model not in possible_models:
            raise ValueError(f"Backbone '{model}' not available. Choose from {possible_models}.")
        if size not in possible_sizes:
            raise ValueError(f"Backbone size '{model}' not available. Choose from {possible_sizes}.")
        if model == "convnext":
            if size == "tiny":
                return models.convnext_tiny(pretrained=pretrained)
            if size == "small":
                return models.convnext_small(pretrained=pretrained)
            if size == "base":
                return models.convnext_base(pretrained=pretrained)
            if size == "large":
                return models.convnext_large(pretrained=pretrained)
        if model == "resnet":
            if size == "tiny":
                return models.resnet18(pretrained=pretrained)
            if size == "small":
                return models.resnet34(pretrained=pretrained)
            if size == "base":
                return models.resnet50(pretrained=pretrained)
            if size == "large":
                return models.resnet101(pretrained=pretrained)
        if model == "efficientnet":
            if size == "tiny":
                return models.efficientnet_b1(pretrained=pretrained)
            if size == "small":
                return models.efficientnet_b3(pretrained=pretrained)
            if size == "base":
                return models.efficientnet_b5(pretrained=pretrained)
            if size == "large":
                return models.efficientnet_b7(pretrained=pretrained)

        
    def _load_mlp(self, in_size, out_size):
        """Loads the MLP head of the model.
        
        ---
        Arguments:
            - in_size: integer with the size of the input layer.
            - out_size: integer with the size of the output layer.

        Returns: An nn.Sequential object of the MLP head.
        """
        return nn.Sequential(
            nn.Linear(in_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),

            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            
            nn.Linear(16, out_size)
        )

    def loss_weighting(self, gt, aloss, gloss, multihead=False):
        """Computes the loss weighting for the losses of age and gender.

        ---
        Arguments:
            - gt: torch.tensor with the ground truth labels for gender (first position) and gender (second position).
            - aloss: torch.tensor with the unreduced losses for age.
            - gloss: torch.tensor with the unreduced losses for gender.
            - multihead: boolean indicating if the loss weighting needs to be done for multihead, in which case the losses for age will be increased by a factor to match the same magnitude as the losses for gender.
        
        Returns: The weighted and reduced by mean losses for age (first position) and gender (second position).
        """
        age_weights_dict = {0:.771/.0651, 1:1, 2:.771/.164}
        age_weights = torch.tensor([age_weights_dict[t] for t in gt[1].cpu().numpy()]).to(torch.device("cuda:0" if aloss.is_cuda else "cpu"))
        aloss, gloss = aloss * age_weights, gloss * 5 if multihead else gloss
        return aloss.mean(), gloss.mean()


    def inference(self, sequences, prints=False):
        # Prepare the dataloader
        root = os.getcwd().split("cv-gender-age-recognition")[0]+"cv-gender-age-recognition/data/body"
        samples = []
        for seq in os.listdir(root):
            curated_path = os.path.join(root, seq, 'annotations/curated.json')
            if seq not in sequences or not os.path.exists(curated_path):
                continue
            with open(curated_path) as curated:
                ann = json.load(curated)
                for imgpath in ann:
                    for sample in ann[imgpath]:
                        samples.append({'path':imgpath, 'bbb':sample['bbb'], 'fbb':sample['fbb'], 'gender':sample['gender'], 'age':sample['age']})
        print(f'(Testing) Total: {len(samples)}') if prints else None
        dloader = DataLoader(CustomDataset(samples), batch_size=5, shuffle=True, num_workers=2, drop_last=True)

        # Run inference
        accum_gender_loss = accum_gender_acc = accum_age_loss = accum_age_acc = 0
        gender_loss_fn, age_loss_fn = nn.BCELoss(), nn.MSELoss()
        device = torch.device("cuda:0") if next(self.parameters()).is_cuda else torch.device("cpu")
        self.eval()
        for x, y in tqdm(dloader, disable=not prints):
            x, y = x.to(device), [label.to(device) for label in y]
            aout, gout = self.forward(x)
            aloss, gloss = age_loss_fn(aout.squeeze(1), y[1].to(torch.float)), gender_loss_fn(gout.squeeze(1), y[0].to(torch.float))
            
            accum_age_loss += gloss.item() * x.size(0)
            accum_gender_loss += aloss.item() * x.size(0)
            gpreds = torch.round(gout).detach()
            accum_gender_acc += torch.sum(gpreds == y[0].unsqueeze(1)).item()
            apreds = torch.round(aout).detach()
            apreds[apreds > 2], apreds[apreds < 0] = 2, 0
            accum_age_acc += torch.sum(apreds == y[1].unsqueeze(1)).item()
        n = len(dloader.dataset)
        gender_acc, gender_loss = accum_gender_acc/n, accum_gender_loss/n
        age_acc, age_loss = accum_age_acc/n, accum_age_loss/n
        if prints:
            print('[tests] GLoss: {:.4f}, GAcc: {:.4f}, ALoss: {:.4f}, AAcc: {:.4f}'.format(
                gender_loss, gender_acc, age_loss, age_acc))
        return {'gender_acc':gender_acc, 'gender_loss':gender_loss, 'age_acc':age_acc, 'age_loss':age_loss}


class BaselineBodyPredictor(BodyPredictor):
    """Body predictor subclass implementing the Baseline architecture: there is a separate feature extraction and MLP head for each task."""
    def __init__(self, backbone_model='convnext', backbone_size='small', pretrained_backbone=True, coupling_size=1000, dropout=0):
        """Constructor for the Baseline body predictor.
        
        ---
        Arguments:
            - backbone_model: string with the name of the backbone model. Possible values: 'convnext', 'resnet', 'efficientnet'. Default: 'convnext'.
            - backbone_size: string with the size of the backbone. Possible values: 'tiny', 'small', 'base', 'large'. Default: 'small'.
            - pretrained_backbone: boolean indicating if the backbone should be pretrained. Default: True.
            - coupling_size: integer with the size of the coupling layer (between backbone and MLP). Default: 1000.
            - dropout: float with the dropout rate. Default: 0.
        
        Returns: A BaselineBodyPredictor object.
        """
        super(BaselineBodyPredictor,self).__init__()
        self.dropout = dropout
        self.best_acc = {'age':0, 'gender':0}
        self.age_model = nn.Sequential(
            self._load_backbone(backbone_model, backbone_size, pretrained_backbone),
            self._load_mlp(coupling_size, 1)
        )
        self.gender_model = nn.Sequential(
            self._load_backbone(backbone_model, backbone_size, pretrained_backbone),
            self._load_mlp(coupling_size, 1),
            nn.Sigmoid()
        )
        self.eval()
    
    def forward(self, x):
        """Forward pass of the model.
        
        ---
        Arguments:
            - x: torch.tensor with the input images.
        
        Returns: two torch.tensors with the predictions for age and gender.
        """
        return self.age_model(x), self.gender_model(x)

    def train_model(self, dloaders, optimizers, config, prints=False, ocp=None, wb_run=None):
        """Method to train the Baseline model.

        ---
        Arguments:
            - dloaders: dictionary with two data loaders: one for the training set (with key 'train') and one for the validation set (with key 'valid').
            - optimizers: dictionary with two optimizers: one for the age model (with key 'age') and one for the gender model (with key 'gender').
            - config: dictionary with the configuration parameters. Keys: 'device', 'dropout', 'lr', 'epochs', 'momentum', 'batch_size', 'weight_decay'.
            - prints: boolean - if True, prints the logs of the training progress. Default: False.
            - ocp: string with the path where to save the model parameters' Optimal (in validation) CheckPoint state. If None, these are not saved. Default: None.
            - wb_run: object returned by the wandb.init() function. If None, the wandb logs are not saved. Default: None.

        Returns: A dictionary with the training statistics.
        """
        gender_loss_fn, age_loss_fn = nn.BCELoss(reduce=False), nn.MSELoss(reduce=False)
        wandb.watch(self, log="all", log_freq=50) if wb_run else None
        device = torch.device('cuda:0' if next(self.parameters()).is_cuda else 'cpu')
        epoch_stats = {'train':{'age':{'losses':[], 'accs':[]}, 'gender':{'losses':[], 'accs':[]}},
                       'valid':{'age':{'losses':[], 'accs':[]}, 'gender':{'losses':[], 'accs':[]}}}
        best_gender_model = copy.deepcopy(self.gender_model.state_dict())
        best_age_model = copy.deepcopy(self.age_model.state_dict())
        for epoch in range(config['epochs']):
            print(f"Epoch {epoch+1}/{config['epochs']}") if prints else None
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()
                accum_gender_loss = accum_gender_acc = accum_age_loss = accum_age_acc = 0
                for x, y in dloaders[phase]:
                    x, y = x.to(device), [label.to(device) for label in y]
                    optimizers['gender'].zero_grad(); optimizers['age'].zero_grad()
                    aout, gout = self.forward(x)
                    aloss, gloss = age_loss_fn(aout.squeeze(1), y[1].to(torch.float)), gender_loss_fn(gout.squeeze(1), y[0].to(torch.float))
                    aloss, gloss = self.loss_weighting(y, aloss, gloss, multihead=False)
                    if phase == "train":
                        aloss.backward()
                        optimizers['age'].step()
                        gloss.backward()
                        optimizers['gender'].step()
                    # Training statistics
                    accum_age_loss += aloss.item() * x.size(0)
                    accum_gender_loss += gloss.item() * x.size(0)
                    gpreds = torch.round(gout).detach()
                    accum_gender_acc += torch.sum(gpreds == y[0].unsqueeze(1)).item()
                    apreds = torch.round(aout).detach()
                    apreds[apreds > 2], apreds[apreds < 0] = 2, 0
                    accum_age_acc += torch.sum(apreds == y[1].unsqueeze(1)).item()
                n = len(dloaders[phase].dataset)
                epoch_age_loss, epoch_age_acc = accum_age_loss/n, accum_age_acc/n
                epoch_gender_loss, epoch_gender_acc = accum_gender_loss/n, accum_gender_acc/n
                epoch_stats[phase]['age']['losses'].append(epoch_age_loss); epoch_stats[phase]['age']['accs'].append(epoch_age_acc)
                epoch_stats[phase]['gender']['losses'].append(epoch_gender_loss); epoch_stats[phase]['gender']['accs'].append(epoch_gender_acc)
                if prints:
                    print('[{}] GLoss: {:.4f}, GAcc: {:.4f}, ALoss: {:.4f}, AAcc: {:.4f}'.format(
                        phase, epoch_gender_loss, epoch_gender_acc, epoch_age_loss, epoch_age_acc))
                if wb_run is not None:
                    wandb.log({f"{phase}_gender_loss":epoch_gender_loss, f"{phase}_gender_acc":epoch_gender_acc,
                                f"{phase}_age_loss":epoch_age_loss, f"{phase}_age_acc":epoch_age_acc}, step=epoch)
                if phase == "valid":
                    if self.best_acc['age'] < epoch_age_acc:
                        self.best_acc['age'] = epoch_age_acc
                        best_age_model = copy.deepcopy(self.age_model.state_dict())
                    if self.best_acc['gender'] < epoch_gender_acc:
                        self.best_acc['gender'] = epoch_gender_acc
                        best_gender_model = copy.deepcopy(self.gender_model.state_dict())
        self.age_model.load_state_dict(best_age_model)
        self.gender_model.load_state_dict(best_gender_model)
        if ocp is not None:
            torch.save({
                'age_model_state_dict': self.age_model.state_dict(),
                'age_optimizer_state_dict': optimizers['age'].state_dict(),
                'gender_model_state_dict': self.gender_model.state_dict(),
                'gender_optimizer_state_dict': optimizers['gender'].state_dict()},
                ocp)
            print(f"Model and optimizer state dicts have been saved at {ocp}")
        return epoch_stats
        

class MultiheadBodyPredictor(BodyPredictor):
    """Body predictor subclass implementing the Multihead architecture: there is a feature extraction shared by both heads."""
    def __init__(self, backbone_model="convnext", backbone_size="small", pretrained_backbone=True, coupling_size=1000, dropout=0):
        """Constructor for the Multihead body predictor.
        
        ---
        Arguments:
            - backbone_model: string with the name of the backbone model. Possible values: 'convnext', 'resnet', 'efficientnet'. Default: 'convnext'.
            - backbone_size: string with the size of the backbone. Possible values: 'tiny', 'small', 'base', 'large'. Default: 'small'.
            - pretrained_backbone: boolean indicating if the backbone should be pretrained. Default: True.
            - coupling_size: integer with the size of the coupling layer (between backbone and MLP). Default: 1000.
            - dropout: float with the dropout rate. Default: 0.
        
        Returns: A MultiheadBodyPredictor object.
        """
        super(MultiheadBodyPredictor,self).__init__()
        self.dropout = dropout
        self.best_acc = 0
        self.backbone = self._load_backbone(backbone_model, backbone_size, pretrained_backbone)
        self.age_head = self._load_mlp(coupling_size, 1)
        self.gender_head = nn.Sequential(
            self._load_mlp(coupling_size, 1),
            nn.Sigmoid()
        )
        self.eval()
    
    def forward(self, x):
        """Forward pass of the model.
        
        ---
        Arguments:
            - x: torch.tensor with the input images.
        
        Returns: two torch.tensors with the predictions for age and gender.
        """
        x = self.backbone(x)
        return self.age_head(x), self.gender_head(x)

    def train_model(self, dloaders, optimizer, config, prints=False, ocp=None, wb_run=None):
        """Method to train the Multihead model.

        ---
        Arguments:
            - dloaders: dictionary with two data loaders: one for the training set (with key 'train') and one for the validation set (with key 'valid').
            - optimizer: torch.optim.Optimizer object - in this case, it applies to the whole model.
            - config: dictionary with the configuration parameters. Keys: 'device', 'dropout', 'lr', 'epochs', 'momentum', 'batch_size', 'weight_decay'.
            - prints: boolean - if True, prints the logs of the training progress. Default: False.
            - ocp: string with the path where to save the model parameters' Optimal (in validation) CheckPoint state. If None, these are not saved. Default: None.
            - wb_run: object returned by the wandb.init() function. If None, the wandb logs are not saved. Default: None.

        Returns: A dictionary with the training statistics.
        """
        gender_loss_fn, age_loss_fn = nn.BCELoss(reduce=False), nn.MSELoss(reduce=False)
        wandb.watch(self, log="all", log_freq=50) if wb_run else None
        device = torch.device('cuda:0' if next(self.parameters()).is_cuda else 'cpu')
        epoch_stats = {'train':{'age':{'losses':[], 'accs':[]}, 'gender':{'losses':[], 'accs':[]}},
                       'valid':{'age':{'losses':[], 'accs':[]}, 'gender':{'losses':[], 'accs':[]}}}
        best_model = copy.deepcopy(self.state_dict())
        for epoch in range(config['epochs']):
            print(f"Epoch {epoch+1}/{config['epochs']}") if prints else None
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()
                accum_gender_loss = accum_gender_acc = accum_age_loss = accum_age_acc = 0
                for x, y in dloaders[phase]:
                    x, y = x.to(device), [label.to(device) for label in y]
                    optimizer.zero_grad()
                    try:
                        aout, gout = self.forward(x)
                        aloss, gloss = age_loss_fn(aout.squeeze(1), y[1].to(torch.float)), gender_loss_fn(gout.squeeze(1), y[0].to(torch.float))
                        aloss, gloss = self.loss_weighting(y, aloss, gloss, multihead=True)
                        if phase == 'train':
                            loss = (aloss + gloss)/2  # combinació de les dos losses
                            loss.backward()
                            optimizer.step()
                        # Training statistics
                        accum_age_loss += aloss.item() * x.size(0)
                        accum_gender_loss += gloss.item() * x.size(0)
                        gpreds = torch.round(gout).detach()
                        accum_gender_acc += torch.sum(gpreds == y[0].unsqueeze(1)).item()
                        apreds = torch.round(aout).detach()
                        apreds[apreds > 2], apreds[apreds < 0] = 2, 0
                        accum_age_acc += torch.sum(apreds == y[1].unsqueeze(1)).item()
                    except:
                        print("Failed inside train loop.")
                n = len(dloaders[phase].dataset)
                epoch_age_loss, epoch_age_acc = accum_age_loss/n, accum_age_acc/n
                epoch_gender_loss, epoch_gender_acc = accum_gender_loss/n, accum_gender_acc/n
                epoch_stats[phase]['age']['losses'].append(epoch_age_loss); epoch_stats[phase]['age']['accs'].append(epoch_age_acc)
                epoch_stats[phase]['gender']['losses'].append(epoch_gender_loss); epoch_stats[phase]['gender']['accs'].append(epoch_gender_acc)
                if prints:
                    print('[{}] GLoss: {:.4f}, GAcc: {:.4f}, ALoss: {:.4f}, AAcc: {:.4f}'.format(
                        phase, epoch_gender_loss, epoch_gender_acc, epoch_age_loss, epoch_age_acc))
                if wb_run is not None:
                    try:
                        wandb.log({f"{phase}_gender_loss":epoch_gender_loss, f"{phase}_gender_acc":epoch_gender_acc,
                                    f"{phase}_age_loss":epoch_age_loss, f"{phase}_age_acc":epoch_age_acc}, step=epoch)
                    except:
                        print("Failed while logging.")
                if phase == "valid" and self.best_acc < (epoch_age_acc + epoch_gender_acc)/2:
                    self.best_acc = (epoch_age_acc + epoch_gender_acc)/2
                    best_model = copy.deepcopy(self.state_dict())
        self.load_state_dict(best_model)
        if ocp is not None:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                ocp)
        return epoch_stats


class BodyBranch():
    """The body branch uses the body detector and the body predictor to first detect humans and then predict their age and gender."""
    def __init__(self, detector_arch="faster_rcnn", confidence_thrs=None, baseline_bodypred=True, backbone_model="convnext", backbone_size="small", pretrained_backbone=True, coupling_size=1000, dropout=0):
        # Prepare models with passed configuration
        self.body_detector = BodyDetector(detector_arch, confidence_thrs)
        if baseline_bodypred:
            self.body_predictor = BaselineBodyPredictor(backbone_model, backbone_size, pretrained_backbone, coupling_size, dropout)
        else:
            self.body_predictor = MultiheadBodyPredictor(backbone_model, backbone_size, pretrained_backbone, coupling_size, dropout)

    def predict(self, image):
        """Performs the body detection and prediction combining the body detector and the body predictor.

        ---
        Arguments:
            - image: ndarray representing the image where the bodies need to be detected, or string with the path to the image.
        
        Returns: A list of dictionaries with fields 'bbox' (tuple of the bounding box in tlbr), 'score' (float; detection score), 'age' (float; prediction on age class encodings - integers 0, 1, 2), 'gender' (float; between 0 and 1 - '0' for woman and '1' for man).
        """
        # Prepare image for detection
        if type(image) is str:
            image = plt.imread(image)
        if type(image) is np.ndarray:
            img = transforms.ToTensor()(image)
        else:
            assert(type(image) is torch.Tensor)
            img = image
        if next(self.body_detector.detector.parameters()).is_cuda and next(self.body_predictor.parameters()).is_cuda:
            img = img.cuda()
        
        # Detection
        detections = self.body_detector.predict(img)
        if not detections:
            return None
        
        # Inference
        preds = []
        for det in detections:
            bbox, score = det['bbox'], det['score']
            det = transforms.functional.crop(img, top=bbox[1], left=bbox[0], height=bbox[3]-bbox[1], width=bbox[2]-bbox[0])
            det_resized = transforms.Resize((600, 200))(det).unsqueeze(0)
            age, gender = [pred.detach().cpu().numpy()[0] for pred in self.body_predictor(det_resized)]
            preds.append({'bbox':tuple(bbox), 'score':score, 'gender':gender[0], 'age':age[0]})
        return preds

    def eval(self):
        """Set model to evaluation mode - will not save computation graph after forward pass."""
        self.body_predictor.eval()

    def train(self):
        """Set model to training mode - will save computation graph after forward pass."""
        self.body_predictor.train()

    def to(self, device):
        """Move model to device.
        
        ---
        Arguments:
            - device: string with the name of the device. Possible values are 'cpu' or 'cuda'.

        Returns: (void)
        """
        self.body_detector.to(device)
        self.body_predictor.to(device)


# Joint face + body ensemble model

class EnsembleModel():
    """
    The ensemble model combines the face and body branches into a harmonized prediction using a weighting factor. This increases the overall robustness of the model: if the face branch is not able to detect a face, \
    the body branch will be used to predict, and vice versa. Additionally, if one of the branches is 'unsure' about the prediction, the other branch will be able to impose its prediction.
    """
    def __init__(self, facedet_backbone:str="resnet", bodydet_arch:str="faster_rcnn", conf_thrs=None, baseline_bodypred=True,
                 bodypred_backbone:str="efficientnet", bodypred_size:str="tiny", pretrained_bodypred:bool=True, coupling_size:int=1000, dropout:float=0):
        """Constructor for the ensemble model. Initializes the face and body branches.
        
        ---
        Arguments:
            - facedet_backbone: string with the name of the backbone to use for the face detector. Possible values are 'mobilenet' or 'resnet'. Default: 'resnet'.
            - bodydet_arch: string with the name of the architecture to use for the body detector. Possible values are 'faster_rcnn', 'retinanet'. Default: 'faster_rcnn'.
            - conf_thrs: float with the confidence threshold for the body detector. Default: 0.5.
            - baseline_bodypred: boolean, if True: use the Baseline body predictor, else: use the Multihead. Default: True.
            - bodypred_backbone: string with the name of the backbone to use for the body predictor. Possible values are 'resnet', 'efficientnet', 'convnext'. Default: 'efficientnet'.
            - bodypred_size: string with the size to use for the body predictor's backbone. Possible values are 'tiny', 'small', 'base', 'large'. Default: 'tiny'.7
            - pretrained_bodypred: boolean, if True: use pretrained backbone for the body predictor. Default: True.
            - coupling_size: int with the size of the coupling layer (between the backbone and the MLP). Default: 1000.
            - dropout: float with the dropout probability. Default: 0.
        
        Returns: An EnsembleModel object.
        """
        self.face_branch = FaceBranch(facedet_backbone)
        self.is_baseline = baseline_bodypred
        self.body_branch = BodyBranch(bodydet_arch, conf_thrs, baseline_bodypred, bodypred_backbone, bodypred_size, pretrained_bodypred, coupling_size, dropout)

    def predict(self, imgpath:str, alpha=0.5):
        """Predicts the age and gender of the people in an image using ensemble predictions.

        ---
        Arguments:
            - imgpath: string with the path to the image.
            - alpha: float between 0 and 1 of the weighting factor between the predictions of the face and body branches. If 0, predicts only with the body branch - if 1, predicts only with the face branch. Default: 0.5.
        
        Returns: A list of dictionaries with fields 'bbb' (tuple of the body bounding box in tlbr; None if body was not detected), 'fbb' (tuple of the face bounding box in tlbr; None if face was not detected), \
        'body_score' (float; score for body detection; None if body was not detected), 'age' (float; prediction on age class encodings - integers 0, 1, 2), 'gender' (float; between 0 and 1 - '0' for woman and '1' for man).
        """
        facepreds = self.face_branch.predict(imgpath)
        bodypreds = self.body_branch.predict(imgpath)
        preds = []
        pairing = bbox_pairing(extract_field(facepreds, field="bbox"), extract_field(bodypreds, field="bbox"), iou_thrs=1e-6)
        for face_idx, body_idx in pairing:
            if face_idx is None:
                preds.append({'bbb':tuple(self._chdtype(bodypreds[body_idx]['bbox'], int)), 'fbb':None,
                              'body_score': bodypreds[body_idx]['score'], 'face_score': None,
                              'age':float(bodypreds[body_idx]['age']),
                              'gender':float(bodypreds[body_idx]['gender'])})
            elif body_idx is None:
                preds.append({'bbb':None, 'fbb':tuple(self._chdtype(facepreds[face_idx]['bbox'], int)),
                              'body_score': None, 'face_score': facepreds[face_idx]['score'],
                              'age':self._cont2enc(facepreds[face_idx]['age']),
                              'gender':float(facepreds[face_idx]['gender'])})
            else:
                joint_age = alpha*self._cont2enc(facepreds[face_idx]['age'])+(1-alpha)*bodypreds[body_idx]['age']
                joint_gender = (facepreds[face_idx]['gender']+bodypreds[body_idx]['gender'])/2
                preds.append({'bbb':tuple(self._chdtype(bodypreds[body_idx]['bbox'], int)),
                              'fbb':tuple(self._chdtype(facepreds[face_idx]['bbox'], int)),
                              'body_score': bodypreds[body_idx]['score'], 'face_score': facepreds[face_idx]['score'],
                              'age':float(joint_age), 'gender':float(joint_gender)})
        return preds

    def to(self, device):
        """Moves the model to the indicated device.

        ---
        Arguments:
            - device: string with the device where to move the model. Can be either 'cpu' or 'cuda:0'.
        
        Returns: (void)
        """
        self.body_branch.to(device)

    def load_body_sdict(self, state_dict:dict):
        """Loads state dict for body predictor.

        ---
        Arguments:
            - state_dict: dictionary with the state dict for the body predictor. If body predictor is baseline, state_dict must contain 'age_model_state_dict' and 'gender_model_state_dict'. \
        Otherwise, it must contain 'model_state_dict' as the state dict for the whole body predictor model.

        Returns: (void)
        """
        if self.is_baseline:
            if 'age_model_state_dict' in state_dict.keys():
                self.body_branch.body_predictor.age_model.load_state_dict(state_dict['age_model_state_dict'])
                print("Age model state dict has been successfully loaded.")
            else:
                print("State dict not found for age model.")
            if 'gender_model_state_dict' in state_dict.keys():
                self.body_branch.body_predictor.gender_model.load_state_dict(state_dict['gender_model_state_dict'])
                print("Gender model state dict has been successfully loaded.")
            else:
                print("State dict not found for gender model.")
        else:
            assert('model_state_dict' in state_dict.keys())
            self.body_branch.body_predictor.load_state_dict(state_dict['model_state_dict'])
            print("Model state dict has been successfully loaded.")


    def _chdtype(self, l:list, dtype):
        """Change datatype of list elements.

        ---
        Arguments:
            - l: list with elements to change datatype.
            - dtype: new datatype.

        Returns: list with elements of the changed datatype.
        """
        return [dtype(x) for x in l]

    def _enc2cont(self, ageprob:list):
        """Encoded age value to continuous age.

        Arguments:
            - ageprob: list with the probabilities for each age.

        Returns: Expected age based on the weighted sum of all ages.
        """
        return sum([age*prob for age,prob in enumerate(ageprob)])

    def _cont2enc(self, age:float):
        """Continuous age value to encoder value: ages from 0 to 20 are mapped to 0 ('Young'), ages from 20 to 60 to 1 ('Adult'), and more than 60 are mapped to 2 ('Old').

        ---
        Arguments:
            - age: continuous age value.

        Returns: Encoded age value.
        """
        if age <= 20:
            return 0
        if age > 20 and age <= 60:
            return 1
        return 2
