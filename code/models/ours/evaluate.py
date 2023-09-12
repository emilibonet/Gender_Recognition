import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint

import torch
from torchvision import transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from initial_design import EnsembleModel, extract_field, bbox_pairing


def preds_reformat(preds):
    if type(preds) is not list:
        preds = [preds]
    res = []
    agedict = {0:"Young", 1:"Adult", 2:"Old"}
    genderdict = {0:"Woman", 1:"Man"}
    for p in preds:
        res.append({'bbb': p['bbb'], 'fbb': p['fbb'],
                    'age': agedict[min(max(round(p['age']), 0), 2)],
                    'gender': genderdict[min(max(round(p['gender']), 0), 1)]})
    return res


def determine_range(bbox_ratio, close_lim:float=0.65, long_lim:float=0.25):
    if bbox_ratio > close_lim:
        return "close"
    if bbox_ratio < long_lim:
        return "long"
    return "mid"


def torchmetrics_formatter(preds_by_frames, bpart:str, gt:bool=False):
    if bpart not in ['body', 'face']:
        raise ValueError(f"Body part {bpart} not in ['body', 'face'].")
    bbox_field, score_field = ('bbb', 'body_score') if bpart == "body" else ('fbb', 'face_score')
    frames = []
    for preds in preds_by_frames:
        boxes, scores = [], []
        for bbox, score in [(p[bbox_field], p[score_field] if not gt else None) for p in preds if p[bbox_field] is not None]:
            boxes.append(bbox); scores.append(score) if not gt else None
        if gt:
            frames.append({'boxes':torch.tensor(boxes), 'labels':torch.tensor([0]*len(boxes))})
        else:
            frames.append({'boxes':torch.tensor(boxes), 'labels':torch.tensor([0]*len(boxes)), 'scores':torch.tensor(scores)})
    return frames


def eval_confusion_matrices(results):
    unranged_confmat = sum([results[range_] for range_ in results.keys()])
    nclasses = len(unranged_confmat)

    # Regular Accuracy
    acc = sum([unranged_confmat[i,i] for i in range(nclasses)])/sum(sum(unranged_confmat))

    # MCA
    accuracy_per_class = []
    for i in range(nclasses):
        accuracy_per_class.append(unranged_confmat[i,i]/sum(unranged_confmat[:,i]))
    mca = sum(accuracy_per_class)/nclasses

    return acc, mca, unranged_confmat
    

def evaluate(ensemble, data):
    # Initialization of metrics
    detection_metrics = {
        'face': {'tp':0, 'fp':0, 'fn':0},
        'body': {'tp':0, 'fp':0, 'fn':0},
    }
    gender2idx = {'Woman':0, 'Man':1}
    gender_metrics = {
        'close':np.zeros((2, 2)),
        'mid':np.zeros((2, 2)),
        'long':np.zeros((2, 2))
    }
    gender_metrics_oracle = {
        'close':np.zeros((2, 2)),
        'mid':np.zeros((2, 2)),
        'long':np.zeros((2, 2))
    }
    adult_gender_metrics = {
        'close':np.zeros((2, 2)),
        'mid':np.zeros((2, 2)),
        'long':np.zeros((2, 2))
    }
    adult_gender_metrics_oracle = {
        'close':np.zeros((2, 2)),
        'mid':np.zeros((2, 2)),
        'long':np.zeros((2, 2))
    }
    age2idx = {'Young':0, 'Adult':1, 'Old':2}
    age_metrics = {
        'close':np.zeros((3, 3)),
        'mid':np.zeros((3, 3)),
        'long':np.zeros((3, 3))
    }
    age_metrics_oracle = {
        'close':np.zeros((3, 3)),
        'mid':np.zeros((3, 3)),
        'long':np.zeros((3, 3))
    }

    close_lim, long_lim = 0.8, 0.2
    preds, targets = [], []
    for imgpath in tqdm(data):
        img = transforms.ToTensor()(plt.imread(imgpath))
        pred, gt = ensemble.predict(imgpath), data[imgpath]
        preds.append(pred)
        targets.append(gt)

        predr, gtr = preds_reformat(pred), preds_reformat(gt)
        total_height = plt.imread(imgpath).shape[0]
        pred_boxes, gt_boxes = extract_field(pred, field="bbb"), extract_field(gt, field="bbb")
        for pred_idx, gt_idx in bbox_pairing(pred_boxes, gt_boxes):
            # Detection
            detection_metrics['body']['tp'] += int(gt_idx is not None and pred_idx is not None)
            detection_metrics['body']['fp'] += int(gt_idx is None and pred_idx is not None)
            detection_metrics['body']['fn'] += int(gt_idx is not None and pred_idx is None)

            if gt_idx is not None and pred_idx is not None:
                det_height = gt_boxes[gt_idx][3]-gt_boxes[gt_idx][1]
                r = determine_range(det_height/total_height, close_lim, long_lim)
                # Oracle predictions
                gt_det = transforms.Resize((600, 200))(transforms.functional.crop(img, top=gt_boxes[gt_idx][1], left=gt_boxes[gt_idx][0], height=gt_boxes[gt_idx][3]-gt_boxes[gt_idx][1], width=gt_boxes[gt_idx][2]-gt_boxes[gt_idx][0])).unsqueeze(0)
                age, gender = [pred.detach().cpu().numpy()[0][0] for pred in ensemble.body_branch.body_predictor(gt_det)]
                oracle_preds = preds_reformat({'bbb':None, 'fbb':None, 'age':age, 'gender':gender})[0]
                # Gender
                gender_metrics[r][gender2idx[predr[pred_idx]['gender']], gender2idx[gtr[gt_idx]['gender']]] += 1
                gender_metrics_oracle[r][gender2idx[oracle_preds['gender']], gender2idx[gtr[gt_idx]['gender']]] += 1
                if gtr[gt_idx]['age'] == "Adult":
                    adult_gender_metrics[r][gender2idx[predr[pred_idx]['gender']], gender2idx[gtr[gt_idx]['gender']]] += 1
                    adult_gender_metrics_oracle[r][gender2idx[oracle_preds['gender']], gender2idx[gtr[gt_idx]['gender']]] += 1
                # Age
                age_metrics[r][age2idx[predr[pred_idx]['age']], age2idx[gtr[gt_idx]['age']]] += 1
                age_metrics_oracle[r][age2idx[oracle_preds['age']], age2idx[gtr[gt_idx]['age']]] += 1

        pred_boxes, gt_boxes = extract_field(pred, field="fbb"), extract_field(gt, field="fbb")
        if not pred_boxes or not gt_boxes:
            continue  # skip frames where there are no detections for the face in one of the lists
        for pred_idx, gt_idx in bbox_pairing(pred_boxes, gt_boxes):
            detection_metrics['face']['tp'] += int(gt_idx is not None and pred_idx is not None)
            detection_metrics['face']['fp'] += int(gt_idx is None and pred_idx is not None)
            detection_metrics['face']['fn'] += int(gt_idx is not None and pred_idx is None)

    # Detection metrics for body
    detection_evaluator = MeanAveragePrecision()
    detection_evaluator.update(torchmetrics_formatter(preds, "body"), torchmetrics_formatter(targets, "body", gt=True))
    detection_body_results = detection_evaluator.compute()
    detection_body_results['precision'] = detection_metrics['body']['tp']/(detection_metrics['body']['tp']+detection_metrics['body']['fp'])
    detection_body_results['recall'] = detection_metrics['body']['tp']/(detection_metrics['body']['tp']+detection_metrics['body']['fn'])
    detection_body_results['f1'] = 2/(1/detection_body_results['precision']+1/detection_body_results['recall'])
    # Detection metrics for face
    detection_evaluator = MeanAveragePrecision()
    detection_evaluator.update(torchmetrics_formatter(preds, "face"), torchmetrics_formatter(targets, "face", gt=True))
    detection_face_results = detection_evaluator.compute()
    detection_face_results['precision'] = detection_metrics['face']['tp']/(detection_metrics['face']['tp']+detection_metrics['face']['fp'])
    detection_face_results['recall'] = detection_metrics['face']['tp']/(detection_metrics['face']['tp']+detection_metrics['face']['fn'])
    detection_face_results['f1'] = 2/(1/detection_face_results['precision']+1/detection_face_results['recall'])

    detection_results = {'body': detection_body_results, 'face': detection_face_results}

    # GAR metrics
    gender_acc, gender_mca, gender_confmat = eval_confusion_matrices(gender_metrics)
    age_acc, age_mca, age_confmat = eval_confusion_matrices(age_metrics)
    gar_results = {
        'gender': {'acc':gender_acc, 'mca':gender_mca, 'confmat':gender_confmat},
        'age': {'acc':age_acc, 'mca':age_mca, 'confmat':age_confmat}
    }
    gender_acc, gender_mca, gender_confmat = eval_confusion_matrices(gender_metrics_oracle)
    age_acc, age_mca, age_confmat = eval_confusion_matrices(age_metrics_oracle)
    gar_results_oracle = {
        'gender': {'acc':gender_acc, 'mca':gender_mca, 'confmat':gender_confmat},
        'age': {'acc':age_acc, 'mca':age_mca, 'confmat':age_confmat}
    }

    # Effective range
    ranges = ['close', 'mid', 'long']
    acc_per_range = {
        'gender': {r:0 for r in ranges},
        'age': {r:0 for r in ranges}
    }
    acc_per_range_oracle = {
        'gender': {r:0 for r in ranges},
        'age': {r:0 for r in ranges}
    }
    for r in ranges:
        acc_per_range['gender'][r] = sum([adult_gender_metrics[r][i,i] for i in range(2)])/sum(sum(adult_gender_metrics[r]))
        acc_per_range['age'][r] = age_metrics[r][1,1]/sum(age_metrics[r][:,1])
        acc_per_range_oracle['gender'][r] = sum([adult_gender_metrics_oracle[r][i,i] for i in range(2)])/sum(sum(adult_gender_metrics_oracle[r]))
        acc_per_range_oracle['age'][r] = age_metrics[r][1,1]/sum(age_metrics[r][:,1])
    return detection_results, gar_results, acc_per_range, gar_results_oracle, acc_per_range_oracle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_path', help="Path to the state dict of the trained model.")
    parser.add_argument('--backbone_model', help="Determines the backbone.", default='efficientnet', choices=['convnext', 'resnet', 'efficientnet'])
    parser.add_argument('--backbone_size', help="Determines the size of backbone.", default='tiny', choices=['tiny', 'small', 'base', 'large'])
    args = parser.parse_args()

    print("Preparing for evaluation...")
    # Prepare the sequence
    te_seq = "MOT17-09"
    seq_path = os.path.join(os.getcwd().split("cv-gender-age-recognition")[0], f"cv-gender-age-recognition/data/body/{te_seq}/annotations/curated.json")
    with open(seq_path, 'r') as curated:
        data = json.load(curated)

    # Determine if baseline or multihead based on the passed state dict
    # body_sdict = torch.load(args.state_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # model_type = "baseline" if len(body_sdict.keys()) == 4 else "multihead"
    # print(f"Detected model type: {model_type}. Loading...")
    model_type = "baseline"
    if model_type == "baseline":
        body_sdict = {
            'age_model_state_dict': torch.load("/home/emili/Documents/Universitat/TFG/cv-gender-age-recognition/trained_models/ensemble/optimal_baseline/state_dict_fold3.pth", map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))['age_model_state_dict'],
            'gender_model_state_dict': torch.load("/home/emili/Documents/Universitat/TFG/cv-gender-age-recognition/trained_models/ensemble/optimal_baseline/state_dict_fold2.pth", map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))['gender_model_state_dict'],
        }
    else:
        body_sdict = torch.load("/home/emili/Documents/Universitat/TFG/cv-gender-age-recognition/trained_models/ensemble/optimal_multihead/state_dict_fold3.pth", map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Load model with state dict, inferred model type and passed Arguments for the backbone
    model = EnsembleModel(baseline_bodypred=model_type=="baseline", bodypred_backbone=args.backbone_model, bodypred_size=args.backbone_size)
    model.load_body_sdict(body_sdict)

    # Run evaluation function
    # Det: F1, mAP
    # GAR: acc, MCA
    # ER:  acc (only adults)
    print("Starting evaluation...")
    detection_results, gar_results, effective_range, gar_results_oracle, effective_range_oracle = evaluate(model, data)

    # Pretty print results
    sep_len = 50
    print("EVALUATION RESULTS")
    print(f"Used sequence: {te_seq}")
    print(f"Body predictor type: {model_type}")
    print("="*sep_len)
    print("Detection metrics:")
    pprint(detection_results)
    print("="*sep_len)
    print("Gender and Age metrics:")
    pprint(gar_results)
    print("─"*sep_len)
    print("Gender and Age metrics (oracle):")
    pprint(gar_results)
    print("="*sep_len)
    print("Effective range:")
    pprint(effective_range)
    print("─"*sep_len)
    print("Effective range (oracle):")
    pprint(effective_range_oracle)

