import os
import argparse
import numpy as np
import cv2
import torch

import DSFD.face_ssd as mod
import DSFD.demo as pred
from DSFD.data import config, widerface, TestBaseTransform
cfg = config.widerface_640


img_dir = "data/dataset_train_samples/"
gt_dir = "data/ground_truth_dataset_train_samples/"
ann_dir = "data/annotated_train_samples/"
pred_dir = "data/predicted_train_samples/"


def arguments():
    parser = argparse.ArgumentParser(description='DSFD:Dual Shot Face Detector')
    parser.add_argument('--trained_model', default='DSFD/weights/WIDERFace_DSFD_RES152.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default=pred_dir, type=str,
                        help='Dir to save results')
    parser.add_argument('--visual_threshold', default=0.1, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool,
                        help='Use cuda to train model')
    parser.add_argument('--img_root', default=None, help='Location of test images directory')
    parser.add_argument('--widerface_root', default=widerface.WIDERFace_ROOT, help='Location of WIDERFACE root directory')
    args = parser.parse_args()

    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    return args

def load_model(cuda, trained_model):
    net = mod.build_ssd('test', num_classes=2) # initialize SSD
    device = 'cuda' if cuda else 'cpu'
    net.load_state_dict(torch.load(trained_model, map_location=torch.device(device)))
    if cuda:
        net.cuda()
    net.eval()
    return net

# Please excuse the mess and the shitty comments. They are not mine.
def detect(net, img, args):
    thresh = cfg['conf_thresh']
    transform = TestBaseTransform((104, 117, 123))  # no sé per què mean té aquests valors ni què representen 
    max_im_shrink = ( (2000.0*2000.0) / (img.shape[0] * img.shape[1])) ** 0.5
    shrink = max_im_shrink if max_im_shrink < 1 else 1
    det0 = pred.infer(net , img , transform , thresh , args.cuda , shrink)
    det1 = pred.infer_flip(net , img , transform , thresh , args.cuda , shrink)
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = pred.infer(net , img , transform , thresh , args.cuda , st)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    factor = 2
    bt = min(factor, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = pred.infer(net , img , transform , thresh , args.cuda , bt)
    # enlarge small iamge x times for small face
    if max_im_shrink > factor:
        bt *= factor
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, pred.infer(net , img , transform , thresh , args.cuda , bt)))
            bt *= factor
        det_b = np.row_stack((det_b, pred.infer(net , img , transform , thresh , args.cuda , max_im_shrink) ))
    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]
    det = np.row_stack((det0, det1, det_s, det_b))
    det = mod.bbox_vote(det)
    return det

if __name__ == "__main__":
    args = arguments()
    net = load_model(args.cuda, args.trained_model)
    print("CUDA available:", args.cuda)
    print("Number of parameters in the network:", sum(p.numel() for p in net.parameters() if p.requires_grad))
    image_names = [name[:-4] for name in os.listdir(img_dir)]
    print(image_names)
    assert(False)
    for img_name in image_names:
        args.img_root = f'{img_dir}{img_name}.jpg'
        save_path = f'{ann_dir}{img_name}.jpg'
        img = cv2.imread(args.img_root, cv2.IMREAD_COLOR)
        # gt_path = gt_dir+img_name+".csv"
        # gt = utils.read_annotations(gt_path)
        det = detect(net, img, args)
        print(det)
    