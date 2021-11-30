import os
import pandas as pd               # dataframes
import sklearn.cluster as sklc    # clustering
import torch                      # feature extractor

# Projections
import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt   # feature space plotting

import utils
face_dir = f"{utils.root}data/images/faces/"
dirfiles = os.listdir(face_dir)
dirfiles.sort()

img_file_path = f"{utils.root}../MagFace/inference/img_list"
img_file = open(img_file_path, "w")
img_file.write('\n'.join([face_dir+name for name in dirfiles]))
img_file.close()

embd_file_path = f"{utils.root}data/face_features"
embd_file = open(embd_file_path, "w")
embd_file.close()

os.chdir(f"{utils.root}../MagFace/inference/")

os.system(
    f"""python3 {utils.root}../MagFace/inference/gen_feat.py \
        --inf_list {img_file_path} \
        --feat_list {embd_file_path} \
        --resume {utils.root}../MagFace/magface_epoch_00025.pth"""
)

os.chdir(f"{utils.root}")

