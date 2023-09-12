import os
import json
import numpy as np
import torch
from tqdm import tqdm
from models.ours.initial_design import EnsembleModel


def add_pathset():
    """Queries the user to define the path to a single pathset.
    
    ---
    Returns:
        string representing the path to the pathset.
    """
    while True:
        try:
            new_ps = input("Path to pathset: ")
            if not os.path.isdir(new_ps):
                print(f"Path to directory '{new_ps}' not found or is not a directory.")
                continue
            if not os.path.isfile(new_ps+"/imgpaths.txt"):
                print(f"Path to file '{new_ps+'/imgpaths.txt'}' not found or is not a file. Make sure that '{new_ps}' is a pathset directory with standard structure.")
                continue
            return new_ps
        except KeyboardInterrupt:
            print("\nCancelling new pathset addition.")
            return None

def generate_pseudolabels(pathsets):
    """Updates the annotations directory of each pathset with a json containing the synthetic annotations.

    ---
    Arguments:
        - pathsets: list of strings representing all the paths to the pathsets for which we need to synthesize annotations.

    Returns:
        (void)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nbins = 4
    model = EnsembleModel(nbins=nbins, baseline_bodypred=False, bodypred_model="resnet", bodypred_size="tiny", pretrained_bodypred=False)
    model.to(device)
    on_cuda = next(model.body_branch.body_detector.detector.parameters()).is_cuda and next(model.body_branch.body_predictor.parameters()).is_cuda
    print(f"Model running on {'cuda' if on_cuda else 'cpu'}.")
    for ps in pathsets:
        print(f"Processing pathset '{ps}'.")
        try:
            ann = []
            with open(ps+"/imgpaths.txt", 'r') as imgpaths_file:
                for imgpath in tqdm([line.strip("\n") for line in imgpaths_file.readlines()]):
                    preds = model.predict(imgpath, alpha=1)
                    for i in range(len(preds)):
                        if preds[i]['age'] < 0: preds[i]['age'] = 0
                        elif preds[i]['age'] > nbins: preds[i]['age'] = nbins
                        preds[i]['age'] = round(preds[i]['age'])
                        preds[i]['gender'] = round(preds[i]['gender'])
                    ann.append({'frame_id':imgpath.split("/")[-1], "annotations":preds})
        finally:
            print(f"Writing synthesized annotations to '{ps + '/annotations/synthetic.json'}'.")
            with open(ps + "/annotations/synthetic.json", "w") as synth_file:
                json.dump(ann, synth_file, indent=4)


if __name__ == "__main__":
    pathsets = []
    try:
        while True:  # Query all the pathsets for which we need to create synthetic annotations.
                yn = input("Add pathset ([y]/n)? ")
                if not yn in ["", "y", "Y", "n", "N"]:
                    continue
                if yn in ["n", "N"]:
                    break
                new_ps = add_pathset()
                if new_ps is None:
                    print("")
                    continue
                pathsets.append(new_ps)
        while True:  # Generate the pseudo-labels for the given pathsets.
            yn = input(f"Generate pseudo-labels with pathsets {pathsets} ([y]/n)? ")
            if not yn in ["", "y", "Y", "n", "N"]:
                continue
            if yn in ["n", "N"]:
                print("\nCancelling process.")
                raise KeyboardInterrupt
            break
        generate_pseudolabels(pathsets)
    except KeyboardInterrupt:
        print("")