import shutil
import utils
import os

def annotation_query():
    """Queries the user about multiple possible annotation files that need to be included in the pathset.

    ---
    Returns:
        (void)
    """
    ann_files = []
    while True:
        yn_ann = input("Annotation file ([y]/n)? ")
        if not yn_ann in ["", "y", "Y", "n", "N"]:
            continue
        if yn_ann in ["n", "N"]:
            return ann_files
        ann_file = input("Annotation file: ")
        if os.path.isfile(ann_file):
            ann_files.append(ann_file)
        else:
            print(f"File {ann_file} not found.")


def create_pathset(ds_name, ds_type, img_folder, ann_files):
    """Creates a pathset that is allocated to a directory defined based on the name and type provided, with an image file containing the paths to the images of the original dataset, as well as an annotations directory with the annotation file(s), if there are any.
    
    ---
    Arguments:
        - ds_name: string with name of the original dataset (which will be used to name the pathset accordingly).
        - ds_type: string with either 'face' or 'body' depending on if the dataset is about face or body data.
        - img_folder: folder containing the images from the original dataset; used to extract all the paths to these images.

    Returns:
        (void)
    """
    while True:
        goahead = input(f"Create pathset for dataset {ds_name} of type {ds_type}, with image folder {img_folder} and annotation files {ann_files} ([y]/n)? ")
        if goahead in ["", "y", "Y", "n", "N"]:
            break
    if goahead in ["n", "N"]:
        print("Cancelling creation.")
        return
    newdir = f"{utils.data}{ds_type}/{ds_name}/"
    if os.path.exists(newdir):
        shutil.rmtree(newdir)
    os.mkdir(newdir)
    images = [img_folder+"/"+imgname for imgname in os.listdir(img_folder) if imgname[-3:] in ["png", "jpg"]]

    with open(newdir + "imgpaths.txt", "w") as file:
        file.write("\n".join(images))
    if len(ann_files):
        os.mkdir(newdir + "annotations/")
        for ann_file in ann_files:
            shutil.copy(ann_file, newdir + "annotations/" + ann_file.split("/")[-1])
    


def main():
    while True:   # for each loop, define and create a new pathset
        try:
            ds_name = input("Dataset name: ")
            if not len(ds_name):
                break
            
            while True:
                ds_type = input("Dataset type (face/body): ")
                if ds_type in ["face", "body"]:
                    break
                else:
                    print("Dataset type must be either 'face' or 'body'.")
            
            img_folder = input("Image folder: ")
            while not os.path.isdir(img_folder):
                print(f"Folder {img_folder} not found.")
                img_folder = input("Image folder: ")
            ann_files = annotation_query()
            create_pathset(ds_name, ds_type, img_folder, ann_files)
        except KeyboardInterrupt:
            print("")
            return

if __name__ == "__main__":
    main()
    
        