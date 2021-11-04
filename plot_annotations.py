from os import listdir
# import pandas as pd
# import cv2
# from ast import literal_eval  # CSV bbox as list

import utils

# def plot_annotation(img_path, gt_path, save_path):
#     gender2color = {"man":(255, 0, 0), "woman":(0, 0, 255)}
#     try:
#         raw_img = cv2.imread(img_path)
#         annotated = raw_img.copy()
#     except:
#         if img_path.split("/")[-1][0] != ".":  # if false: hidden file; ignore
#             print("Image not found:", img_path)
#         return False                           # return false: image cannot be drawn
#     gt = pd.read_csv(gt_path, converters={"bbox": literal_eval})
#     for i in range(len(gt)):
#         # Get rectangle coordinates
#         start_point = (gt["bbox"][i][0], gt["bbox"][i][1])
#         end_point = (gt["bbox"][i][2], gt["bbox"][i][3])
#         # Draw the rectangle
#         cv2.rectangle(annotated, start_point, end_point, gender2color[gt["gender"][i]], thickness=3, lineType=cv2.LINE_8) 
#         cv2.putText(annotated, gt["gender"][i], (start_point[0], start_point[1]-5), color=gender2color[gt["gender"][i]], fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = .35)
#         # Save the annotated image
#         cv2.imwrite(save_path, annotated)
#     return True

if __name__ == "__main__":
    img_dir = "data/dataset_train_samples/"
    gt_dir = "data/ground_truth_dataset_train_samples/"
    ann_dir = "data/annotated_train_samples/"

    image_names = [name[:-4] for name in listdir(img_dir)]
    for img_name in image_names:
        img_path = f'{img_dir}{img_name}.jpg'
        gt_path = gt_dir+img_name+".csv"
        save_path = f"{ann_dir}{img_name}.jpg"
        utils.plot_annotation(img_path, gt_path, save_path)
