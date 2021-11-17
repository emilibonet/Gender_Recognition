from os import listdir
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

import utils

def parse_xml(path_to_xml):
    root = ET.parse(path_to_xml).getroot()
    rows = []
    for object in root.findall("object"):
        gender = object.find("name").text
        xmin, ymin = int(object.find("bndbox").find("xmin").text), int(object.find("bndbox").find("ymin").text)
        xmax, ymax = int(object.find("bndbox").find("xmax").text), int(object.find("bndbox").find("ymax").text)
        rows.append({"gender":gender, "bbox":[xmin, ymin, xmax, ymax]})
    return rows


if __name__ == "__main__":
    gt_dir = utils.root + "data/annotations/ground_truth/xmls/"

    print("Starting conversion...")
    for xml_file in tqdm(listdir(gt_dir+"xmls/")):
        rows = parse_xml(gt_dir+"xmls/"+xml_file)
        df = pd.DataFrame(rows)
        df.to_csv(f'{gt_dir}{xml_file[:-4]}.csv', index=False)
    print("Conversion ended successfully.")
