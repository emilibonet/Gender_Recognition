from os import listdir
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm


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
    path = "./data/ground_truth_dataset_train_samples/"

    print("Starting conversion...")
    for xml_file in tqdm(listdir(path+"xml_format/")):
        rows = parse_xml(path+"xml_format/"+xml_file)
        df = pd.DataFrame(rows)
        df.to_csv(f'{path}{xml_file[:-4]}.csv', index=False)
    print("Conversion ended successfully.")
