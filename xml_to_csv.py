import utils
from os import listdir
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    gt_dir = utils.root + "data/annotations/ground_truth/"

    print("Starting conversion...")
    for xml_file in tqdm(listdir(gt_dir+"xmls/")):
        rows = utils.parse_xml(gt_dir+"xmls/"+xml_file)
        df = pd.DataFrame(rows)
        df.to_csv(f'{gt_dir}{xml_file[:-4]}.csv', index=False)
    print("Conversion ended successfully.")
