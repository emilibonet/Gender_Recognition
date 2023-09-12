from tqdm import tqdm
import shutil
import json
import os
import utils
from models.ours.initial_design import bbox_pairing, extract_field


class MyTranslator:
    def __init__(self,gt_path,path_to_new_gt):
        self.gt_path = gt_path
        f = open(os.path.join(self.gt_path,'classes.json'))
        self.classes = json.load(f)
        self.dict_children = {'face':{'male':None,'female':None,'young':None,'adult':None,'elder':None},'body':{'male':None,'female':None,'young':None,'adult':None,'elder':None}}
        self.dict_parent = {'face':{'age':None,'gender':None},'body':{'age':None,'gender':None}}
        self.dict_bbox = {'face':None,'body':None}
        self.path_to_new_gt = path_to_new_gt
        self._class_labels()
        
    
    def _class_labels(self):
      
        for class_ in self.classes:
            
            self.dict_bbox[class_['name']] = class_['id']
            for att_parents in class_['attribute_groups']:
                name = att_parents['name']
                id = att_parents['id']
                self.dict_parent[class_['name']][name] = id
                for att_children in att_parents['attributes']:
                    name = att_children['name']
                    id = att_children['id']
                    self.dict_children[class_['name']][name] = id
            

    def _convert2num(self,child, parent):
        if parent == 'age':
            if child == 'young':
                child = 0
            elif child == 'adult':
                child = 1
            else:
                child = 2
        else:
            if child == 'male':
                child = 1
            else:
                child = 0
        return child

    def parse_all_gt(self):
        final_dict = {'face':{'data':[]},'body':{'data':[]}}
        all_images_dict = {}
        for file in os.listdir(self.gt_path):
            final_dict = {'face':{'data':[]},'body':{'data':[]}}
            if '.json' in file and file != 'classes.json':
                f = open(os.path.join(self.gt_path,file))
                loaded_json = json.load(f)
                for instance in loaded_json['instances']:
                    temp_dict = {'box':None,'gender':None,'age':None}.copy()
                    bbox = [instance['points']['x1'],instance['points']['y1'],instance['points']['x2'],instance['points']['y2']]
                    id = instance['classId']
                    name = list(self.dict_bbox.keys())[list(self.dict_bbox.values()).index(id)]
                    temp_dict['box'] = bbox
                    
                    for attributes in instance['attributes']:
                        child_id = attributes['id']
                        parent_id = attributes['groupId']
                       
                        child = list(self.dict_children[name].keys())[list(self.dict_children[name].values()).index(child_id)]
                        parent = list(self.dict_parent[name].keys())[list(self.dict_parent[name].values()).index(parent_id)]
                        
                        child = self._convert2num(child,parent)
                        temp_dict[parent] = child
                   
                    final_dict[name]['data'].append(temp_dict)
                all_images_dict[file] = final_dict

        anndict = {}
        count_flawed = 0
        for imgname in all_images_dict.keys():
            imgpath = os.path.join(self.gt_path, imgname.strip(".json"))
            fbb = extract_field(all_images_dict[imgname]['face']['data'], "box")
            bbb = extract_field(all_images_dict[imgname]['body']['data'], "box")
            pairs = bbox_pairing(fbb, bbb, iou_thrs=1e-3)
            for face_idx, body_idx in pairs:
                gender = age = None
                if face_idx is not None and body_idx is not None:
                    # Manage nulls and mismatches for gender labels
                    if all_images_dict[imgname]['face']['data'][face_idx]['gender'] is None:
                        if all_images_dict[imgname]['body']['data'][body_idx]['gender'] is None:
                            count_flawed += 1
                            continue
                        else:
                            gender = all_images_dict[imgname]['body']['data'][body_idx]['gender']
                    elif all_images_dict[imgname]['body']['data'][body_idx]['gender'] is None:
                        gender = all_images_dict[imgname]['face']['data'][face_idx]['gender']
                    elif all_images_dict[imgname]['face']['data'][face_idx]['gender'] != all_images_dict[imgname]['body']['data'][body_idx]['gender']:  # both are not none, but mismatch
                        count_flawed += 1
                        continue
                    else:   # both are not none and there is no mismatch
                        gender = all_images_dict[imgname]['face']['data'][face_idx]['gender']
                    
                    # Manage nulls and mismatches for age labels
                    if all_images_dict[imgname]['face']['data'][face_idx]['age'] is None:
                        if all_images_dict[imgname]['body']['data'][body_idx]['age'] is None:
                            count_flawed += 1
                            continue
                        else:
                            age = all_images_dict[imgname]['body']['data'][body_idx]['age']
                    elif all_images_dict[imgname]['body']['data'][body_idx]['age'] is None:
                        age = all_images_dict[imgname]['face']['data'][face_idx]['age']
                    elif all_images_dict[imgname]['face']['data'][face_idx]['age'] != all_images_dict[imgname]['body']['data'][body_idx]['age']:
                        count_flawed += 1
                        continue
                    else:
                        age = all_images_dict[imgname]['face']['data'][face_idx]['age']
                elif face_idx is not None:
                    if all_images_dict[imgname]['face']['data'][face_idx]['gender'] is None or all_images_dict[imgname]['face']['data'][face_idx]['age'] is None:
                        count_flawed += 1
                        continue
                    gender = all_images_dict[imgname]['face']['data'][face_idx]['gender']
                    age = all_images_dict[imgname]['face']['data'][face_idx]['age']
                else:
                    if all_images_dict[imgname]['body']['data'][body_idx]['gender'] is None or all_images_dict[imgname]['body']['data'][body_idx]['age'] is None:
                        count_flawed += 1
                        continue
                    gender = all_images_dict[imgname]['body']['data'][body_idx]['gender']
                    age = all_images_dict[imgname]['body']['data'][body_idx]['age']
                assert(age is not None and gender is not None)
                ann = {'fbb':[max(round(n), 0) for n in all_images_dict[imgname]['face']['data'][face_idx]['box']] if face_idx is not None else None,
                       'bbb':[max(round(n), 0) for n in all_images_dict[imgname]['body']['data'][body_idx]['box']] if body_idx is not None else None,
                       'gender':gender,
                       'age':age}
                if imgpath not in anndict.keys():
                    anndict[imgpath] = []
                anndict[imgpath].append(ann)
        out_file = open(self.path_to_new_gt, "w")  
        json.dump(anndict, out_file, indent = 4)
        out_file.close()
        return anndict, count_flawed

if __name__ == "__main__":
    # Transforms the data outputted by the curation process back to the format previously used for the data synthesis.
    # The json with the curated and reformatted data is saved to the annotations of the pathset in question.
    total_count = 0
    for seq in tqdm(os.listdir(os.path.join(utils.root, "data/body/curated"))):
        ds_name = seq.strip('_resampled-fin')
        newdir = os.path.join(utils.data,"body", ds_name)
        if os.path.exists(newdir):
            shutil.rmtree(newdir)
        os.mkdir(newdir)
        if not os.path.exists(os.path.join(newdir, "annotations")):
            os.mkdir(os.path.join(newdir, "annotations"))
        translator = MyTranslator(os.path.join(os.path.join(utils.root, "data/body/curated"), seq), os.path.join(newdir, "annotations/curated.json"))
        _, count_flawed = translator.parse_all_gt()
        total_count += count_flawed
    print(f"A total of {total_count} defective samples were removed.")