import tarfile
import os
import json
from pycocotools.coco import COCO


target_dir = '/home/janus/iwi5-datasets/synthetic-odor'
target_folder = 'extracted'
filenmaes = os.listdir(target_dir)
dataset_name = 'ODOR_coco_generated'
for i, fn in enumerate(filenmaes):
    if 'tar.gz' in fn:
        if i > 34:
            break
        print(f"{i+1} Extracting: ", os.path.join(target_dir, fn))
        my_tar = tarfile.open(os.path.join(target_dir, fn))
        my_tar.extractall(os.path.join(target_dir, target_folder))
        my_tar.close()
ann_path = os.path.join(os.path.join(target_dir, target_folder), os.path.join(dataset_name, 'coco_style/annotations'))
ann_files = os.listdir(ann_path)
coco=COCO(os.path.join(ann_path, ann_files[0]))
coco.dataset['images'] = []
coco.dataset['annotations'] = []
for ann_file in ann_files:
    temp_coco = COCO(os.path.join(ann_path, ann_file))
    coco.dataset['images'] += temp_coco.dataset['images']
    coco.dataset['annotations'] += temp_coco.dataset['annotations']

with open(os.path.join(ann_path, f'instances_train2017.json'), 'w') as f:
    json.dump(coco.dataset, f)

print("Done...")