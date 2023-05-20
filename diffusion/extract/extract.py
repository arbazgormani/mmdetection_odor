import tarfile
import os
import pandas as pd
from pycocotools.coco import COCO
import json

print("I am extracting the tar file...........")
my_tar = tarfile.open('/home/hpc/iwi5/iwi5108h/workspace/stable-diffusion-data-augmentation/mmdetection/dataset.tar.gz')
save_path = os.path.join(os.environ.get("TMPDIR"), os.environ.get("SLURM_JOB_ID"))
os.makedirs(save_path, exist_ok=True)
print("I am extracting the tar file...........")
my_tar.extractall(save_path)
my_tar.close()
print(os.listdir(save_path))
print("Done. Successfully extracted the dataset.")

ann_path = save_path + '/ODOR_coco_generated/coco_style/annotations/instances_train2017.json'
coco=COCO(ann_path)
ann_df = pd.DataFrame(coco.dataset['annotations'])
ann_df['id'] = ann_df.reset_index()['index'] + 1
coco.dataset['annotations'] = ann_df.to_dict(orient='records')
with open(ann_path, 'w') as f:
    json.dump(coco.dataset, f)

print("hello world!")

