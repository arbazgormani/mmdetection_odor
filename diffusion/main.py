import requests
from io import BytesIO
import PIL
import matplotlib
import torch
import shutil

import json
import os
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib import cm
import skimage.io as io
import cv2
import copy
import uuid
import tarfile

path = "/home/hpc/iwi5/iwi5108h/workspace/stable-diffusion-data-augmentation/mmdetection/datasets/ODOR_coco/coco_style/"
ann_path = os.path.join(path, "annotations/instances_train2017.json")
coco=COCO(ann_path)

dataset_name = 'ODOR_coco_generated'
target_dir = '/home/janus/iwi5-datasets/synthetic-odor'
# target_img_dir = os.path.join(target_dir, dataset_name, 'coco_style/train2017')
# target_ann_dir = os.path.join(target_dir, dataset_name, 'coco_style/annotations')
# os.makedirs(target_img_dir, exist_ok=True)
# os.makedirs(target_ann_dir, exist_ok=True)

# print(target_img_dir)
# print(target_ann_dir)

# if os.path.isfile(os.path.join(target_ann_dir, 'instances_train2017.json')):
#     new_coco_obj=COCO(os.path.join(target_ann_dir, 'instances_train2017.json'))
# else:
#     new_coco_obj = copy.deepcopy(coco)
#     new_coco_obj.dataset['images'] = []
#     new_coco_obj.dataset['annotations'] = []

save_path = os.path.join(os.environ.get("TMPDIR"), os.environ.get("SLURM_JOB_ID"))
img_save_path = os.path.join(save_path, dataset_name, 'coco_style/train2017')
ann_save_path = os.path.join(save_path, dataset_name, 'coco_style/annotations')
os.makedirs(img_save_path, exist_ok=True)
os.makedirs(ann_save_path, exist_ok=True)

print(save_path)
print(img_save_path)
print(ann_save_path)


def create_image(data_path, coco_obj, new_coco_obj, start_idx, end_idx, train_path='train2017'):
    
    image_id = str(uuid.uuid4())
    ann_id = len(new_coco_obj.dataset['annotations']) + 1 
    canvas = np.zeros((512,512,3))
    mask = np.ones((512,512,3))*255
    canvas_info = {
        'id': image_id,
        'width': canvas.shape[0],
        'height': canvas.shape[1],
        'file_name': f'{image_id}.png',
        'license': 0,
        'flickr_url': '',
        'coco_url': '',
        'date_captured': 0
    }
    bounding_boxes = []
    annotations = []
    images = []
    curr_x, curr_y = np.random.randint(0,20), np.random.randint(0,20)
    max_y = 0
    for i in range(start_idx, end_idx):
        img = cv2.imread(os.path.join(data_path, os.path.join(train_path, coco_obj.imgs[i]['file_name'])))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotations += coco_obj.imgToAnns[i]
        for i in range(len(coco_obj.imgToAnns[i])):
            images.append(img)
    merged = list(zip(images, annotations))
    random.shuffle(merged)
    images, annotations = zip(*merged)
    new_coco_obj.dataset['images'].append(canvas_info)
    for i, img in enumerate(images):
        ann = annotations[i]
        x,y,width,height = map(int, ann['bbox'])
        num = np.random.randint(0,20)
        if num+curr_x+width <= canvas.shape[0]:
            if num+curr_y+height > canvas.shape[1]:
                break
#                 num = np.random.randint(0,20)
            canvas[num+curr_y:num+curr_y+height, curr_x:curr_x+width] =  img[y:y+height, x:x+width]
            mask[num+curr_y:num+curr_y+height, curr_x:curr_x+width] =  0
            bbox = [curr_x, num+curr_y, width, height]
            bounding_boxes.append(bbox)
            object_ann = copy.deepcopy(ann)
            object_ann['id'] = ann_id
            object_ann['image_id'] = image_id
            object_ann['bbox'] = bbox
            new_coco_obj.dataset['annotations'].append(object_ann)
            ann_id += 1
            curr_x = curr_x + width + np.random.randint(10,100)
            if max_y < num+curr_y+height:
                max_y = num+curr_y + height + np.random.randint(10, 30)
            if curr_x >= canvas.shape[0]:
                curr_x = 0
                curr_y = max_y #+ np.random.randint(10,100)
            if curr_y >= canvas.shape[1]:
                print("Exiting the Code...")
                return canvas, mask, image_id
        else:
            curr_x = 0
            curr_y = max_y #+ np.random.randint(10,100)
    return canvas, mask, image_id


def generate_data(save_path, coco_obj, new_coco_obj, pipe, start, end, step, prompt="historical oil painting"):
    for i in range(start, end+1):
        if i+step <= end+1:
            canvas, mask, image_id = create_image(path, coco_obj, new_coco_obj, i, i+step)
            # canvas = Image.fromarray(canvas).convert('RGB')
            mask = Image.fromarray(np.uint8(mask))
            # canvas = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_BGR2RGB)
            # image = pipe(prompt=prompt, image=canvas, mask_image=mask).images[0]
            image = pipe(prompt=prompt, image=canvas, mask_image=mask, guidance_scale=7.5, generator=torch.Generator(device="cuda").manual_seed(0)).images[0]
            # image = image.convert('RGB') 
            image = np.array(image) 
            # Convert RGB to BGR 
            # image = image[:, :, ::-1].copy() 

            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # canvas = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_BGR2RGB)
            
            # image = image[:,:,::-1].copy()
            # image.save(os.path.join(save_path, f"{image_id}.png"))
            cv2.imwrite(os.path.join(save_path, f"{image_id}.png"), image)
            # cv2.imwrite(os.path.join(save_path, f"original_{image_id}.png"), canvas)
            # cv2.imwrite(os.path.join(save_path, f"mask_{image_id}.png"), np.array(mask))
        else:
            print("I am returning")
            return   


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float32,
)
prompt = "historical oil painting"
pipe = pipe.to("cuda")

step = 5
for i in range(3):
    print("I am here in writing the file with step: ", step)
    os.makedirs(img_save_path, exist_ok=True)
    # os.makedirs(img_save_path, exist_ok=True)
    new_coco_obj = copy.deepcopy(coco)
    new_coco_obj.dataset['images'] = []
    new_coco_obj.dataset['annotations'] = []
    generate_data(img_save_path, coco, new_coco_obj, pipe, 1, len(coco.dataset['images']), step)
    with open(os.path.join(ann_save_path, f'instances_train2017_{step}.json'), 'w') as f:
        json.dump(new_coco_obj.dataset, f)
    make_tarfile(os.path.join(save_path, dataset_name)+f'_{step}.tar.gz', os.path.join(save_path, dataset_name))
    shutil.move(os.path.join(save_path, dataset_name)+f'_{step}.tar.gz', target_dir)
    shutil.rmtree(img_save_path)
    os.remove(os.path.join(ann_save_path, f'instances_train2017_{step}.json'))
    step += 1

# img_save_path = './results/1'
# shutil.rmtree(img_save_path)
# os.makedirs(img_save_path, exist_ok=True)
# new_coco_obj = copy.deepcopy(coco)
# new_coco_obj.dataset['images'] = []
# new_coco_obj.dataset['annotations'] = []
# generate_data(img_save_path, coco, new_coco_obj, pipe, 1, 5, 2)


