import numpy as np
import os

print(os.path.abspath(os.getcwd()))
# The new config inherits a base config to highlight the necessary modification
root_path = '/home/hpc/iwi5/iwi5108h/workspace/stable-diffusion-data-augmentation/mmdetection/'
_base_ = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_bounded_iou_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            num_classes=87,
            loss_bbox=dict(type='BoundedIoULoss', loss_weight=10.0))))

# Modify dataset related settings
dataset_type = 'COCODataset'
# classes = ('balloon',)
classes = ('anemone','carnation','columbine','cornflower','daffodil','geranium','heliotrope','hyacinth','iris','jasmine','lavender','lilac','lily',
            'lily of the valley','neroli','petunia','poppy','rose','tulip','violet','other flower','other fruit','apple','cherry','peach','currant',
            'fig','grapes','lemon','melon','pear','plum','strawberry','other vegetable','artichoke','carrot','garlic','mushroom','olive','onion',
            'pumpkin','other vessel','glass with stem','glass without stem','jug','cup','chalice','wine bottle','carafe','coffeepot','teapot',
            'other vertebrate','animal carcass','bird','cat','cow','dog','donkey','fish','goat','horse','pig','sheep','whale','other invertebrate',
            'bivalve','butterfly','caterpillar','fly','lobster','prawn','bug','other jewellery','bracelet','pomander','ring','ashtray',
            'bread','candle','censer','cheese','fire','gloves','meat','nut','pipe','smoke')
# classes = tuple(np.arange(87).astype(str))
data = dict(
    train=dict(
        type='CocoDataset',
        # img_prefix= root_path + 'datasets/ODOR_coco/coco_style/train2017/',
        img_prefix= os.path.join(os.environ.get("TMPDIR"), os.environ.get("SLURM_JOB_ID")) + '/ODOR_coco_generated/coco_style/train2017/',
        classes=classes,
        ann_file= os.path.join(os.environ.get("TMPDIR"), os.environ.get("SLURM_JOB_ID")) + '/ODOR_coco_generated/coco_style/annotations/instances_train2017.json'),
        # ann_file=root_path + 'datasets/ODOR_coco/coco_style/annotations/instances_train2017.json'),
    val=dict(
        type='CocoDataset',
        img_prefix= root_path + 'datasets/ODOR_coco/coco_style/val2017/',
        classes=classes,
        ann_file= root_path + 'datasets/ODOR_coco/coco_style/annotations/instances_val2017.json'),
    test=dict(
        type='CocoDataset',
        img_prefix= root_path + 'datasets/ODOR_coco/coco_style/val2017/',
        classes=classes,
        ann_file= root_path + 'datasets/ODOR_coco/coco_style/annotations/instances_val2017.json'))

runner = dict(type='EpochBasedRunner', max_epochs=100)

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/ddod/ddod_r50_fpn_1x_coco/ddod_r50_fpn_1x_coco_20220523_223737-29b2fc67.pth'
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth' 
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_bounded_iou_1x_coco-98ad993b.pth'

