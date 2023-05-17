##from https://github.com/pytorch/vision/blob/main/references/detection/coco_eval.py

import copy
import os

import torch
import torch.utils.data
import torchvision
# import transforms as T
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        data = ds.get_img_and_bxs(img_idx)
        img = data["img"]
        image_id = data["image_id"]
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[0]
        img_dict["width"] = img.shape[1]
        dataset["images"].append(img_dict)
        bboxes = data["boxes"]
        if len(bboxes) > 0:
            bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = data["labels"]
        areas = data["area"].tolist()
        iscrowd = data["iscrowd"].tolist()
        
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = [bboxes[i][0], bboxes[i][1], bboxes[i][2] - bboxes[i][0], bboxes[i][3]- bboxes[i][1]]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)