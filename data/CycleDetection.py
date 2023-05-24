import torch
import os
import numpy as np
import cv2
import albumentations as A
from PIL import Image
import glob
from typing import List, Dict
import matplotlib.pyplot as plt
import torchvision.transforms as torch_transforms
from utils.plot import plot_img
# internal:
from torch.utils import data

class CycleDetection(data.Dataset):
    def __init__(self, dataset_config, filenames_list: str = None, transform: A.core.composition.Compose = None):
        self.img_folder = dataset_config["img_folder"]
        self.transform = transform
        self.input_size = dataset_config["input_size"]

        # get list of files in dataset:
        self.create_img_list(filenames_list=filenames_list)

        # get kpts annotations
        self.anno_dir = dataset_config["anno_folder"]
        self.BOX_COORDS, self.LABELS = self.load_box_annotations(self.img_list)
        # basic transformations:
        self.basic_transform = torch_transforms.Compose([
            torch_transforms.ToTensor(),
            torch_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # to 0-1 values
            
    def create_img_list(self, filenames_list: str) -> None:
        """
        Called during construction. Creates a list containing paths to frames in the dataset
        """
        img_list_from_file = []
        with open(filenames_list) as f:
            img_list_from_file.extend(f.read().splitlines())
        self.img_list = img_list_from_file
    
    def load_box_annotations(self, img_list: List) -> np.ndarray:
        """ Creates an array of annotated keypoints coorinates for the frames in the dataset. """
        BOX_COORDS = []
        LABELS = []
        if self.anno_dir is not None:
            for fname in img_list:
                annot_dir = np.load(os.path.join(self.anno_dir, fname.replace("png", "npy")), allow_pickle=True)
                annot_dir = annot_dir.item()
                BOX_COORDS.append(annot_dir['bbox'])
                LABELS.append(annot_dir['label'])
        return BOX_COORDS, LABELS

    def img_to_torch(self, img: np.ndarray) -> torch.Tensor:
        """ Convert original image format to torch.Tensor """
        # resize:
        if img.shape[0] != self.input_size:
            img = cv2.resize(img, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR_EXACT)
        # transform:
        img = Image.fromarray(np.uint8(img))
        img = self.basic_transform(img)

        return img

    def get_img_and_bxs(self, index: int):
        """
        Load and parse a single data point.
        Args:
            index (int): Index
        Returns:
            img (ndarray): RGB frame in required input_size
            kpts (ndarray): Denormalized, namely in img coordinates
            img_path (string): full path to frame file in image format (PNG or equivalent)
        """
        # ge paths:
        img_path = os.path.join(self.img_folder, self.img_list[index])
        
        # get image: (PRE-PROCESS UNIQUE TO UltraSound data)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #load boxes and convert them to tensors 
        boxes = self.BOX_COORDS[index].astype(int)
        #boxes = self.normalize_pose(pose=boxes, frame=img)

        labels = [self.LABELS[index]] * len(boxes)
        
        boxes_length = len(boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)

        data = {"img": img,
                "boxes": boxes,
                "img_path": img_path,
                "labels": labels,
                "area":area,
                "iscrowd":iscrowd, 
                "image_id":index + 1
                }
        return data


    def normalize_pose(self, pose: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """ Normalizing a set of frame keypoint to [0, 1] """
        pose = pose.astype("float")
        for p in pose:
            p[0] = p[0] / frame.shape[1]
            p[1] = p[1] / frame.shape[0]
            p[2] = p[2] / frame.shape[1]
            p[3] = p[3] / frame.shape[0]
        return pose 
    
    def denormalize_pose(self, pose: np.ndarray, frame: np.ndarray) -> np.ndarray:

        """ DeNormalizing a set of frame keypoint back to image coordinates """
        for p in pose:
            p[0] = p[0] * frame.shape[1]
            p[1] = p[1] * frame.shape[0]
            p[2] = p[2] * frame.shape[1]
            p[3] = p[3] * frame.shape[0]
        return pose       

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = self.get_img_and_bxs(index)

        if self.transform is not None:
            transformed = self.transform(image=data["img"], bboxes=data["boxes"], labels = data["labels"])
            img = transformed['image']
            boxes = transformed['bboxes']
        else:
            img = data["img"]
            boxes = data["boxes"]
        
        #boxes = self.normalize_pose(pose=boxes, frame=img)

        # transform:
        img = Image.fromarray(np.uint8(img))
        img = self.basic_transform(img) #normalize to 0 1 values

        target = {}
        #boxes
        boxes = torch.tensor(boxes).float() #convert box annotations to tensors 
        target["boxes"] = boxes
        target["labels"] = torch.as_tensor(data["labels"],dtype=torch.int64 )
        target["image_id"] = torch.tensor(data["image_id"])
        target["area"] = torch.as_tensor(data["area"])
        target["iscrowd"] = data["iscrowd"]

        return img, target, data["img_path"]
    
    def plot_item(self, index: int, do_augmentation: bool = True, print_folder: str = './visu/') -> None:
        """ Plot frame and gt annotations for a single data point """
        data = self.get_img_and_bxs(index)
        print_fname = "".join(data["img_path"].split("/")[-1].split(".")[:-1])

        # data aug:
        if do_augmentation and self.transform is not None:
            transformed = self.transform(image=data["img"], bboxes=data["boxes"], labels = data["labels"])
            img = transformed['image']
            boxes = transformed['bboxes']
            print_fname = "{}_aug".format(print_fname)
        else:
            img = data["img"]
            boxes = data["boxes"]

        # plot:
        plt.clf()
        boxes = self.denormalize_pose(np.array(boxes), img)
        plot_img(img, boxes)
        nnm = os.path.join(print_folder, print_fname)
        plt.savefig(nnm + ".png")
        print(nnm)

    def get_labels(self):
        return {1:"Ute Art.", 2:"Aor. Isth.",3: "Duct. Ven.", 
                4:"L.Ventr In/Out", 5:"Umb Art.",
                6:"Mid. Cere. Art." }                              