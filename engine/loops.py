import sys
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Tuple

from engine.forwards import run_forward
from data.USKpts import USKpts
from utils.utils_files import AverageMeter
from utils.coco_utils import get_coco_api_from_dataset
from utils.coco_eval import CocoEvaluator
########################################################
########################################################
# Run single epoch for Train/Validate/Eval
########################################################
########################################################
def train_vanilla(epoch: int,
                  loader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  model: torch.nn.Module,
                  device: torch.device,
                  criterion: torch.nn.Module,
                  prossesID: int = None
                  ) -> Tuple[int, list]:
    """vanilla training"""

    model.train()
    prefix = 'Training'
    if prossesID is not None:
        prefix = "[{}]{}".format(prossesID, prefix)

    losses = {"main": AverageMeter(), "bxs":AverageMeter(), 
              "bbox_loss":AverageMeter(), "class_loss":AverageMeter(), "object_loss": AverageMeter(),
               "rpn_loss": AverageMeter() }

    with tqdm(total=len(loader), ascii=True, desc=('{}: {:02d}'.format(prefix, epoch))) as pbar:
        for batch_i, data in enumerate(loader, 0):

            # ================= Extract Data ==================
            filenames = data[2]

            # =================== forward =====================
            batch_loss, batch_output = run_forward(model, data, criterion, device)

            # =================== backward ====================
            if optimizer is not None:
                optimizer.zero_grad()
                batch_loss["loss"].backward()
                optimizer.step()

            pbar.update()
            # for dat_name, dat in batch_output.items():
            #     to_numpy(dat)

            # accumulate losses:
            losses["main"].update(batch_loss["reported_loss"], len(filenames))
            if "reported_loss" in batch_loss:
                losses["bxs"].update(batch_loss["reported_loss"], len(filenames))
            if "bbox_loss" in batch_loss:
                losses["bbox_loss"].update(batch_loss["bbox_loss"], len(filenames))
            if "class_loss" in batch_loss:
                losses["class_loss"].update(batch_loss["class_loss"], len(filenames))
            if "object_loss" in batch_loss:
                losses["object_loss"].update(batch_loss["object_loss"], len(filenames))
            if "rpn_loss" in batch_loss:
                losses["rpn_loss"].update(batch_loss["rpn_loss"], len(filenames))
    return losses


def validate(mode: str,
             epoch: int,
             loader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             device: torch.device,
             criterion: torch.nn.Module,
             prossesID: int = None
             ) -> Tuple[int, list]:
    """validation"""

    model.eval()
    if mode == 'validation':
        prefix = 'Validating'
    elif mode == 'test':
        prefix = 'Testing'
    if prossesID is not None:
        prefix = "[{}]{}".format(prossesID, prefix)

    cpu_device = torch.device("cpu")

    inputs, outputs = dict(), dict()
    maps = { }

    coco = get_coco_api_from_dataset(loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    with tqdm(total=len(loader), ascii=True, desc=('{}: {:02d}'.format(prefix, epoch))) as pbar:
        for batch_i, data in enumerate(loader, 0):

            # ================= Extract Data ==================
            filenames = data[2]
            
            # =================== forward =====================
            batch_loss, batch_output = run_forward(model, data, criterion, device)

            pbar.update()
            

            boxes_pred = [{k: v.to(cpu_device) for k, v in t.items()} for t in batch_output["boxes_pred"]]
            boxes_gt = [{k: v.to(cpu_device) for k, v in t.items()} for t in batch_output["boxes_gt"]]

            res = {target["image_id"].item(): output for target, output in zip(data[1], boxes_pred)}
            coco_evaluator.update(res)
            # for dat_name, dat in batch_output.items():
            #     if dat_name in ["boxes_pred","boxes_gt"]:
            #         numpy_dat = []
            #         for img in dat:
            #             img_data = {}
            #             for k,v in img.items():
            #                 img_data[k] = to_numpy(v)
            #             numpy_dat.append(img_data)
            #     else:
            #         numpy_dat = []
            #         for img in dat:
            #             numpy_dat.append(to_numpy(img))
            #     batch_output[dat_name] = numpy_dat
            # accumulate losses:
            # losses["main"].update(stats[0], len(filenames))
            # if "reported_loss" in batch_loss:
            #     losses["bxs"].update(batch_loss["reported_loss"], len(filenames))

            # accumulate outputs nad inputs:
            for ii, filename in enumerate(filenames):
                inputs[filename] = boxes_gt[ii]
                outputs[filename] = boxes_pred[ii]

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()
    maps["MAP@0.5.0.95"] = stats[0]
    maps["MAP@0.5"] = stats[1]
    maps["MAP@0.75"] = stats[2]
    maps["MAP@0.5.0.95.s"] = stats[3]
    maps["MAP@0.5.0.95.m"] = stats[4]
    maps["MAP@0.5.0.95.l"] = stats[5]

    return maps, outputs, inputs

########################################
########################################
# Sample dataset
########################################
########################################
def sample_dataset(trainset: USKpts, valset: USKpts, testset: USKpts, overfit: bool, batch_size: int = 8, num_workers: int = 8):
    if overfit:  # sample identical very few examples for both train ans val sets:
        num_samples_for_overfit = 10
        annotated = np.random.choice(np.arange(trainset.num_of_annotated_files), num_samples_for_overfit)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(annotated),
                                                  shuffle=False, pin_memory=True, collate_fn=collate_fn)
        valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(annotated),
                                                 shuffle=False, pin_memory=True, collate_fn=collate_fn)
        print("DATA: Sampling identical sets of {} ANNOTATED examples for train and val sets.. ".format(num_samples_for_overfit))

    else:
        # --- Train: ---
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=num_workers, collate_fn=collate_fn)
        # --- Val: ---
        if valset is not None:
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                     shuffle=False, pin_memory=True,
                                                     num_workers=num_workers, collate_fn=collate_fn)

    # --- Test: ---
    if testset is not None:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, pin_memory=True,
                                                num_workers=num_workers, collate_fn=collate_fn)
    else:
        testloader = []

    return trainloader,valloader, testloader



def collate_fn(batch):
    return tuple(zip(*batch))