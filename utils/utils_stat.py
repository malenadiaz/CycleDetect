import fiftyone as fo
import fiftyone.utils.coco as fouc
import fiftyone.utils.eval as fue
import torch
import os 
from datetime import datetime
import re
import matplotlib.pyplot as plt

def convert_to_fityone(ds):
    samples = []
    label_map = ds.get_labels()
    for img_idx in range(len(ds)):
        data = ds.get_img_and_bxs(img_idx)

        sample = fo.Sample(filepath=os.path.join(os.getcwd(), ds.img_folder, ds.img_list[img_idx]))
       
        img = data['img']
        w = img.shape[1]
        h = img.shape[0]

        labels = data["labels"]
        bboxes = data["boxes"].tolist()
        num_objs = len(bboxes)
        detections = []
        for i in range(num_objs):
            label = labels[i]
            bbox = bboxes[i]
            bounding_box = [bbox[2] /w , bbox[3]/h, (bbox[2] - bbox[0])/w, (bbox[3]- bbox[1])/h]
            detections.append(
            fo.Detection(label=label_map[label], bounding_box=bounding_box)
            )
        sample["ground_truth"] = fo.Detections(detections=detections)
        samples.append(sample)
    # Create dataset
    dataset = fo.Dataset()
    dataset.add_samples(samples)
    return dataset

def convert_torch_predictions(preds, det_id, s_id, w, h, classes):
    # Convert the outputs of the torch model into a FiftyOne Detections object
    dets = []
    for bbox, label, score in zip(
        preds["boxes"], 
        preds["labels"], 
        preds["scores"]
    ):
        # Parse prediction into FiftyOne Detection object
        x0,y0,x1,y1 = bbox
        coco_obj = fouc.COCOObject(det_id, s_id, int(label), [x0, y0, x1-x0, y1-y0])
        det = coco_obj.to_detection((w,h), classes)
        det["confidence"] = float(score)
        dets.append(det)
        det_id += 1
        
    detections = fo.Detections(detections=dets)
        
    return detections, det_id

def add_detections( dataset, view, pred, img_path):
    # Run inference on a dataset and add results to FiftyOne
    classes = list(dataset.get_labels().values())
    det_id = 0


    img_path = os.path.join(os.getcwd(), img_path)
    img_path = re.sub(r"(/[^/]+?)/\.\./", "/", img_path)
    sample = view[img_path]
    s_id = sample.id
    w = 600
    h = 300
    detections, det_id = convert_torch_predictions(
            pred["pred"],
            det_id, 
            s_id, 
            w, 
            h, 
            classes,
        )
                    
    sample["predictions"] = detections
    sample.save()

def evaluate_51(ds, preds,output_dir):
    print(ds.img_folder)
    fo_dataset = convert_to_fityone(ds)

    for pred in preds:
        add_detections(ds, fo_dataset, preds[pred], pred)
    results = fue.evaluate_detections(
    fo_dataset, 
    "predictions", 
    classes=list(ds.get_labels().values()), 
    eval_key="eval", 
    compute_mAP=True
    )
    plot = results.plot_confusion_matrix(backend='matplotlib')
    plot.savefig(os.path.join(output_dir, "Confusion matrix.png"))
    plot = results.plot_pr_curves(backend='matplotlib')
    plot.savefig(os.path.join(output_dir, "PR Curve.png"))