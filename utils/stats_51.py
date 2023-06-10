import fiftyone as fo
import fiftyone.utils.coco as fouc
import fiftyone.utils.eval as fue
import torch
import os 
from datetime import datetime
import re
import matplotlib.pyplot as plt
from fiftyone import ViewField as F
from utils.utils_stat import filter_nms
from torchvision.ops import nms
from utils.utils_files import to_numpy
import sys

def convert_to_fityone(ds):
    samples = []
    label_map = ds.get_labels()
    for img_idx in range(len(ds)):
        data = ds.get_img_and_bxs(img_idx)
        f = os.path.join(os.getcwd(), ds.img_folder, ds.img_list[img_idx])
        sample = fo.Sample(filepath=f)
       
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
            bounding_box = [bbox[0] /w , bbox[1]/h, (bbox[2] - bbox[0])/w, (bbox[3]- bbox[1])/h]
            detections.append(
            fo.Detection(label=label_map[label], bounding_box=bounding_box)
            )
        sample["ground_truth"] = fo.Detections(detections=detections)
        samples.append(sample)

    # Create dataset
    dataset = fo.Dataset()
    dataset.add_samples(samples)
    return dataset

def convert_torch_predictions(preds, det_id, s_id, w, h, classes, nms_t=None):
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
        det["nms"] = nms_t
        dets.append(det)
        det_id += 1
        
    detections = fo.Detections(detections=dets)
        
    return detections, det_id

def add_detections( dataset, view, pred, img_path, nms_t=None, key="predictions"):
    # Run inference on a dataset and add results to FiftyOne
    classes = list(dataset.get_labels().values())
    det_id = 0


    img_path = os.path.join(os.getcwd(), img_path)
    img_path_prev = ""
    while img_path_prev != img_path:
        img_path_prev = img_path
        img_path = re.sub(r"(/[^/]+?)/\.\./", "/", img_path)
    sample = view[img_path]
    s_id = sample.id
    w = 600
    h = 300
    detections, det_id = convert_torch_predictions(
            pred,
            det_id, 
            s_id, 
            w, 
            h, 
            dataset.get_labels(),
            nms_t
        )
    sample[key] = detections
    sample.save()

def evaluate_51(fo_dataset, ds, predictions, output_dir, nms_t, conf):
    torch_preds ={t.split('/')[-1]:{k: to_numpy(v) for k, v in predictions[t]["pred"].items()} for t in predictions}
    classes = list(ds.get_labels().values())
    for filename in torch_preds:
        add_detections(ds, fo_dataset, torch_preds[filename], os.path.join(ds.img_folder, filename), nms_t )

    results = fue.evaluate_detections(
                    fo_dataset, 
                    "predictions", 
                    classes=list(ds.get_labels().values()), 
                    eval_key="eval", 
                    classwise=False, missing="No Object",
                    compute_mAP=True
            )

    plot = results.plot_confusion_matrix(backend='matplotlib')
    plot.savefig(os.path.join(output_dir, "Confusion matrix_NMS{}_C{}.png".format(nms_t,conf)))
    plot = results.plot_pr_curves(classes=classes, backend='matplotlib')
    plot.savefig(os.path.join(output_dir, "PR Curve_NMS{}_C{}.png".format(nms_t,conf)))

    FP_frames = []
    FN_frames = []
    fp_key = "eval" + "_fp"
    fn_key = "eval" + "_fp"

    for sample in fo_dataset:
        if sample[fp_key] > 0:
            FP_frames.append(sample.filepath.split("/")[-1])
        if sample[fn_key] > 0:
            FN_frames.append(sample.filepath.split("/")[-1])
    print("Number of frames with FP:{}. Sample:{}".format(len(FP_frames), FP_frames[0]))
    print("Number of frames with FP:{}. Sample:{}".format(len(FN_frames), FN_frames[0]))

    original_stdout = sys.stdout 	

    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        sys.stdout = f
        # Print a classification report for the top-10 classes
        results.print_report(classes=classes)

        # Print some statistics about the total TP/FP/FN counts
        print("TP: %d" % fo_dataset.sum("eval_tp"))
        print("FP: %d" % fo_dataset.sum("eval_fp"))
        print("FN: %d" % fo_dataset.sum("eval_fn"))

        sys.stdout = original_stdout

    return FP_frames, FN_frames

def evaluate_51_NMS(fo_dataset, ds, predictions, output_dir):

    NMS_THRESH = [0.4]
    CONF_THRESH = [0.5]
    
    torch_preds ={t.split('/')[-1]:{k: v for k, v in predictions[t]["pred"].items()} for t in predictions}

    prediction_keys = []
    for nms_t in NMS_THRESH:
        nms_key = "predictions_{}".format(int(nms_t * 10) )
        prediction_keys.append(nms_key)
        for filename in torch_preds:
            pred = torch_preds[filename]
            pos = nms(pred['boxes'], pred['scores'], iou_threshold=nms_t)
            pred = {k:v[pos] for k,v in pred.items()}
            pred = {k: to_numpy(v) for k,v in pred.items()}
            add_detections(ds, fo_dataset, pred, os.path.join(ds.img_folder, filename), nms_t, key = nms_key )
    views = {}
    for conf in CONF_THRESH:
        for i in range(len(prediction_keys)):
            nms_t =  NMS_THRESH[i]
            nms_key = prediction_keys[i]
            views[conf] = (
            fo_dataset
            .filter_labels(nms_key, (F("confidence") > conf) )
            )

            key = "eval_C{}_NMS{}".format(int(conf*10), int(nms_t*10))

            results = fue.evaluate_detections(
                    views[conf], 
                    nms_key, 
                    classes=list(ds.get_labels().values()), 
                    eval_key=key, 
                    classwise=False, missing="No Object",
                    compute_mAP=True
            )

            plot = results.plot_confusion_matrix(backend='matplotlib')
            plot.savefig(os.path.join(output_dir, "Confusion matrix_NMS{}_C{}.png".format(nms_t,conf)))
            # plot = results.plot_pr_curves(backend='matplotlib')
            # plot.savefig(os.path.join(output_dir, "PR Curve_NMS{}_C{}.png".format(nms_t,conf)))

            FP_frames = []
            fp_key = key + "_fp"

            for sample in views[conf]:
                if sample[fp_key] > 0:
                    FP_frames.append(sample.filepath.split("/")[-1])
            print("Number of frames with FP:{}. Sample:{}".format(len(FP_frames), FP_frames[0]))

        return FP_frames