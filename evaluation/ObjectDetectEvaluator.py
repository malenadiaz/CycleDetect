"""
Created by Malena Díaz Río. 
"""
import copy
import logging
import os
import pickle
import random
import shutil
from collections import OrderedDict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import datas
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils.plot import plot_boxes_self_preds
from utils.stats_51 import convert_to_fityone, evaluate_51
from utils.utils_files import to_numpy
from utils.utils_stat import (filter_conf, filter_nms, get_label, plot_matrix,
                              save_stats)

from .BaseEvaluator import DatasetEvaluator

SMOOTH = 1e-6


class ObjectDetectEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset: datas,
        tasks: List = ["bxs"],
        output_dir: str = None,
        verbose: bool = True
    ):
        """
        Args:
            dataset (dataset object): Note: used to be dataset_name: name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                "json_file": the path to the LVIS format annotation
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "single_iter", "multi_iter".
                By default, will infer this automatically from predictions.
            output_dir (str): optional, an output directory to dump results.
        """

        self._dataset = dataset
        self._verbose = verbose
        self._tasks = tasks
        self._output_dir = output_dir
        if self._verbose:
            self.set_logger(logname=os.path.join(output_dir, "eval_log.log"))
            self._logger = logging.getLogger(__name__)

        self._cpu_device = torch.device("cpu")
        self._do_evaluation = True  # todo: add option to evaluate without gt
        self.nms_t = 0.3
        self.conf_t = 0.5
        self.multiclass = len(self._dataset.get_labels()) > 1

    def set_logger(self, logname):
        print("Evaluation log file is set to {}".format(logname))
        logging.basicConfig(filename=logname,
                            filemode='w', #'a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)    #level=logging.DEBUG)    # level=logging.INFO)

    def convert_metric(self, prediction):
        gt = np.array(prediction["bxs"]['boxes'])
        zero_array = np.zeros(shape = (gt.shape[0], 3))
        gt = np.hstack((gt,zero_array))

        preds = np.array(prediction["bxs_prediction"]['boxes'])
        class_id = np.zeros(shape = (preds.shape[0], 1))
        scores = np.array(prediction["bxs_prediction"]['scores']).reshape(-1,1)
        preds = np.hstack((preds,class_id))
        preds = np.hstack((preds,scores))
        return gt, preds

    def process(self, inputs: Dict, outputs: Dict) -> None:
        """
        Args:
            inputs: the inputs to a EF and Kpts model. It is a list of dicts. Each dict corresponds to an image and
                contains keys like "keypoints", "ef".
            outputs: the outputs of a EF and Kpts model. It is a list of dicts with keys
                such as "ef_prediction" or "keypoints_prediction" that contains the proposed ef measure or keypoints coordinates.
        """
        some_val_output_item = next(iter(outputs.items()))[1]
        tasks = []
        if "boxes" in some_val_output_item:
            tasks.append("bxs") 
        self._tasks = tasks

        self._predictions = dict()
        for ii, data_path in enumerate(outputs):
            prediction = dict()

            if "bxs" in tasks:
                prediction["gt"] = inputs[data_path]
                pred = outputs[data_path]
                pred = filter_nms(pred, self.nms_t) #filter through nms and confidence threshold
                pred = filter_conf(pred, self.conf_t)
                prediction["pred"] = pred

            # get case name:
            prediction["data_path_from_root"] = data_path.replace(self._dataset.img_folder, "")

            self._predictions[data_path] = prediction

    def get_tasks(self) -> List:
        return self._tasks
    
    def evaluate(self, tasks: List = None):
        if tasks is not None:
            self._tasks = tasks

        predictions = self._predictions

        if len(predictions) == 0 and self._verbose:
            self._logger.warning("[ObjectDetectEvaluator] Did not receive valid predictions.")
            return {}
        
        if self._output_dir is not None: #write the predictions in a pickle file
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "objectDetect_predictions.pkl")

            numpy_preds = {} #convert them to numpy format
            for data_path in predictions:
                boxes_gt = {k: to_numpy(v) for k, v in predictions[data_path]["gt"].items()}
                boxes_pred = {k: to_numpy(v) for k, v in predictions[data_path]["pred"].items()}
                numpy_preds[data_path] = {"data_path_from_root": data_path, "gt":boxes_gt, "pred":boxes_pred}
            with open(file_path, 'wb') as handle:
                pickle.dump(numpy_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if not self._do_evaluation and self._verbose:
            self._logger.info("Annotations are not available for evaluation.")
            return

        if self._verbose:
            self._logger.info("Evaluating predictions ...")
        self._results = OrderedDict()
        tasks = self._tasks #or self._tasks_from_predictions(lvis_results)
        for task in sorted(tasks):
            if self._verbose:
                self._logger.info("Preparing results in the CycleDetect format for task {} ...".format(task))
            if task == "bxs":
                res = self._eval_boxes_predictions(predictions) #evaluate the predictions
            self._results[task] = res

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
    
    def _eval_boxes_predictions(self, predictions: Dict) -> Dict:
        """
        Evaluate keypoints predictions
        Args:
            predictions (list[dict]): list of predictions from the model
        """
        if self._verbose:
            self._logger.info("Eval stats for boxes")
        
        preds = []
        targets = []
        # metric_fn = MetricBuilder.build_evaluation_metric("map_2d ", async_mode=True, num_classes=1)

        if self._output_dir: #create confusion matrix and PR and get the false negatives and false positives at IoU 0.5
            fo_dataset = convert_to_fityone(self._dataset)
            self.fp, self.fn = evaluate_51(fo_dataset,self._dataset, predictions,self._output_dir, self.nms_t, self.conf_t)

        pred_labels = []
        gt_labels = []
        for prediction in predictions.values(): #generate real labels 
            targets.append(prediction["gt"])
            preds.append(prediction['pred'])
            pred_labels.append(get_label(prediction["pred"]) - 1)
            gt_labels.append(prediction["gt"]['labels'][0].int() - 1)
        plot_matrix(gt_labels, pred_labels, self._dataset.get_labels(), self._output_dir) #create real labels confusion matrix


        #compare NMS vs CT in order to define the best thresholds 
        #NMS_vs_CT(preds, targets, self._dataset.get_labels(), self._output_dir)
        
        metric = MeanAveragePrecision(class_metrics=True) #create classification report with MAPs and MARs
        metric.update(preds, targets)
        stats = metric.compute()
        save_stats(stats, self._output_dir, self._dataset.get_labels(), per_class = self.multiclass)
        map = stats['map'].float()     
        if self._verbose:
            self._logger.info("MAP @ error is {}".format(map))
        return map
    
    #plot ground truth vs predictions 
    def plot(self, num_examples_to_plot: int) -> None:
        fig = plt.figure(constrained_layout=True, figsize=(25, 15))
        plot_directory = os.path.join(self._output_dir, "plots")
        if os.path.exists(plot_directory):
            shutil.rmtree(plot_directory)
        os.makedirs(plot_directory)
        os.makedirs(os.path.join(plot_directory, "FP"))
        os.makedirs(os.path.join(plot_directory, "FN"))

        self._logger.info("plotting {} prediction examples to {}".format(num_examples_to_plot, plot_directory))
        
        for data_path in self.fp:#plot FP
            file_path = os.path.join(self._dataset.img_folder, data_path)
            prediction = self._predictions[file_path]
            fig.clf()
            fig = self._plot_boxes_prediction(fig, prediction["data_path_from_root"],prediction["pred"])
            plot_filename = "{}.jpg".format(os.path.splitext(prediction["data_path_from_root"])[0].replace("/", "_"))
            fig.savefig(fname=os.path.join(plot_directory, "FP", plot_filename))

        for data_path in self.fn:#plot FN
            file_path = os.path.join(self._dataset.img_folder, data_path)
            prediction = self._predictions[file_path]
            fig.clf()
            fig = self._plot_boxes_prediction(fig, prediction["data_path_from_root"],prediction["pred"])
            plot_filename = "{}.jpg".format(os.path.splitext(prediction["data_path_from_root"])[0].replace("/", "_"))
            fig.savefig(fname=os.path.join(plot_directory, "FN", plot_filename))

        for data_path in random.sample(list(self._predictions), 10):
            prediction = self._predictions[file_path]
            fig.clf()
            fig = self._plot_boxes_prediction(fig, prediction["data_path_from_root"],prediction["pred"])
            plot_filename = "{}.jpg".format(os.path.splitext(prediction["data_path_from_root"])[0].replace("/", "_"))
            fig.savefig(fname=os.path.join(plot_directory, plot_filename))
        
    #plot ground truth objects and predictions of one single sample
    def _plot_boxes_prediction(self,fig, data_path_from_root,boxes_prediction):
        datapoint_index = self._dataset.img_list.index(data_path_from_root)
        data = self._dataset.get_img_and_bxs(datapoint_index)
        img = data["img"]
        boxes = data["boxes"]
        labels = data["labels"]

        boxes_pred = {k: to_numpy(v) for k, v in boxes_prediction.items()}
        boxes = self._dataset.normalize_pose(boxes, img)

        plot_boxes_self_preds(fig, img, gt_boxes=boxes, pred_boxes=boxes_pred, scores=boxes_prediction['scores']
                              , gt_labels = labels, pred_labels=boxes_prediction['labels'], label_map = self._dataset.get_labels() )

        return fig

