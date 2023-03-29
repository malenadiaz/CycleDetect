from .BaseEvaluator import DatasetEvaluator
from typing import Dict, List, Tuple
from datasets import datas
import torch 
import os 
import logging
import pickle
from collections import OrderedDict
import copy
import matplotlib.pyplot as plt
import shutil
import random
import cv2 
from utils.plot import plot_boxes_self_preds
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from mean_average_precision import MetricBuilder

SMOOTH = 1e-6


class ObjectDetectEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset: datas,
        tasks: List = ["bxs"],
        output_dir: str = "./visu",
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

    def set_logger(self, logname):
        print("Evaluation log file is set to {}".format(logname))
        logging.basicConfig(filename=logname,
                            filemode='w', #'a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)    #level=logging.DEBUG)    # level=logging.INFO)


    def _eval_boxes_predictions(self, predictions: Dict) -> Dict:
        """
        Evaluate keypoints predictions
        Args:
            predictions (list[dict]): list of predictions from the model
        """
        if self._verbose:
            self._logger.info("Eval stats for boxes")
        
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

        for prediction in predictions.values():
            gt,preds = self.convert_metric(prediction)
            metric_fn.add(preds, gt)

            # m.update(torch.Tensor([prediction["bxs_prediction"]]), torch.Tensor([prediction["bxs"]]))
            # pred = prediction["bxs_prediction"], prediction["boxes"]
            # /iou = self.iou_numpy(prediction["bxs_prediction"]['boxes'], prediction["bxs"]['boxes'])
            # dist_pred_gt_kpts.append(100 * match_two_kpts_set(prediction["keypoints"].reshape(num_kpts * num_annotated_frames, 2),
            #                                                   prediction["keypoints_prediction"].reshape(num_kpts * num_annotated_frames, 2)))

        map = metric_fn.value(iou_thresholds=0.1)['mAP']        
        if self._verbose:
            self._logger.info("Mean keypoints error is {}".format(map))
        return map
    

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
    # def iou_numpy(self,outputs: np.array, labels: np.array):
    #     outputs = outputs.squeeze(1)
        
    #     intersection = (outputs & labels).sum((1, 2))
    #     union = (outputs | labels).sum((1, 2))
        
    #     iou = (intersection + SMOOTH) / (union + SMOOTH)
        
    #     thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
        
    #     return thresholded  # Or thresholded.mean()



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
        if some_val_output_item["bxs_prediction"] is not None:
            tasks.append("bxs") 
        self._tasks = tasks

        self._predictions = dict()
        for ii, data_path in enumerate(outputs):
            prediction = dict()

            if some_val_output_item["bxs_prediction"] is not None:
                prediction["bxs_prediction"] = outputs[data_path]["bxs_prediction"]
                prediction["bxs"] = inputs[data_path]["bxs"]

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

        if self._output_dir is not None:
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "objectDetect_predictions.pkl")
            with open(file_path, 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
                res = self._eval_boxes_predictions(predictions)
            self._results[task] = res

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
   
   
    def plot(self, num_examples_to_plot: int) -> None:
        fig = plt.figure(constrained_layout=True, figsize=(16, 16))
        plot_directory = os.path.join(self._output_dir, "plots")
        if os.path.exists(plot_directory):
            shutil.rmtree(plot_directory)
        os.makedirs(plot_directory)
        self._logger.info("plotting {} prediction examples to {}".format(num_examples_to_plot, plot_directory))
        for data_path in random.sample(list(self._predictions), num_examples_to_plot):
            prediction = self._predictions[data_path]
            fig.clf()
            fig = self._plot_boxes_prediction(fig, prediction["data_path_from_root"],prediction["bxs_prediction"] )
            plot_filename = "{}.jpg".format(os.path.splitext(prediction["data_path_from_root"])[0].replace("/", "_"))
            fig.savefig(fname=os.path.join(plot_directory, plot_filename))

    def _plot_boxes_prediction(self,fig, data_path_from_root,boxes_prediction):
        datapoint_index = self._dataset.img_list.index(data_path_from_root)
        data = self._dataset.get_img_and_kpts(datapoint_index)
        img = data["img"]
        boxes = data["boxes"]
        # normalize:
        boxes = self._dataset.normalize_pose(boxes, img)
        selected_boxes = np.argwhere(boxes_prediction['scores']>0.45).flatten()
        boxes_pred = boxes_prediction['boxes'][selected_boxes, :]
        boxes_pred = self._dataset.normalize_pose(boxes_pred, img)
        img = cv2.resize(img, dsize=(400, 300), interpolation=cv2.INTER_AREA)

        boxes_pred = self._dataset.denormalize_pose(boxes_pred, img)
        boxes = self._dataset.denormalize_pose(boxes, img)

        plot_boxes_self_preds(fig, img, gt_boxes=boxes, pred_boxes=boxes_pred)

        return fig
