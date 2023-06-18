import pandas as pd
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import nms
import torch 
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools

#save the stats to a .txt
def save_stats(stats, output_dir, label_map, per_class = True):
    results = """
    MAP @50-95: {:.4f}
    MAP @50: {:.4f}
    MAP @75: {:.4f}
    MAP @50-95 s: {:.4f}
    MAP @50-95 m: {:.4f}
    MAP @50-95 l: {:.4f}
    ===================
    MAR @1: {:.4f}
    MAR @10: {:.4f}
    MAR @100: {:.4f}
    MAR @small: {:.4f}
    MAR @medium: {:.4f}
    MAR @large: {:.4f}
    ==================
    """.format(stats["map"].float(),stats["map_50"].float(), stats["map_75"].float(), 
               stats["map_small"].float(), stats["map_medium"].float(),stats["map_large"].float(), 
               stats["mar_1"].float(), stats["mar_10"].float(), stats["mar_100"].float(),
               stats["mar_small"].float(), stats["mar_medium"].float(),stats["mar_large"].float(),  
                )
    
    if per_class:
        for i, label in label_map.items():
            results += "MAP @50-95 {}: {:.4f}\n".format(label, stats["map_per_class"][i - 1].float())
        results += "    ==================\n"     

        for i, label in label_map.items():
            results += "    MAR @100 {}: {:.4f}\n".format(label, stats["mar_100_per_class"][i - 1].float())

    with open(os.path.join(output_dir, "map.txt"), "w") as fp:
        fp.write(results)

#filter per position
def filter_pred_positions(pred, pos):
    if len(pos) > 0: 
        pred = {k:v[pos] for k,v in pred.items()}
    return pred

#get label of the prediction
def get_label(pred):
    sorted , indices = torch.sort( pred['scores'])
    indices = indices[:5]
    value = pred['labels'][indices].mode().values
    value = value.item()
    return value

#filter by label
def filter_labels(pred):
    sorted , indices = torch.sort( pred['scores'])
    indices = indices[:5]
    value = pred['labels'][indices].mode().values
    value = value.item()
    pred['labels'] =  torch.full_like(pred['labels'], value)
    return pred

#apply nms to predictions and filters
def filter_nms(pred, thresh):
    pos = nms(pred['boxes'], pred['scores'], iou_threshold=thresh)
    pred = filter_pred_positions(pred, pos)
    return pred

#filter predictions by confidence threshold
def filter_conf(pred, thresh):
    pos = [ i for i, v in enumerate(pred['scores']) if v >= thresh]
    pred = filter_pred_positions(pred, pos)
    return pred

#plot confusion matrix same as 51 function
def plot_matrix(true, pred, labels, out_dir):
    labels = list(labels.values())
    cm = confusion_matrix(true,pred)

    fig, ax = plt.subplots()

    cm = np.asarray(cm)
    nrows = cm.shape[0]
    ncols = cm.shape[1]

    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")

    # Print text with appropriate color depending on background
    cmap_min = im.cmap(0)
    cmap_max = im.cmap(256)
    thresh = (cm.max() + cm.min()) / 2.0

    for i, j in itertools.product(range(nrows), range(ncols)):
        color = cmap_max if cm[i, j] < thresh else cmap_min

        text_cm = format(cm[i, j], ".2g")
        if cm.dtype.kind != "f":
            text_d = format(cm[i, j], "d")
            if len(text_d) < len(text_cm):
                text_cm = text_d
        ax.text(j, i, text_cm, ha="center", va="center", color=color)

    ax.set(
        xticks=np.arange(ncols),
        yticks=np.arange(nrows),
        xticklabels=labels[:ncols],
        yticklabels=labels[:nrows],
        xlabel="Predicted label",
        ylabel="True label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45.0)

    ax.set_ylim((nrows - 0.5, -0.5))  # flip axis

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    plt.tight_layout()

    # Save the confusion matrix plot
    output_path = os.path.join(out_dir, 'cm_real_class.png')
    plt.savefig(output_path)

#compute map and mar for different NSM and confidence threshold and plot them
def NMS_vs_CT(preds, targets, labels, output_dir):

    NMS_THRESH = [0.1, 0.2, 0.3, 0.4, 0.5]
    CT_THRESH = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    label_names = ['general'] + list(labels.values())
    map_results = {k1:None for k1 in NMS_THRESH}
    mar_results = {k1:None for k1 in NMS_THRESH}

    for nms in NMS_THRESH: 
        nms_filtered_preds = list(map(lambda x: filter_nms(x, nms), preds)) #filter preds by nms 
        
        ct_results_map = {k2:[] for k2 in label_names}
        ct_results_mar = {k2:[] for k2 in label_names}

        for ct in CT_THRESH: #filter preds by confidence thresholds 
            ct_filtered_preds = list(map(lambda x: filter_conf(x, ct), nms_filtered_preds))
            
            metric = MeanAveragePrecision(class_metrics=True)
            metric.update(ct_filtered_preds, targets)
            stats = metric.compute()
            
            ct_results_map['general'].append(stats['map'].float())
            ct_results_mar['general'].append(stats['mar_100'].float())

            for i, value in enumerate(stats['map_per_class']):
                ct_results_map[labels[i + 1]].append(value.float())
            for i, value in enumerate(stats['mar_100_per_class']):
                ct_results_mar[labels[i + 1]].append(value.float())

        map_results[nms] = ct_results_map
        mar_results[nms] = ct_results_mar

    plot_NMS_vs_CT(map_results, CT_THRESH , output_dir, label_names, ylabel = "Precision")
    plot_NMS_vs_CT(mar_results, CT_THRESH , output_dir, label_names, ylabel = "Recall")

#plot curve NMS vs CT. MAP or MAR is the y axis, CT the x axis and there is one line per NMS 
def plot_NMS_vs_CT(results, x, output_dir, label_names, ylabel = None):
    for label in label_names:
        plt.clf()
        ax = plt.gca()
        for nms in results:
            ax.plot(x, results[nms][label], label = "@NMS_{}".format(nms))
        ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        plt.savefig(os.path.join(output_dir, "{}_{}_NMS_vs_CT.png".format(ylabel,label)))