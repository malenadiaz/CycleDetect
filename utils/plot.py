import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

def plot_boxes(og_img, boxes, color = 'r', labels=[], scores=None, label_map=None):
    img = og_img.copy()
    if color == 'r':
        c = (0,0,255)
    else:
        c = (0,255,0)
    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = boxes[i] #coordinates #1,2 3,4
        w = x_max - x_min
        h = y_max - y_min

        lw = max(round(sum(img.shape) / 2 * 0.003), 2)  # Line width.
        tf = max(lw - 1, 1) # Font thickness.
        
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        
        cv2.rectangle(
            img,
            (x_min, y_min), (x_max, y_max),
            color=c, 
            thickness=lw,
            lineType=cv2.LINE_AA
        )

        final_label = label_map[labels[i] + 2] if label_map is not None else str(labels[i])
        final_label += ' ' + str(round(scores[i], 2)) if scores is not None else ''

        w, h = cv2.getTextSize(
            final_label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.3, 
            thickness=tf
        )[0]  # text width, height

        w = int(w - (0.20 * w))

        if scores is not None:
            outside = y_min - h >= 3 #
            y_min_txt = y_min
            y_max_txt = y_min - h - 3 if outside else y_min + h + 3
        else:
            outside = y_max + h <= img.shape[1] - 3
            y_min_txt  = y_max
            y_max_txt = y_max + h + 3 if outside else y_max - h - 3 
            
        cv2.rectangle(
            img, 
            (x_min, y_min_txt), 
            (x_min + w, y_max_txt), 
            color=c, 
            thickness=-1, 
            lineType=cv2.LINE_AA
        )  
        cv2.putText(
            img, 
            final_label, 
            (x_min, y_min - 3 if outside else y_min + h),
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.3, 
            color=(0, 0, 0), 
            thickness=tf, 
            lineType=cv2.LINE_AA
        )
    return img

def plot_img(img, boxes):
    ax = plt.gca()
    plt.imshow(img)
    plot_boxes(ax, boxes, color = 'r')

def plot_boxes_self_preds(fig, img, gt_boxes, pred_boxes, scores, gt_labels, pred_labels, label_map):
     # option 1: clean img + kpts_img
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img),
    ax.set_axis_off()
    #
    ax = fig.add_subplot(1, 2, 2)
    img = plot_boxes(img, gt_boxes, color = 'g', labels=gt_labels, label_map=label_map)
    img = plot_boxes(img, pred_boxes, color = 'r', labels=pred_labels, scores=scores, label_map=label_map)
    ax.imshow(img)

    return fig 


