"""
Created by Malena Díaz Río. 
"""
import cv2
import matplotlib.pyplot as plt

#plots boxes of a single type: either all predictions or all annotations. Predictions should have scores != None.
def plot_boxes(og_img, boxes, labels=[], scores=None, label_map=None):
    img = og_img.copy()
    if scores is not None: #preds in red
        c = (255,0,0)
    else: #real values in green
        c = (0,255,0)

    for i in range(len(boxes)): 
        x_min, y_min, x_max, y_max = boxes[i] #coordinates #1,2 3,4
        w = x_max - x_min
        h = y_max - y_min

        lw = max(round(sum(img.shape) / 2 * 0.003), 0.7)  # Line width.
        tf = max(lw - 1, 1) # Font thickness.s
        
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        
        cv2.rectangle( #plot rectangle
            img,
            (x_min, y_min), (x_max, y_max),
            color=c, 
            thickness=lw,
            lineType=cv2.LINE_AA
        )

        final_label = label_map[labels[i]] if label_map is not None else str(labels[i])
        final_label += ' ' + str(round(scores[i], 2)) if scores is not None else ''

        w, h = cv2.getTextSize( #to draw background rectangle behing text
            final_label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.4, 
            thickness=tf
        )[0]  # text width, height

        w = int(w)

        if scores is not None: #predictions' annotations on top of the bounding box
            outside = y_min - h >= 3 #
            y_min_txt = y_min
            y_max_txt = y_min - h - 3 if outside else y_min + h + 3
        else: #ground truth boxes under the bounding box
            outside = y_max + h <= img.shape[1] - 3
            y_min_txt  = y_max
            y_max_txt = y_max + h + 3 if outside else y_max - h - 3 
            
        cv2.rectangle( #draw rectangle for text
            img, 
            (x_min, y_min_txt), 
            (x_min + w, y_max_txt), 
            color=c, 
            thickness=-1, 
            lineType=cv2.LINE_AA
        )   
        cv2.putText( #write text
            img, 
            final_label, 
            (x_min, y_min_txt if scores is not None else y_max_txt),
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.4, 
            color=(0, 0, 0), 
            thickness=tf, 
            lineType=cv2.LINE_AA
        )
    return img

def plot_img(img, boxes):
    ax = plt.gca()
    plt.imshow(img)
    plot_boxes(ax, boxes, color = 'r')

# plots ground truth boxes and the predictions with their corresponding annotations
def plot_boxes_self_preds(fig, img, gt_boxes, pred_boxes, scores, gt_labels, pred_labels, label_map):
    ax = fig.gca()
    img = plot_boxes(img, gt_boxes,labels=gt_labels, label_map=label_map)
    img = plot_boxes(img, pred_boxes, labels=pred_labels, scores=scores, label_map=label_map)
    ax.axis('off')
    ax.imshow(img)

    return fig 


