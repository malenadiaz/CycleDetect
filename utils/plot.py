import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_boxes(boxes, color = 'r'):
    ax = plt.gca()

    for box in boxes:
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min
        rect = Rectangle((x_min, y_min), w,h,linewidth=1,edgecolor=color,facecolor='none')
        ax.add_patch(rect)

def plot_img(img, boxes):
    ax = plt.gca()
    plt.imshow(img)
    plot_boxes(ax, boxes, color = 'r')

def plot_boxes_self_preds(fig, img, gt_boxes, pred_boxes):
     # option 1: clean img + kpts_img
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)
    ax.set_axis_off()
    #
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(img)
    plot_boxes(gt_boxes, color = 'r')
    plot_boxes(pred_boxes, color = 'g')
    return fig 


