"""
Created by Malena Díaz Río. 
"""
import json

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from utils import load_image


## DRAW IMAGE
def draw_image(img_path):
    pixel_array_rgb = load_image(img_path)
    if len(pixel_array_rgb.shape) == 4:
        plt.imshow(pixel_array_rgb[0], interpolation='spline36')
    else:
        plt.imshow(pixel_array_rgb, interpolation='spline36')

#used to plot the validations when preprocessing the dataset, plots boxes without annotations on image 
def plot_img_boxes(img, limits_x, limits_y, name):
    plt.clf()
    ax = plt.gca()
    plt.imshow(img)
    for i in range(len(limits_y)):
        min_x = limits_x[i][0]
        min_y = limits_y[i][0]
        w = limits_x[i][1] - limits_x[i][0]
        h = limits_y[i][1] - limits_y[i][0]
        rect = Rectangle((min_x, min_y), w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.savefig( name + ".png")

