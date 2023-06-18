"""
Created by Malena Díaz Río. 
"""
import cv2
import numpy as np


#if region of Doppler spectrogram not in DICOM data
def crop_manually(img):
    medium_height = img.shape[0] // 8 * 5
    transform_x = lambda x : x  
    transform_y = lambda y  : y - medium_height 
    return img[medium_height : ,], transform_x, transform_y

#crop the image to only contain the Doppler spectrogram
def crop_image_doppler(ds, img):
    found = False
    if 'SequenceOfUltrasoundRegions' in ds:
        for item in ds.SequenceOfUltrasoundRegions:
            if 'RegionDataType' in item:
                if item.RegionDataType == 3:
                    found = True
                    x_min = item.RegionLocationMinX0
                    x_max = item.RegionLocationMaxX1
                    y_min = item.RegionLocationMinY0
                    y_max = item.RegionLocationMaxY1
                    cropped_img = img [y_min : y_max,x_min :x_max, :]
                    transform_x = lambda x : x - x_min 
                    transform_y = lambda y  : y - y_min 
                    break
    if not found:
        print("Manual crop used")
        cropped_img, transform_x, transform_y = crop_manually(img)
    return cropped_img, transform_x, transform_y

#get only the annotated cycles of the image
def crop_image_annotated_limits(img, min_x, max_x, margin = 0):
    minimum = max(0,min_x - margin)
    maximum = min(img.shape[1],max_x + margin)
    cropped_img = img [:, minimum :maximum, :]
    transform_x  = lambda x : x - minimum
    return cropped_img, transform_x

#add padding x and padding y to bounding box. 
def add_padding_box(max_values,limits_x, limits_y, padding_x = 0, padding_y = 20):
    padded_limits_x = []
    padded_limits_y = []
    for i in range(len(limits_x)):
        x_min = max(limits_x[i][0]- padding_x, 0)
        y_min = max(limits_y[i][0]- padding_y, 0)
        x_max = min(max_values[1], limits_x[i][1] + padding_x)
        y_max = min(max_values[0], limits_y[i][1] + padding_y)
        padded_limits_x.append([x_min, x_max ])
        padded_limits_y.append([y_min, y_max ])
    return padded_limits_x, padded_limits_y

#resize the image 
def resize (img, dims = None):
    if dims is not None: #resize
        t = max((dims[0] - img.shape[0])//2, 0) 
        b = max(dims[0] - img.shape[0] - t, 0) 
        l = max((dims[1] - img.shape[1])//2, 0) 
        r = max(dims[1] - img.shape[1] - l, 0)
        # c_img = cv2.copyMakeBorder(img.copy(),0, 0, l, r,cv2.BORDER_REFLECT)
        # c_img = cv2.copyMakeBorder(c_img.copy(),t, b, 0, 0,cv2.BORDER_REPLICATE)
        
        # resize image
        c_img = cv2.resize(img, dims, interpolation = cv2.INTER_AREA)

        transform_x = lambda x : x  + l 
        transform_y = lambda y  : y  + t
    return c_img, transform_x, transform_y

#scales the x kpts
def transform_x(x, cycle_metadata):
    return int(x*cycle_metadata["ratio"] + cycle_metadata["x_shift"])

#scales the y kpts
def transform_y(y, cycle_metadata):
    return int(y*cycle_metadata["ratio"]  + cycle_metadata["y_shift"])

#convert to original pixel coordinates
def transform_list(input_list, cycle_metadata, axis="x"):
    output_list = []
    function = transform_x if axis == "x" else transform_y
    for kpt in input_list:
        output_list.append([function(kpt[0], cycle_metadata), function(kpt[1], cycle_metadata)])
    return output_list

#label encoder
def encode_label(label):
    if label in ["Uterine Artery"]:
        return 1
    if label in ["Aortic Isthmus"]:
        return 2
    if label in ["Ductus Venosus"]:
        return 3
    if label in ["Left Ventricular Inflow Outflow"]:
        return 4
    if label in ["Umbilical Artery"]:
        return 5
    if label in ["Middle Cerebral Artery"]:
        return 6
    return 0

#resize the image by adding padding preserving its aspect ratio
def resize_image_padding(c_img, dims):
    crop_data = {}
    ratio = 1
    if c_img.shape[0] > dims [0] or c_img.shape[1]  > dims [1]:
        ratio =  min(float(dims[0])/c_img.shape[0], float(dims[1])/c_img.shape[1])
        w = int(c_img.shape[1] * ratio)
        h = int(c_img.shape[0] * ratio)
        c_img = cv2.resize(c_img, (w,h), interpolation = cv2.INTER_AREA)
    
    t = max((dims[0] - c_img.shape[0])//2, 0) 
    b = max(dims[0] - c_img.shape[0] - t, 0) 
    l = max((dims[1] - c_img.shape[1])//2, 0) 
    r = max(dims[1] - c_img.shape[1] - l, 0)
    
    c_img = cv2.copyMakeBorder(c_img.copy(),t,b,l,r,cv2.BORDER_CONSTANT,value=[0,0,0])

    crop_data["x_shift"] = l 
    crop_data["y_shift"] = t
    crop_data["ratio"] = ratio
    crop_data["margin"] = 0
    return c_img, crop_data
