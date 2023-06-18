"""
Created by Malena Díaz Río. 
"""
import random
import cv2
import dopplerProcessing.crop as cp
import numpy as np
from dopplerProcessing.cycle_utils import get_cycle_annotations
from dopplerProcessing.dirsParser import parse_arguments_directories
from dopplerProcessing.utils import *
from dopplerProcessing.visualization import plot_img_boxes


def process_image(img_file, output_file):
    ds = load_dicom(img_file)

    if ds is None: #controls if it is a dicom
        return 0
    if isSegmented(ds): #controls if it is already segmented
        return 0

    output_name =  transform_name_no_dir (img_file) #get name 
    img = load_image(ds) #get pixel array

    json = get_json_image(img_file, json_data)
    patient = json["patient_id"]
    patient_files = []

    img_metadata = {}
    img_metadata["gen"] = get_img_metadata(ds) #to compute the physical metrics afterwards

    limits_x, limits_y, label = get_cycle_annotations(json)
    label = cp.encode_label(label)

    #crop image to get doppler image 
    c_img, transform_x, transform_y = cp.crop_image_doppler(ds, img)
    limits_x = np.array([list(map(transform_x, doppler_limits)) for doppler_limits in limits_x])
    limits_y = np.array([list(map(transform_y, doppler_limits)) for doppler_limits in limits_y])

    #check that the annotations are not outside of the bounding image 
    if np.any(np.array(limits_x) < 0)   or  np.any(np.array(limits_x) > c_img.shape[1]) or  np.any(np.array(limits_y) < 0) or  np.any(np.array(limits_y) > c_img.shape[0]):
        print("Annotations outside of spectral doppler for image: {}".format(output_name))

    #crop image to get only annotated cycles
    c_img, transform_x = cp.crop_image_annotated_limits(c_img
                                                    , min(limits_x.flatten())
                                                    , max(limits_x.flatten())
                                                    , margin = 10)
    limits_x = np.array([list(map(transform_x, annot_limits)) for annot_limits in limits_x])

    #clone number of cycles
    for clone in range(random.randint(0,6)):
        og_size = c_img.shape
        c_img = np.concatenate((c_img, c_img[:, limits_x[clone%3][0]:limits_x[clone%3][1]]), axis =1)
        
        cycle_width = limits_x[clone%3][1]-limits_x[clone%3][0]
        new_limits = np.array([[og_size[1],og_size[1] + cycle_width]]).astype(int)
        limits_x = np.concatenate((limits_x,new_limits))

        limits_y = np.concatenate((limits_y,[limits_y[clone%3]]))


    #resize with padding
    if c_img.shape[0] == 0  or c_img.shape[1]  == 0:
        print("Error with image: ", output_name, ". One dimension == 0.")
        return
    c_img, crop_data = cp.resize_image_padding(c_img, dims=DIMS)
    limits_x = cp.transform_list(limits_x, crop_data, axis="x")
    limits_y = cp.transform_list(limits_y, crop_data, axis="y")

    img_metadata['crop'] = crop_data

    #check that coordinates are inside bounding boxes
    if np.any(np.array(limits_x) < 0)   or  np.any(np.array(limits_x) > c_img.shape[1]) or  np.any(np.array(limits_y) < 0) or  np.any(np.array(limits_y) > c_img.shape[0]):
        print("Error with image: ", output_name, ". Bbx out of image.")
        return
    
    #save 
    limits = np.concatenate((limits_x, limits_y), axis = 1)
    limits[:, [1, 2]] = limits[:, [2, 1]] #format x_min, y_min, x_max, y_max 
    np.save( output_dir + '/annotations/' + output_name + '.npy', {'bbox':limits, 'label':label}) #annotations
    cv2.imwrite(output_dir +"/frames/" + output_name + '.png', cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY )) #frame in grayscale
    plot_img_boxes(c_img, limits_x, limits_y, output_dir + '/validations/' + output_name) #validation
    np.save( output_dir + '/metadata/' + output_name + '.npy', img_metadata) #metadata

    patient_files.append(output_name)

    if patient in list_images.keys(): #check that we do not put same patient in test and train 
        list_images[patient] = list_images[patient] + patient_files
    else:
        list_images[patient] = patient_files

if __name__ == "__main__": 
    #dimensions
    DIMS = [300,600]
    TRAIN = 0.6
    TEST = 0.2
    VAL = 0.2

    VALID_LABELS = ["Uterine Artery", "Umbilical Artery", "Middle Cerebral Artery"
		 			,"Aortic Isthmus", "Ductus Venosus", "Left Ventricular Inflow Outflow"]

    NUM_IMAGES = 250

    input_dir, output_dir , json_path = parse_arguments_directories()
    
    list_images = {} 

    if json_path is None:
        print("A segmentation file is required. Please input it with the parameter -j.")

    assert TRAIN + TEST + VAL == 1, "The percentages should add up to one."

    #load json 
    json_data = load_json(json_path)

    create_dir(output_dir + '/frames')
    create_dir(output_dir + '/annotations')
    create_dir(output_dir + '/validations')
    create_dir(output_dir + '/metadata')
    create_dir(output_dir + '/filenames')

    #process all files form a directory
    _ = open_directory_v2(input_dir, output_dir + '/frames', process_image, VALID_LABELS, NUM_IMAGES) 

    #LABEL = "Umbilical Artery" #given a.txt file process those files
    #root = (os.path.join(input_dir, LABEL))
    #_ = open_files_from_file(FILE, output_dir + '/frames', process_image, root) 

    #distribute images
    keys = list_images.keys()
    random.shuffle(list(keys)) 
    list_images_shuffled = [list_images[key] for key in keys]
    train_files, test_files, val_files, test_aux_files = distribute_dataset(output_dir,"doppler",list_images_shuffled, TRAIN, TEST, VAL)

    number_patients_processed = len(list_images)
    number_images_created = len([img for patient in list_images for img in patient])
    
    with open(output_dir + "/count.txt", "w") as fp0:
        fp0.write("{} files have been processed.\n".format(number_patients_processed))
        fp0.write("{} images have been created.\n".format(number_images_created))
