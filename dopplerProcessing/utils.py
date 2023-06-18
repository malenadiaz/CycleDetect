"""
Created by Malena Díaz Río. 
"""
import json
import os

from pydicom import dcmread


####### LOADING UTILS  
#loads json segmentation
def load_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data

#loads dicom
def load_dicom(img_path):
    try:
        ds = dcmread(img_path)
        return ds
    except:
        return None

#loads pixel array from dicom
def load_image(ds):
    if ds.PhotometricInterpretation != 'RGB':
        pixel_array_rgb = convert_color_space(ds.pixel_array, ds.PhotometricInterpretation, 'RGB', True) 
    else:
        pixel_array_rgb = ds.pixel_array
    return pixel_array_rgb


#get json annotations for an image given its path
def get_json_image(img_file, json_data):
    img_file = transform_name_no_dir(img_file)
    for annotator in json_data.values():
        for image in annotator.values():
            if transform_name_no_dir(image['path']) == img_file:
                return image
    raise ValueError('There is no information about that image in the file')

####### DICOM UTILS 
#returns doppler region
def get_region (ds):
    if 'SequenceOfUltrasoundRegions' in ds:
        for region in ds.SequenceOfUltrasoundRegions:
            if 'RegionDataType' in region:
                if region.RegionDataType == 3:
                    return region
    return None

#return physical metadata about the image 
def get_img_metadata(ds):
    metadata = {}
    region = get_region (ds)
    metadata['min_x'] = region.ReferencePixelY0 if hasattr(region,"RegionLocationMinX0") else None
    metadata['min_y'] = region.ReferencePixelY0 if hasattr(region,"RegionLocationMinY0") else None
    metadata['zero_line'] = region.ReferencePixelY0 if hasattr(region,"ReferencePixelY0") else None
    metadata['physical_delta_x']  = region.PhysicalDeltaX if hasattr(region,"PhysicalDeltaX") else None
    metadata['physical_delta_y']  = region.PhysicalDeltaY if hasattr(region,"PhysicalDeltaY") else None
    return metadata


#return true if the imgae si already segmented in the dicom
def isSegmented(ds):
    if 'BurnedInAnnotation' in ds:
        print("Burned in annotation tag found!")
        if 'BurnedInAnnotation' == 'YES': #https://dicom.innolitics.com/ciods/encapsulated-cda/encapsulated-document/00280301
            return True
    return False

####### DIRECTORY UTILS 
#iterates over a directory 
def open_directory(directory, output_dir, function):
	"""
	Input:
		- directory: directory from which to get the files
		- file: .txt file in which we want the names written
	Output: 
	Recursively opens folders and adds the name of the files to the .txt file. 
	"""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	file_count = 0
	for root, dirs, files in os.walk(directory):
		os.system("find . -name '.DS_Store' -delete") #mac
		for filename in files:
			print(root + "/" + filename)
			function(root + "/" + filename, output_dir + "/" + filename)
			file_count += 1
	return file_count

## CREATE CSV FILE WITH NAMES OF A FOLDER
def open_directory_v2(directory, output_dir, function, valid_labels, num_images):
    """
    Input:
    - directory: directory from which to get the files
    - file: .txt file in which we want the names written
    Output: 
    Recursively opens folders and adds the name of the files to the .txt file. 
    """
    create_dir(output_dir)

    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))] #adds subdir 
    for subdir in subdirs:
        file_count = 0
        if subdir in valid_labels:
            for root, dirs, files in os.walk(os.path.join(directory,subdir)):
                os.system("find . -name '.DS_Store' -delete") #mac
                for filename in files:
                    print(root + "/" + filename)
                    function(root + "/" + filename, output_dir + "/" + filename)
                    file_count += 1
                if file_count > num_images:
                    break
    return file_count

def open_files_from_file(filepath, output_dir, function, root):
    """
    Input:
    - directory: directory from which to get the files
    - file: .txt file in which we want the names written
    Output: 
    Recursively opens folders and adds the name of the files to the .txt file. 
    """
    create_dir(output_dir)

    img_list = []
    with open(filepath, "r") as f:
        img_list.extend(f.read().splitlines())

    for img in img_list:
        print(img)
        function(os.path.join(root, img), os.path.join(output_dir, img))
    return None

#creates a directory
def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

#transform three directories into a single name
def transform_name_no_dir(img_name):
    return "__".join(img_name.split("/")[-3:])

#transform name into three directories 
def transform_name_to_dir(img_name):
    return "/".join(img_name.split("__")[-3:])

###### DATASET SPLITTING
def distribute_dataset(output_dir, dataset_name, list_images, train, test, val):
    num_images = len(list_images)
    train_files = int(num_images * train + 0.5 )
    test_files = int(num_images * test + 0.5)
    val_files = int(num_images * val + 0.5 ) 
    test_aux_files = num_images - train_files - test_files - val_files
    counter= [0,0,0,0]
    with open(output_dir + "/filenames/" + dataset_name + "_train_filenames.txt", "w") as fp:
        for patient in list_images[ : train_files]:
            for item in patient:
                fp.write("%s\n" % (item + '.png'))
                counter[0]+=1

    with open(output_dir + "/filenames/" + dataset_name + "_test_filenames.txt", "w") as fp:
        for patient in list_images[train_files : train_files + test_files]:
            for item in patient:
                fp.write("%s\n" % (item + '.png'))
                counter[1]+=1

    with open(output_dir + "/filenames/" + dataset_name + "_val_filenames.txt", "w") as fp:
        for patient in list_images[train_files + test_files : train_files + test_files + val_files]:
            for item in patient:
                fp.write("%s\n" % (item + '.png'))
                counter[2]+=1

    if test_aux_files > 0 :
        with open(output_dir + "/filenames/" + dataset_name + "_test_aux_filenames.txt", "w") as fp:
            for patient in list_images[train_files + test_files + val_files : ]:
                for item in patient:
                    fp.write("%s\n" % (item + '.png'))
                    counter[3]+=1

    return counter[0], counter[1], counter[2], counter[3]