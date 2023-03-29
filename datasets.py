
import os
from data.CycleDetection import CycleDetection
import albumentations as A 
import CONST as CONST
from transforms import load_transform

class datas(object):
    """
    A simple class to hold the train/val/test dataset objects.
    """
    def __init__(self, loader_func: CycleDetection, dataset_config: dict, input_transform: A.core.composition.Compose,
                 train_filenames_list: str, val_filenames_list: str, test_filenames_list: str):
        self.loader_func = loader_func
        self.input_transform = input_transform

        self.dataset_config = dataset_config

        assert os.path.exists(dataset_config["img_folder"]), "image repository does not exist."
        self.trainset = self.load_train(train_filenames_list)
        self.valset = self.load_test(val_filenames_list)
        self.testset = self.load_test(test_filenames_list)


    def load_train(self, train_filenames_list: str) -> CycleDetection:
        trainset = None
        if train_filenames_list is not None:
            trainset = self.loader_func(dataset_config=self.dataset_config, filenames_list=train_filenames_list, transform=self.input_transform)
        return trainset
    
    def load_test(self, test_filenames_list: str) -> CycleDetection:
        testset = None
        if test_filenames_list is not None:
            testset = self.loader_func(dataset_config=self.dataset_config, filenames_list=test_filenames_list, transform=None)
        return testset

def load_dataset(ds_name: str, input_transform: A.core.composition.Compose = None, input_size:tuple[int] = (400,300)) -> datas:
    data_folder = CONST.US_MultiviewData
    loader_func = CycleDetection
    img_dirname = os.path.join(data_folder, "frames/")
    anno_dirname = os.path.join(data_folder, "annotations/")
    train_filenames_list = os.path.join(data_folder, 'filenames/train_filenames.txt')
    val_filenames_list = os.path.join(data_folder, 'filenames/val_filenames.txt')
    test_filenames_list = os.path.join(data_folder, 'filenames/test_filenames.txt')


    dataset_config = {"img_folder": img_dirname, "anno_folder": anno_dirname, "transform": input_transform, "input_size": input_size}

    ds = datas(loader_func=loader_func, dataset_config=dataset_config, input_transform=input_transform,
               train_filenames_list=train_filenames_list, val_filenames_list=val_filenames_list, test_filenames_list=test_filenames_list)

    if ds.trainset is not None and ds.testset is not None:
        print("loading dataset : {}.. number of train examples is {}, number of val examples is {}, number of test examples is {}."
                .format(ds_name, len(ds.trainset), len(ds.valset), len(ds.testset)))
    else:
        print('loading empty dataset.')

    return ds


if __name__ == '__main__':
    ds_name = "prueba"#"debug"#"echonet_random"#"echonet_random"#"echonet_cycle"
    augmentation_type = "strong_echo_cycle" #"strongkeep" #"twochkeep" #"strongkeep"


    #input_transform = None
    input_transform = load_transform()

    print_folder=os.path.join("./visu/", ds_name)

    if not os.path.exists(print_folder):
        os.mkdir(print_folder)
    ds = load_dataset(ds_name=ds_name, input_transform=input_transform)
    g = ds.trainset#ds.valset#ds.trainset
    for k in range(1, 3, 1): #len(g)):
        dat = g.get_img_and_kpts(index=k)
        g.plot_item(k, do_augmentation=False, print_folder=os.path.join("./visu/", ds_name))
        g.plot_item(k, do_augmentation=True, print_folder=os.path.join("./visu/", ds_name))