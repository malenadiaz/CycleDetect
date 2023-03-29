import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
def load_model(cfg, is_gpu = None):
    
    # # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.TRAIN.NUM_CLASSES) 

    # model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model.output_type = 'objectDetect'

    return model