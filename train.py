from config.defaults import cfg_costum_setup, default_argument_parser, get_run_id,convert_to_dict,create_tensorboard_run_dict
import torch
from models import load_model
from tqdm.auto import tqdm
from datasets import load_dataset
from transforms import load_transform
from engine.loops import sample_dataset, train_vanilla, validate
from losses import load_loss
from torch.utils.tensorboard import SummaryWriter
import CONST as CONST
import os 
import numpy as np
import socket
from evaluation.ObjectDetectEvaluator import ObjectDetectEvaluator
from engine.checkpoints import save_model
from optimizers import load_optimizer
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message=r"The frame.append", category=FutureWarning)

logs_dir = CONST.STORAGE_DIR


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hostname = socket.getfqdn()

    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        is_gpu = True
    else:
        is_gpu = False

    print("Train Config:\n", cfg)

    if cfg.AUG.PROB > 0:
        insize = cfg.TRAIN.INPUT_SIZE_IRR
        input_transform = load_transform()
    else:
        input_transform = None
    
    run_id = get_run_id()
    basename = "{}_{}".format(cfg.TRAIN.DATASET, cfg.MODEL.NAME)


    ds = load_dataset(ds_name=cfg.TRAIN.DATASET, input_transform=input_transform, input_size=cfg.TRAIN.INPUT_SIZE_IRR)

    #data loader
    trainloader, testloader, _ = sample_dataset(trainset=ds.trainset,
                                                valset=ds.valset,
                                                testset=None,
                                                overfit=cfg.TRAIN.OVERFIT,
                                                batch_size=cfg.TRAIN.BATCH_SIZE,
                                                num_workers=cfg.DATA_LOADER.NUM_WORKERS)

    
    model = load_model(cfg,is_gpu=is_gpu) # notice the default num of keypooints

    model.to(device)
    print('training model {}..'.format(model.__class__.__name__))

    log_folder = os.path.join(logs_dir, 'logs', cfg.TRAIN.DATASET,
                                cfg.MODEL.NAME, str(cfg.MODEL.BACKBONE), run_id)
    
    if len(cfg.MODEL.LOSS_FUNC) == 1:
        loss = cfg.MODEL.LOSS_FUNC[0]
    else:
        loss = cfg.MODEL.LOSS_FUNC #workaround for handling multiple losses in cfg 
    criterion = load_loss(loss, device=device)

    writer = SummaryWriter(log_dir=log_folder)
    
        # ----- Setup logger -----:
    best_val_MAP = 0
    best_val_metric = {"kpts": np.inf, "ef": np.inf, "sd": np.inf, "bxs":np.inf}
    evaluator = ObjectDetectEvaluator(dataset=ds.valset, output_dir=None, verbose=False)
    
    optimizer = load_optimizer(method_name=cfg.SOLVER.OPTIMIZER, parameters=model.parameters(), learningrate=cfg.SOLVER.BASE_LR)
    
    with open(os.path.join(log_folder,"train_config.yaml"), "w") as f:
        f.write(cfg.dump())   # save config to file

    print("Training in batches of size {}..".format(cfg.TRAIN.BATCH_SIZE))
    print("Using data augmentation type {} for {:.2f}% of the input data".format(cfg.AUG.METHOD, 100 * cfg.AUG.PROB))
    with tqdm(total=cfg.TRAIN.EPOCHS) as pbar_main:
        for epoch in range(1, cfg.TRAIN.EPOCHS+1):
            pbar_main.update()

            train_losses = train_vanilla(epoch=epoch,
                                       loader=trainloader,
                                       model=model,
                                       criterion = criterion,
                                       device=device, 
                                       optimizer = optimizer)
            train_loss = train_losses["main"].avg
            writer.add_scalar('TrainLoss/Loss', train_loss, epoch)
            writer.add_scalar('TrainLoss/BBox', train_losses["bbox_loss"].avg, epoch)
            writer.add_scalar('TrainLoss/Class', train_losses["class_loss"].avg, epoch)
            writer.add_scalar('TrainLoss/RPN', train_losses["rpn_loss"].avg, epoch)
            writer.add_scalar('TrainLoss/Object', train_losses["object_loss"].avg, epoch)

            #print(torch.cuda.memory_stats(device=device))

            # eval:
            if epoch % cfg.TRAIN.EVAL_INTERVAL == 0:
                val_maps, val_outputs, val_inputs = validate(mode='validation',
                                                               epoch=epoch,
                                                               loader=testloader,
                                                               model=model,
                                                               device=device,
                                                               criterion=criterion)
                #val_loss = val_losses["main"].avg
                writer.add_scalar('MAPVal/@0.5.0.95', val_maps["MAP@0.5.0.95"], epoch)
                writer.add_scalar('MAPVal/@0.5', val_maps["MAP@0.5"], epoch)
                writer.add_scalar('MAPVal/@0.75', val_maps["MAP@0.75"], epoch)
                writer.add_scalar("MAPVal/@0.5.0.95.s", val_maps["MAP@0.5.0.95.s"], epoch)
                writer.add_scalar("MAPVal/@0.5.0.95.m", val_maps["MAP@0.5.0.95.m"], epoch)
                writer.add_scalar("MAPVal/@0.5.0.95.l", val_maps["MAP@0.5.0.95.l"], epoch)
                writer.add_scalar("MARVal/@10", val_maps["MAR@10"], epoch)


                if val_maps["MAP@0.5.0.95"] > best_val_MAP:
                    filename = os.path.join(log_folder, 'weights_{}_best_map.pth'.format(basename))
                    best_val_MAP = val_maps["MAP@0.5.0.95"]
                    save_model(filename, epoch, model, cfg, train_loss, best_val_MAP, best_val_metric, hostname)
                    print("Saved at MAP {:.5f}\n".format(best_val_MAP))
                    writer.add_scalar('BestVal/MAP', best_val_MAP, epoch)
    
    # Save & Close:
    print('Finished Training')
    filename = os.path.join(log_folder, 'weights_{}_ep_{}.pth'.format(basename, epoch))
    save_model(filename, epoch, model, cfg, train_loss, val_maps["MAP@0.5.0.95"], best_val_MAP, hostname)
    writer.close()  # close tensorboard
    return train_loss


if __name__ == '__main__':

    args = default_argument_parser()
    cfg = cfg_costum_setup(args)


    plt.ioff()
    train(cfg)
