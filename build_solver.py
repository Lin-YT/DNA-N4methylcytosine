# third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, TensorDataset, DataLoader

# adjacent folders imports
from model import Taeyeon_Net
from adabelief_pytorch import AdaBelief

# import config
from yacs.config import CfgNode


def build_dataloader(train_ds: Dataset, val_ds: Dataset, batch_size: int) -> DataLoader:
    """
    param:
            train_ds: training dataset
            val_ds: validation dataset
            batch_size
    return:
            train dataloader, val dataloader
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

def build_model(model_cfg: CfgNode) -> nn.Module:
    """
    param:
        model_cfg: model config
    return:
        model
    """
    if model_cfg.NAME == "Taeyeon_Net":
        model = Taeyeon_Net(model_cfg.INPUT_CHANNEL, model_cfg.INPUT_LENGTH)

    elif model_cfg.NAME == "other model":
        model = "other model"

    return model

def build_optimizer(model: nn.Module, opt_cfg: CfgNode) -> Optimizer:
    """
    param:
        model: already gpu pushed model
        opt_cfg: optimizer config
    return:
        optimizer
    """
    if opt_cfg.NAME == "SGD":
        opt = optim.SGD(model.parameters(), lr=opt_cfg.SGD.BASE_LR, momentum=opt_cfg.SGD.MOMENTUM, weight_decay =opt_cfg.SGD.WEIGHT_DECAY, nesterov=opt_cfg.SGD.NESTEROV)
    elif opt_cfg.NAME == "Adam":
        opt = optim.Adam(model.parameters(), lr=opt_cfg.ADAM.BASE_LR, weight_decay=opt_cfg.ADAM.WEIGHT_DECAY)
    elif opt_cfg.NAME == "AdaBelief":
        opt = AdaBelief(model.parameters(), lr=opt_cfg.ADABELIEF.BASE_LR, eps=opt_cfg.ADABELIEF.EPS, betas=(0.9, 0.999), weight_decay=5e-4, weight_decouple=False, rectify=False, print_change_log=False)
    else:
        raise Exception(f"invalid optimizer, available choices sgd/adam/adabelief ...")

    return opt

def build_schedular(optimizer: Optimizer, schedular_cfg: CfgNode, model:nn.Module):
    """
    param:
        optimizer: Optimizer
        schedular_cfg: CfgNode
    return:
        schedular
    """
    if schedular_cfg.NAME == "unchange":
        return None
    elif schedular_cfg.NAME == "Reduce_on_plateau":
        schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=schedular_cfg.REDUCELR.PATIENCE, factor=schedular_cfg.REDUCELR.FACTOR,\
                                min_lr=schedular_cfg.REDUCELR.MIN_LR, cooldown=schedular_cfg.REDUCELR.COOL_DOWN)
    else:
        raise(f"invalid schedular, available choices cosine_annealing_warm_restarts/cosine_annealing/reduce_on_plateau ...")

    return schedular