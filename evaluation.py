# standard imports
import os
import pandas as pd
import numpy as np
import glob

# third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# adjacent folders imports
from metrics import calculate_metrics
from build_solver import build_dataloader, build_model, build_optimizer, build_schedular

# config
from yacs.config import CfgNode
from config.config import get_cfg_defaults

# device
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(model, val_dl, loss_func):
    val_loss = 0.
    truth_res = []
    pred_res = []
    pred_porb_res = []

    # model eval
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dl):
            x_batch, y_batch = data[0].to(dev), data[1].to(dev)
            y_pred = model(x_batch)
            val_loss += loss_func(y_pred, y_batch).item()
            pred = F.softmax(y_pred.cpu().detach(), 1)

            truth_res += y_batch.tolist()
            pred_res += [v for v in np.argmax(pred.numpy(), -1).astype(np.float32)]
            pred_porb_res += [v for v in np.array(pred)[:, 1]]

    # calculate val statistic
    val_acc, val_sen, val_spe, val_mcc, tn, fp, fn, tp = calculate_metrics(truth_res, pred_res)
    val_loss /= len(val_dl)

    return val_loss, val_acc, val_sen, val_spe, val_mcc, truth_res, pred_res, pred_porb_res
