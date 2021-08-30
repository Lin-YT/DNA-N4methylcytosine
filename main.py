# standard imports
import os
import pandas as pd
import numpy as np
import time
import math

# third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
from adabelief_pytorch import AdaBelief

# adjacent folders imports
from model import Taeyeon_Net
from train_utils import EarlyStopping
from metrics import calculate_metrics
from dataset import DataReader, DNAdataset
from build_solver import build_dataloader, build_model, build_optimizer, build_schedular
from evaluation import validate
from feature import OneHot
from train import train

# config
from yacs.config import CfgNode
from config.config import get_cfg_defaults

# device
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def k_fold_trainer(x_arr, y_arr, n_splits=10, *want_concat_feature):
    # check path
    log_path = os.path.join(cfg.RESULT_PATH, cfg.DATASET.NAME, "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    save_model_path = os.path.join(cfg.RESULT_PATH, cfg.DATASET.NAME, "models")
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    # k-fold training
    k = 0
    x_arr, y_arr= shuffle(x_arr, y_arr, random_state=cfg.DATASET.RANDOM_SEED)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.DATASET.RANDOM_SEED)
    k_loss, k_acc, k_sen, k_spe, k_mcc = [[] for _ in range(5)]
    k_save_predict = []


    for train_index, val_index in skf.split(x_arr, y_arr):
        k += 1
        print(f'Start training on Fold {k}')
        X_train, X_val = x_arr[train_index], x_arr[val_index]
        Y_train, Y_val = y_arr[train_index], y_arr[val_index]


        # convert to torchDataset
        train_ds = DNAdataset(X_train, Y_train)
        val_ds = DNAdataset(X_val, Y_val)

        # get model
        model = build_model(cfg.MODEL).to(dev)

        # build loss function
        loss_func = nn.CrossEntropyLoss()

        # get optimizer
        opt = build_optimizer(model, cfg.OPTIMIZER)

        # build data_loader
        train_dl, val_dl = build_dataloader(train_ds, val_ds, cfg.TRAIN.BATCH_SIZE)

        # get schedular
        schedular = build_schedular(opt, cfg.SCHEDULAR, model)

        # training
        loss, acc, sen, spe, mcc, one_fold_predict = train(k, model, loss_func, opt, schedular, cfg.TRAIN.EPOCH, train_dl, val_dl,\
                                                            log_path=log_path, save_path=save_model_path, use_early_stopping=cfg.TRAIN.USE_EARLY_STOPPING)

        k_loss.append(loss)
        k_acc.append(acc)
        k_sen.append(sen)
        k_spe.append(spe)
        k_mcc.append(mcc)
        k_save_predict.append(one_fold_predict)
        print(f'current {k}-fold')
        print(f'k_loss: {k_loss} \nk_acc: {k_acc} \nk_sen: {k_sen} \nk_spe: {k_spe} \nk_mcc: {k_mcc}\n')

    # compute k-fold statistic
    print('=' * 50)
    print(f'{n_splits}-fold statistic')
    for i in range(n_splits):
        print(f'{i+1}-fold: acc: {k_acc[i]} | sen: {k_sen[i]} | spe: {k_spe[i]} | mcc: {k_mcc[i]}')

    print(f'mean: acc: {np.asarray(k_acc).mean():.3f} | sen: {np.asarray(k_sen).mean():.3f} | spe: {np.asarray(k_spe).mean():.3f} | mcc: {np.asarray(k_mcc).mean():.3f}')
    print(f'std: acc: {np.asarray(k_acc).std():.3f} | sen: {np.asarray(k_sen).std():.3f} | spe: {np.asarray(k_spe).std():.3f} | mcc: {np.asarray(k_mcc).std():.3f}')


def main(cfg):

    # create DataReader
    cls_DataReader = DataReader(cfg.DATASET.NAME, dir_data=cfg.DATASET.DATA_PATH)

    # load data to list
    x_list, y_list = cls_DataReader.load_data()
    print(f"Reading {cfg.DATASET.NAME} data")
    print(f"Pos data: {len(cls_DataReader.pos_list)}, Neg data: {len(cls_DataReader.neg_list)}\n")

    # define OneHot
    cls_OneHot = OneHot(x_list)

    # onehot encoding
    x_arr = cls_OneHot.one_hot_encoding(4, 41)
    y_arr = np.asarray(y_list).astype(np.int64)

    # train
    k_fold_trainer(x_arr, y_arr, cfg.TRAIN.K_FOLD_TRAINING_TIMES)


if __name__ == "__main__":
    # argparse
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config.yml")
    args = parser.parse_args()

    cfg_path = os.path.join("config/experiments/", args.cfg)

    # load config information
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    main(cfg)
