# standard imports
import os
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import itertools
import math
import time
import glob

# third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

# adjacent imports
from dataset import DataReader, DNAdataset
from metrics import calculate_metrics
# from test import model_load
from feature import *
from model import Taeyeon_Net


# device
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_tenFold_model_list(model_path: str) -> list:
    ten_model_list = sorted(glob.glob(os.path.join(model_path, f"*.pt")), reverse=False)
    return ten_model_list

def load_model(model_path, in_channel, feature_len):
    model = Taeyeon_Net(in_channel, feature_len)
    model.load_state_dict(torch.load(model_path, map_location=dev))
    return model

def average_models_pred(x_list, y_list, model_path):
    # define OneHot
    cls_OneHot = OneHot(x_list)

    # onehot encoding
    x_arr = cls_OneHot.one_hot_encoding(4, 41)
    y_arr = np.asarray(y_list).astype(np.int64)

    # convert to torch dataset then dataloader
    test_ds = DNAdataset(x_arr, y_arr)
    test_dl = DataLoader(test_ds, batch_size=1024)

    # get k_fold models
    model_list = get_tenFold_model_list(model_path)
    models = [load_model(ch, 4, 41) for ch in model_list]

    truth_res = []
    pred_res = []
    pred_porb_res = []

    for i, data in enumerate(test_dl):
        x_batch, y_batch = data[0].to(dev), data[1].to(dev)
        batch_preds = []

        for model in models:
            model = model.to(dev)

            # model eval
            model.eval()
            with torch.no_grad():

                y_pred = model(x_batch)
                pred = F.softmax(y_pred.cpu().detach(), 1)
                batch_preds.append(pred)

        stacked_pred = torch.stack([v.data for v in batch_preds])
        ensemble_pred = torch.mean(stacked_pred, dim=0)
        pred_porb_res += [v for v in np.array(ensemble_pred)[:, 1]]
    return np.asarray(pred_porb_res).T, np.asarray(y_arr).T


if __name__ == "__main__":
    # argparse
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model_name", type=str, default="C.elegans")
    parser.add_argument("--predicted_data_name", type=str, default="C.elegans")
    parser.add_argument("--model_path", type=str, default="./results/")
    parser.add_argument("--test_data_path", type=str, default="./data/test_data")
    args = parser.parse_args()

    model_path = os.path.join(args.model_path, args.trained_model_name, "models")

    # create DataReader
    cls_DataReader = DataReader(args.predicted_data_name, dir_data=args.test_data_path)

    # load data to list
    x_list, y_list = cls_DataReader.load_data()
    print(f"Using model specie: {args.trained_model_name}")
    print(f"Predicting specie: {args.predicted_data_name}")
    print(f"testing pos data: {len(cls_DataReader.pos_list)}, testing neg data: {len(cls_DataReader.neg_list)}")

    # average k fold models pred
    preds, truths = average_models_pred(x_list, y_list, model_path)

    # calculate metrics
    preds = np.where(np.array(preds).astype(np.float) > 0.5, 1, 0).reshape(-1)  # threshold: 0.5
    truths = np.asarray(truths).astype(np.int)
    acc, sen, spe, mcc, tn, fp, fn, tp = calculate_metrics(truths, preds)
    bacc = round((sen+spe)/2, 3)
    print(f"Acc: {acc} | Sen: {sen} | Spe: {spe} | MCC: {mcc} | BAcc: {bacc} | TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

