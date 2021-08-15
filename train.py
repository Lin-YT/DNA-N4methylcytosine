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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter

# adjacent folders imports
from metrics import calculate_metrics
from build_solver import build_dataloader, build_model, build_optimizer, build_schedular
from evaluation import validation, test
from train_utils import EarlyStopping

# config
from yacs.config import CfgNode
from config.config import get_cfg_defaults

# device
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(k_fold_time, model, loss_func, opt, schedular, epochs, train_dl, test_dl, log_path, save_path, use_early_stopping=False):
    # to initialize tensorboard
    writer = SummaryWriter(log_path)

    # save model in each k fold
    save_path = os.path.join(save_path, f"model_{k_fold_time}.pt")

    if use_early_stopping == True:
        early_stopping = EarlyStopping(verbose=True, save_model=True, patience=30, path=save_path)


    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0.
        truth_res = []
        pred_res = []

        # model train
        model.train()
        for i, data in enumerate(train_dl):
            x_batch, y_batch = data[0].to(dev), data[1].to(dev)

            # zero the parameters gradients
            opt.zero_grad()

            ### forward ###
            y_pred = model(x_batch)
            loss = loss_func(y_pred, y_batch)

            ### backward ###
            loss.backward()
            opt.step()

            train_loss += loss.item()
            pred = F.softmax(y_pred.cpu().detach(), 1)

            truth_res += y_batch.tolist()
            pred_res += [v for v in np.argmax(pred.numpy(), -1).astype(np.float32)]

        # calculate train statistic
        train_acc, train_sen, train_spe, train_mcc, tn, fp, fn, tp = calculate_metrics(truth_res, pred_res)
        train_loss /= len(train_dl)

        # model evaluate on validation data
        # val_loss, val_acc, val_sen, val_spe, val_mcc = validation(model, val_dl, loss_func)

        # model evaluate on test data
        te_loss, te_acc, te_sen, te_spe, te_mcc, true_label, predict_label, predict_porb = test(model, test_dl, loss_func)

        # tensorboard
        writer.add_scalars(f"Loss_fold_{k_fold_time}", {
            "train": train_loss,
            # "val": val_loss,
            "test": te_loss,
            }, epoch)
        writer.add_scalars(f"Acc_fold_{k_fold_time}", {
            "train": train_acc,
            # "val": val_acc,
            "test": te_acc,
            }, epoch)
        writer.add_scalars(f"Sen_fold_{k_fold_time}", {
            "train": train_sen,
            # "val": val_sen,
            "test": te_sen,
            }, epoch)
        writer.add_scalars(f"Spe_fold_{k_fold_time}", {
            "train": train_spe,
            # "val": val_spe,
            "test": te_spe,
            }, epoch)
        writer.add_scalars(f"Mcc_fold_{k_fold_time}", {
            "train": train_mcc,
            # "val": val_mcc,
            "test": te_mcc,
            }, epoch)
        writer.add_scalar(f"Learning rate_{k_fold_time}", opt.param_groups[0]['lr'], epoch)

        print(f'Epoch: {epoch+1:02} | Time: {time.time() - start_time:.2f}s')
        print(f'\tTrain loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Train sen: {train_sen:.3f} | '\
            f'Train spe: {train_spe:.3f} | Train mcc: {train_mcc:.3f}')
        # print(f'\tVal loss: {val_loss:.3f}   | Val acc: {val_acc:.3f}   | Val sen: {val_sen:.3f}   | '\
            # f'Val spe: {val_spe:.3f}   | Val mcc: {val_mcc:.3f}')
        print(f'\tTest loss: {te_loss:.3f}  | Test acc: {te_acc:.3f}  | Test sen: {te_sen:.3f}  | '\
            f'Test spe: {te_spe:.3f}  | Test mcc: {te_mcc:.3f}')

        # early stopping / schedular
        # schedular.step(val_loss)
        schedular.step(te_loss)

        if use_early_stopping == True:
            # early_stopping(val_loss, model)
            early_stopping(te_loss, model)

            if early_stopping.save_best_metrics:
                tmp_loss, tmp_acc, tmp_sen, tmp_spe, tmp_mcc = te_loss, te_acc, te_sen, te_spe, te_mcc
                save_tmp_predict = []
                save_tmp_predict.extend([true_label, predict_label, predict_porb])

            if early_stopping.early_stop:
                print('Early stopping')
                print("save best epoch predict of test ........")
                writer.close()
                return tmp_loss, tmp_acc, tmp_sen, tmp_spe, tmp_mcc, pd.DataFrame(np.array(save_tmp_predict).T)

    # save the final epoch information
    print(f'model saved in last epoch {epoch}')
    torch.save(model.state_dict(), save_path)

    return te_loss, te_acc, te_sen, te_spe, te_mcc, pd.DataFrame(np.array([true_label, predict_label, predict_porb]).T)