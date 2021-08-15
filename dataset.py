# standard imports
import os
import numpy as np

# third-party imports
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader


class DataReader():
    def __init__(self, file_name: str, dir_data=str) -> tuple:
        """
        Args:
            file_name(str): species name
            dir_data(str): data path

        return:
            tuple: (x_list, y_list)
        """
        self.dir_data = dir_data
        self.file_name = file_name + ".csv"
        self.data_path = self.get_data_path()
        self.pos_list = []
        self.neg_list = []

    def get_data_path(self) -> str:
        return os.path.join(self.dir_data, self.file_name)

    def load_data(self) -> list:
        try:
            with open(self.data_path, "r") as f:
                data = f.readlines()
        except IOError as e:
            raise IOError("Wrong file or file path ...") from e

        for i in range(0, len(data), 2):
            if data[i].startswith((">P", ">pos", ">Pos", ">GPP")):
                dna_seq = data[i+1].rstrip()
                self.pos_list.append(dna_seq)
            elif data[i].startswith((">N", ">neg", ">Neg")):
                dna_seq = data[i+1].rstrip()
                self.neg_list.append(dna_seq)
        return self.pos_list + self.neg_list, len(self.pos_list) * [1] + len(self.neg_list) * [0]


class DNAdataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])    #(B, C ,L)
        y = torch.tensor(self.y[idx])
        return x, y