# standard imports
import os
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import itertools
import math

# config
from yacs.config import CfgNode
from config.config import get_cfg_defaults


class OneHot():

    def __init__(self, x_list):
        self.x_list = x_list
        self.vocabs2index = {
                             "A": 0,
                             "C": 1,
                             "G": 2,
                             "T": 3,
                            }
        self.vocab_encoded_list = self.vocab_encoding()

    def vocab_encoding(self):
        vocab_encoded_list = []

        for seq in self.x_list:
            vocab_encoded_list.append([self.vocabs2index[vocab] for vocab in seq])
        return vocab_encoded_list

    def one_hot_encoding(self, output_channel, output_length) -> np.ndarray:
        x_arr = np.zeros(shape=(len(self.x_list), output_channel, output_length))      #(B, C, L)

        for i, sample in enumerate(self.vocab_encoded_list):
            for j, index in enumerate(sample):
                x_arr[i, index, j] = 1
        return np.asarray(x_arr).astype(np.float32)


class PSTNP():

    def __init__(self, x_train, y_train):
        self.alphabet = "ACGT"
        self.x_train = x_train
        self.y_train = y_train
        self.matrix = ["".join(e) for e in itertools.product(self.alphabet, repeat=3)]
        self.train_pos_list, self.train_neg_list = self.split_data_to_pos_and_neg()
        self.pos_freq = self.obtain_trinucle_freq_arr(self.train_pos_list)
        self.neg_freq = self.obtain_trinucle_freq_arr(self.train_neg_list)
        self.z_profile = self.pos_freq - self.neg_freq

    def split_data_to_pos_and_neg(self):
        pos_list = []
        neg_list = []
        for i, label in enumerate(self.y_train):
            if label == 1:
                pos_list.append(self.x_train[i])
            else:
                neg_list.append(self.x_train[i])
        return pos_list, neg_list

    def obtain_trinucle_freq_arr(self, seq_list):
        number_of_trinucleotides = 41 - 3 + 1
        frequency_arr = np.zeros((64, 39))
        for seq in seq_list:
            for j in range(number_of_trinucleotides):
                this_trinucleotide = seq[j:j+3]
                index = self.matrix.index(this_trinucleotide)
                frequency_arr[index, j] = frequency_arr[index, j] + 1
        return frequency_arr

    def extract_pstnp_feature_from_one_seq(self, seq):
        number_of_trinucleotides = 41 - 3 + 1
        seq_feature = []
        for j in range(number_of_trinucleotides):
            this_trinucleotide = seq[j:j+3]
            index = self.matrix.index(this_trinucleotide)
            seq_feature.append(self.z_profile[index, j])
        return seq_feature

    def get_pstnp_features(self, seq_list):
        final_features = []
        for seq in seq_list:
            seq_feature = self.extract_pstnp_feature_from_one_seq(seq)
            final_features.append(seq_feature)
        return np.asarray(final_features)