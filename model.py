# standard imports
import numpy as np
import math

# third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Taeyeon_Net(nn.Module):

    def __init__(self, in_channel, feature_len):
        super(Taeyeon_Net, self).__init__()

        self.cbr1 = nn.Sequential(
                            nn.Conv1d(in_channel, 42, kernel_size=3, stride=1, dilation=1, padding=1),
                            nn.BatchNorm1d(42),
                            nn.LeakyReLU()
        )

        self.dropout = nn.Dropout(0.25)

        self.transition_layer1 = nn.Sequential(
                                    nn.Conv1d(42, 42, kernel_size=3, stride=2, dilation=1, padding=1),
                                    nn.BatchNorm1d(42),
                                    nn.LeakyReLU()
        )
        self.cbr2 = nn.Sequential(
                    nn.Conv1d(42, 42, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm1d(42),
                    nn.LeakyReLU()
        )

        self.transition_layer2 = nn.Sequential(
                                    nn.Conv1d(42, 42, kernel_size=3, stride=2, dilation=1, padding=1),
                                    nn.BatchNorm1d(42),
                                    nn.LeakyReLU()
        )
        self.cbr3 = nn.Sequential(
                    nn.Conv1d(42, 42, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm1d(42),
                    nn.LeakyReLU()
        )

        self.transition_layer3 = nn.Sequential(
                                    nn.Conv1d(42, 42, kernel_size=3, stride=2, dilation=1, padding=1),
                                    nn.BatchNorm1d(42),
                                    nn.LeakyReLU()
        )
        self.cbr4 = nn.Sequential(
                    nn.Conv1d(42, 64, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU()
        )
        self.transition_layer4 = nn.Sequential(
                                    nn.Conv1d(64, 64, kernel_size=3, stride=2, dilation=1, padding=1),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU()
        )
        self.classifier = nn.Linear(64*(math.ceil(math.ceil(math.ceil(math.ceil(feature_len/2)/2)/2)/2)), 2)

    def forward(self, x):

        x_out = self.cbr1(x)
        x_out = self.dropout(x_out)
        x_out = self.transition_layer1(x_out)
        x_out = self.dropout(x_out)

        x_out = self.cbr2(x_out)
        x_out = self.dropout(x_out)
        x_out = self.transition_layer2(x_out)
        x_out = self.dropout(x_out)

        x_out = self.cbr3(x_out)
        x_out = self.dropout(x_out)
        x_out = self.transition_layer3(x_out)
        x_out = self.dropout(x_out)

        x_out = self.cbr4(x_out)
        x_out = self.dropout(x_out)
        x_out = self.transition_layer4(x_out)
        x_out = self.dropout(x_out)

        x_out = torch.flatten(x_out, start_dim=1, end_dim=-1)

        x_out = self.classifier(x_out)

        return x_out