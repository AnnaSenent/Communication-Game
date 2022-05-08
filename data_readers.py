"""
This class takes a text file as input and converts it into the format required
by the sum game, a map-style dataset that maps the input integers to one-hot vectors
and stores them in a data-frame.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class SumDataset(Dataset):

    def __init__(self, path, N, n_integers):
        data = np.loadtxt(path, dtype=int_)

        self.data = data
        self.dataset = torch.empty((len(self.data), N*n_integers+1))
        self.labels = torch.empty((len(self.data), N*n_integers+1))

        for i in range(len(self.data)):
            initialize = torch.zeros(N*n_integers+1)
            initialize[int(self.data[i][0])] = 1
            initialize[int(self.data[i][1]) + N] = 1
            self.dataset[i] = initialize

            label = torch.zeros(N*n_integers+1)
            label[int(self.data[i][-1])] = 1
            self.labels[i] = label

    def get_n_features(self):
        return self.dataset[0].size(0)

    def get_dataset(self):
        return self.dataset

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id_item):
        return self.dataset[id_item], self.labels[id_item]


