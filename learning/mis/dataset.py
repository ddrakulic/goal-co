"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from utils.samplers import SamplerVariousSolutionLens


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, adj_matrices, optimal_values):
        self.adj_matrices = adj_matrices
        self.optimal_values = optimal_values
        self.solution_lengths = optimal_values

    def __len__(self):
        return len(self.adj_matrices)

    def __getitem__(self, item):
        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.adj_matrices = torch.tensor(self.adj_matrices[item]).float()
        item_dict.optimal_values = self.optimal_values[item]
        item_dict.solution_lengths = self.optimal_values[item]
        item_dict.can_be_selected = torch.full((self.adj_matrices[item].shape[0], ), True)
        return item_dict


def load_dataset(filename, batch_size, datasets_size, shuffle=False, drop_last=False, what="test", ddp=False, num_nodes=100):
    data = np.load(filename)

    dataset = DataSet(data["adj_matrices"][:datasets_size], data["optimal_values"][:datasets_size])

    if what == "train":
        sampler = SamplerVariousSolutionLens(dataset)
    else:
        sampler = None
    dataset = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, sampler=sampler)
    return dataset

