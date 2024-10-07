"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import pdist, squareform
import numpy as np


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, node_coords, node_prizes, node_penalties, min_collected_prizes, optimal_solutions,
                 solution_lengths):
        self.node_coords = node_coords
        self.node_prizes = node_prizes
        self.node_penalties = node_penalties
        self.min_collected_prizes = min_collected_prizes
        self.optimal_solutions = optimal_solutions
        self.solution_lengths = solution_lengths

    def __len__(self):
        return len(self.node_coords)

    def __getitem__(self, item):
        node_coords = self.node_coords[item]

        dist_matrix = squareform(pdist(node_coords, metric='euclidean'))[:, :, None]

        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.dist_matrices = dist_matrix.astype(np.float32)
        item_dict.node_prizes = self.node_prizes[item].astype(np.float32)
        item_dict.node_penalties = self.node_penalties[item].astype(np.float32)
        item_dict.min_collected_prizes = self.min_collected_prizes[item].astype(np.float32)

        if self.solution_lengths is not None:
            item_dict.solution_lengths = self.solution_lengths[item]
        else:
            item_dict.solution_lengths = torch.Tensor([])
        item_dict.optimal_values = self.optimal_solutions[item]
        return item_dict


def load_dataset(filename, batch_size, datasets_size, shuffle=False, drop_last=True, what="test", ddp=False):
    data = np.load(filename)

    if what == "train":
        assert data["is_training_dataset"]

    solution_lengths = data["solution_lengths"][:datasets_size] if "solution_lengths" in data else None

    dataset = DataSet(data["node_coords"][:datasets_size], data["node_prizes"][:datasets_size],
                      data["node_penalties"][:datasets_size], data["min_collected_prizes"][:datasets_size],
                      data["optimal_solutions"][:datasets_size], solution_lengths)

    dataset = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last)
    return dataset

