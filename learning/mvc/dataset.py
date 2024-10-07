"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import random
import torch
import os
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
from torch.utils.data.dataloader import default_collate
from utils.samplers import SamplerVariousSolutionLens


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, adj_matrices, optimal_values=None, solution_lengths=None):
        self.adj_matrices = adj_matrices
        self.optimal_values = optimal_values
        self.solution_lengths = solution_lengths

    def __len__(self):
        return len(self.adj_matrices)

    def __getitem__(self, item):
        # From list to tensors as a DotDict
        if self.solution_lengths is not None:
            sol_len = self.solution_lengths[item]
        else:
            sol_len = np.array([])

        if self.optimal_values is not None:
            optimal_values = self.optimal_values[item]
        else:
            optimal_values = np.array([])

        adj_matrix = self.adj_matrices[item]
        if isinstance(adj_matrix, np.memmap):
            adj_matrix = np.array(adj_matrix)

        item_dict = DotDict()
        item_dict.adj_matrices = adj_matrix.astype(np.float32)
        item_dict.optimal_values = optimal_values
        item_dict.solution_lengths = sol_len
        return item_dict


def load_dataset(path, batch_size, datasets_size, shuffle, drop_last, what, ddp=False, num_nodes=100):
    filename_extension = path[-3:]
    if filename_extension == "npz":
        data = np.load(path)
        adj_matrices = data["adj_matrices"][:datasets_size]
        if what == "train":
            assert data["is_training_dataset"], "Nodes must be reordered during the training"
            solution_lengths = data["solution_lengths"][:datasets_size]
            optimal_values = None
        else:
            solution_lengths = None
            optimal_values = data["optimal_values"][:datasets_size]
    else:
        assert what == "train"
        adj_matrices = np.memmap(os.path.join(path, "adj_matrices.dat"), dtype=np.float64, mode='r',
                                 shape=(datasets_size, num_nodes, num_nodes))
        solution_lengths = np.memmap(os.path.join(path, "solution_lengths.dat"), dtype=np.int64, mode='r',
                                     shape=(datasets_size,))
        optimal_values = None

    collate_fn = collate_func if what == "train" else None

    dataset = DataSet(adj_matrices, optimal_values, solution_lengths)

    if what == "train":
        sampler = SamplerVariousSolutionLens(dataset)
        if ddp:
            sampler = DistributedSampler(sampler)
    else:
        sampler = None

    dataset = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, collate_fn=collate_fn, sampler=sampler)
    return dataset


def collate_func(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    """
    # find minimal solution len
    sol_length = np.min([el["solution_lengths"] for el in l_dataset_items])
    num_samples_nodes = random.randint(0, sol_length - 1)
    l_dataset_items_new = []
    for d in l_dataset_items:
        new_item = dict()
        to_remove = np.random.choice(d["solution_lengths"], num_samples_nodes, replace=False)
        keep_filter = torch.full([len(d["adj_matrices"])], True)
        keep_filter[to_remove] = False
        new_item["optimal_values"] = d["optimal_values"]
        new_item["adj_matrices_s"] = d["adj_matrices"][keep_filter][:, keep_filter]
        new_item["solution_probs_s"] = torch.full([len(new_item["adj_matrices_s"])], -np.inf)
        new_item["solution_probs_s"][:d["solution_lengths"] - num_samples_nodes] = 1.
        l_dataset_items_new.append(new_item)

    return default_collate(l_dataset_items_new)
