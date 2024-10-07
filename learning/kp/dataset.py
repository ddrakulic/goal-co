"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
from torch.utils.data.dataloader import default_collate
from utils.samplers import SamplerVariousSolutionLens


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, weights, values, capacities, solution_lengths, optimal_values, scale):
        self.capacities = capacities
        self.weights = weights
        self.values = values
        self.solution_lengths = solution_lengths
        self.optimal_values = optimal_values
        self.scale = scale

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.remaining_capacities = self.capacities[item].astype(np.float32)
        item_dict.weights = torch.Tensor(self.weights[item]).float()
        item_dict.values = torch.Tensor(self.values[item]).float()
        if self.solution_lengths is not None:
            item_dict.solution_lengths = self.solution_lengths[item]
        else:
            item_dict.solution_lengths = torch.tensor(-1.)
        item_dict.optimal_values = self.optimal_values[item]
        item_dict.scale = self.scale.astype(np.float32)
        return item_dict


def load_dataset(filename, batch_size, datasets_size, shuffle, drop_last, what, ddp=False):
    data = np.load(filename)

    collate_fn = collate_func if what == "train" else None
    solution_lengths = data["solution_lengths"][:datasets_size] if "solution_lengths" in data else None

    dataset = DataSet(data["weights"][:datasets_size], data["values"][:datasets_size],
                      data["capacities"][:datasets_size], solution_lengths, data["optimal_values"][:datasets_size],
                      data["scale"])

    if what == "train":
        sampler = SamplerVariousSolutionLens(dataset)
        if ddp:
            sampler = DistributedSampler(sampler)
    else:
        sampler = None

    dataset = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, sampler=sampler, collate_fn=collate_fn)
    return dataset


def collate_func(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    """
    sol_length = np.min([el["solution_lengths"] for el in l_dataset_items])
    num_samples_nodes = random.randint(0, sol_length - 1)
    l_dataset_items_new = []
    for d in l_dataset_items:
        new_item = dict()
        to_remove = np.random.choice(d["solution_lengths"], num_samples_nodes, replace=False)

        keep_filter = torch.full([len(d["weights"])], True)
        keep_filter[to_remove] = False
        new_item["weights_s"] = d["weights"][keep_filter]
        new_item["values_s"] = d["values"][keep_filter]
        new_item["remaining_capacities_s"] = d["remaining_capacities"] - torch.tensor(1.) * sum(d["weights"][~keep_filter])
        new_item["optimal_values_s"] = d["optimal_values"]
        new_item["scale_s"] = d["scale"]
        new_item["solution_probs_s"] = torch.full([len(new_item["weights_s"])], -np.inf)
        new_item["solution_probs_s"][:d["solution_lengths"] - num_samples_nodes] = 1.
        l_dataset_items_new.append(new_item)

    return default_collate(l_dataset_items_new)
