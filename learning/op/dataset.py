"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from scipy.spatial.distance import pdist, squareform
import numpy as np
from torch.utils.data.dataloader import default_collate
from utils.samplers import SamplerVariousSolutionLens


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, node_coords, node_values, upper_bounds, collected_rewards, solution_lengths):
        self.node_coords = node_coords
        self.node_values = node_values
        self.upper_bounds = upper_bounds
        self.collected_rewards = collected_rewards
        self.solution_lengths = solution_lengths

    def __len__(self):
        return len(self.node_coords)

    def __getitem__(self, item):
        node_coords = self.node_coords[item]

        dist_matrix = squareform(pdist(node_coords, metric='euclidean'))
        dist_matrix = np.stack([dist_matrix, dist_matrix.transpose()]).transpose(1, 2, 0)

        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.dist_matrices = dist_matrix.astype(np.float32)
        item_dict.node_values = self.node_values[item].astype(np.float32)
        item_dict.upper_bounds = self.upper_bounds[item].astype(np.float32)
        if self.solution_lengths is not None:
            item_dict.solution_lengths = self.solution_lengths[item]
        else:
            item_dict.solution_lengths = torch.Tensor([])
        item_dict.collected_rewards = self.collected_rewards[item]
        return item_dict


def load_dataset(filename, batch_size, datasets_size, shuffle=False, drop_last=True, what="test", ddp=False):
    data = np.load(filename)

    if what == "train":
        assert data["is_training_dataset"]

    collate_fn = collate_func if what == "train" else None
    solution_lengths = data["solution_lengths"][:datasets_size] if "solution_lengths" in data else None

    dataset = DataSet(data["node_coords"][:datasets_size], data["node_values"][:datasets_size],
                      data["upper_bounds"][:datasets_size], data["collected_rewards"][:datasets_size],
                      solution_lengths)

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
    num_sampled_nodes = random.randint(0, sol_length - 2)
    l_dataset_items_new = []
    for d in l_dataset_items:
        d_new = dict()
        for k, v in d.items():
            if k == "node_coords" or k == "node_values":
                v_ = v[num_sampled_nodes:, ...]
            elif k == "dist_matrices":
                v_ = v[num_sampled_nodes:, num_sampled_nodes:]
            elif k == "upper_bounds":
                v_ = (v - sum([d["dist_matrices"][i, i+1, 0] for i in range(0, num_sampled_nodes)])).astype(np.float32)
                assert v_ > 0
            else:
                v_ = v
            d_new.update({k + '_s': v_})
        l_dataset_items_new.append({**d, **d_new})

    return default_collate(l_dataset_items_new)
