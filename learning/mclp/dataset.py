"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import copy
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, matrices, num_facilities, radiuses, solutions, covering_nodes, objective_values=None):
        self.matrices = matrices
        self.num_facilities = num_facilities
        self.radiuses = radiuses
        self.solutions = solutions
        self.covered_nodes = covering_nodes
        self.objective_values = objective_values

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, item):
        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.dist_matrices = torch.Tensor(self.matrices[item]).float()
        item_dict.num_facilities = self.num_facilities[item].astype(np.int32)
        item_dict.radiuses = torch.tensor(self.radiuses[item]).float()

        if self.objective_values is not None:
            item_dict.objective_values = torch.tensor(self.objective_values[item])
        else:
            item_dict.objective_values = torch.Tensor([])

        if self.solutions is not None:
            item_dict.solutions = torch.tensor(self.solutions[item])
            item_dict.solutions = item_dict.solutions[torch.randperm(len(item_dict.solutions))]
        else:
            item_dict.solutions = torch.Tensor([])

        if self.covered_nodes is not None:
            # in training dataset, we need to make step by step data for selected and covered nodes
            # in format [0., 1., 1., 0.] -> second and third nodes are selected/covered

            num_nodes = item_dict.dist_matrices.shape[0]
            num_facilities = item_dict.num_facilities.astype(np.int32)

            curr_covering_nodes_step = torch.zeros(num_nodes)
            curr_selected_nodes_step = torch.zeros(num_nodes)
            covering_nodes = [torch.zeros(num_nodes)]
            selected_nodes = [torch.zeros(num_nodes)]

            for i in range(num_facilities):
                curr_covered_idx = self.covered_nodes[item][item_dict.solutions[i]]
                curr_covered_idx = curr_covered_idx[curr_covered_idx != -1]
                curr_covering_nodes_step[curr_covered_idx] = 1.
                curr_selected_nodes_step[item_dict.solutions[i]] = 1.
                covering_nodes.append(copy.deepcopy(curr_covering_nodes_step))
                selected_nodes.append(copy.deepcopy(curr_selected_nodes_step))
            item_dict.covering_nodes = torch.stack(covering_nodes)
            item_dict.selected_nodes = torch.stack(selected_nodes)
        else:
            item_dict.covering_nodes = torch.Tensor([])
            item_dict.selected_nodes = torch.Tensor([])

        return item_dict


def load_dataset(filename, batch_size, datasets_size, shuffle, drop_last, what="test", ddp=False):
    data = np.load(filename)

    objective_values = data["objective_values"][:datasets_size] if "objective_values" in data else None
    covering_nodes = data["covering_nodes"][:datasets_size] if "covering_nodes" in data else None
    solutions = data["solutions"][:datasets_size] if "solutions" in data else None

    dataset = DataSet(data["matrices"][:datasets_size], data["num_facilities"][:datasets_size],
                      data["radiuses"][:datasets_size], solutions, covering_nodes, objective_values)

    dataset = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
    return dataset