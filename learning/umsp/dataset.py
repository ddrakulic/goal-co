"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import torch
import random
from torch.utils.data.dataloader import default_collate


class DotDict(dict):
    def __init__(self, **kwds):
        super().__init__()
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, execution_times, solutions, optimal_values, scales):
        self.execution_times = execution_times
        self.solutions = solutions
        self.optimal_values = optimal_values
        self.scales = scales

    def __len__(self):
        return len(self.execution_times)

    def __getitem__(self, item):
        # From list to tensors as a DotDict
        item_dict = DotDict()

        execution_times = torch.Tensor(self.execution_times[item])
        solutions = torch.Tensor(self.solutions[item])
        sorted_matrix_indices = np.argsort((-execution_times).max(axis=1))[0]
        execution_times = execution_times[sorted_matrix_indices]
        solutions = solutions[sorted_matrix_indices]

        item_dict.execution_times = execution_times
        item_dict.solutions = solutions
        item_dict.optimal_values = self.optimal_values[item]
        item_dict.scales = self.scales[item].astype(np.float32)
        item_dict.machine_states = torch.zeros(len(item_dict.execution_times))
        return item_dict


def load_dataset(path, batch_size, datasets_size, shuffle=False, drop_last=False, what="test", ddp=False):

    data = np.load(path)

    dataset = DataSet(data["processing_times"][:datasets_size],
                      data["solutions"][:datasets_size],
                      data["optimal_values"][:datasets_size],
                      data["scales"][:datasets_size])

    collate_fn = collate_func if what == "train" else None

    if ddp:
        sampler = DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None

    dataset = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, collate_fn=collate_fn,
                         sampler=sampler)
    return dataset


def collate_func(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    """
    num_jobs = len(l_dataset_items[0]["execution_times"])
    num_removed_jobs = random.randint(0, num_jobs - 2)
    l_dataset_items_new = []
    for d in l_dataset_items:
        new_item = dict()

        new_item["execution_times_s"] = d["execution_times"][num_removed_jobs:]
        new_item["solutions_s"] = d["solutions"][num_removed_jobs:]
        new_item["optimal_values_s"] = d["optimal_values"]
        new_item["scales_s"] = d["scales"]
        machine_states = (d["execution_times"][:num_removed_jobs] * d["solutions"][:num_removed_jobs]).sum(axis=0)
        new_item["machine_states_s"] = machine_states

        l_dataset_items_new.append(new_item)

    return default_collate(l_dataset_items_new)
