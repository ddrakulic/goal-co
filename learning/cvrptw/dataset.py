"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from scipy.spatial.distance import pdist, squareform
import numpy as np
from torch.utils.data.dataloader import default_collate


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, node_coords, dist_matrices, node_demands, total_capacities, service_times, time_windows,
                 tour_lens, departure_times=None, remaining_capacities=None, via_depots=None):
        assert (node_coords is not None) ^ (dist_matrices is not None)
        self.node_coords = node_coords
        self.dist_matrices = dist_matrices
        self.node_demands = node_demands
        self.total_capacities = total_capacities
        self.remaining_capacities = remaining_capacities
        self.departure_times = departure_times
        self.service_times = service_times
        self.via_depots = via_depots
        self.time_windows = time_windows
        self.tour_lens = tour_lens

    def __len__(self):
        return len(self.node_demands)

    def __getitem__(self, item):

        remaining_capacities = self.remaining_capacities[item] if self.remaining_capacities is not None else np.array([])
        via_depots = self.via_depots[item] if self.via_depots is not None else np.array([])
        departure_times = self.departure_times[item] if self.departure_times is not None else np.array([])

        if self.dist_matrices is None:
            dist_matrix = squareform(pdist(self.node_coords[item], metric='euclidean'))
        else:
            dist_matrix = self.dist_matrices[item]

        dist_matrix = np.stack([dist_matrix, dist_matrix.transpose()]).transpose(1, 2, 0)

        if self.tour_lens is not None:
            tour_lens = self.tour_lens[item]
        else:
            tour_lens = np.array([])

        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.dist_matrices = torch.Tensor(dist_matrix).int()
        item_dict.node_demands = torch.Tensor(self.node_demands[item]).int()
        item_dict.total_capacities = torch.Tensor([self.total_capacities[item]]).int()
        item_dict.time_windows = torch.Tensor(self.time_windows[item]).int()
        item_dict.service_times = torch.Tensor(self.service_times[item]).int()
        item_dict.remaining_capacities = torch.Tensor(remaining_capacities).int()
        item_dict.departure_times = torch.Tensor(departure_times).int()
        item_dict.tour_lens = self.tour_lens[item]
        item_dict.via_depots = torch.Tensor(via_depots).long()
        return item_dict


def load_dataset(filename, batch_size, datasets_size, shuffle=False, drop_last=False, what="test", ddp=False):
    data = np.load(filename)

    node_coords = data["node_coords"][:datasets_size] if "node_coords" in data else None
    dist_matrices = data["dist_matrices"][:datasets_size] if "dist_matrices" in data else None
    node_demands = data["node_demands"][:datasets_size]
    service_times = data["service_times"][:datasets_size]
    time_windows = data["time_windows"][:datasets_size]
    tour_lens = data["tour_lens"][:datasets_size]
    via_depots = data["via_depots"][:datasets_size] if "via_depots" in data else None
    total_capacities = data["total_capacities"][:datasets_size]
    departure_times = data["departure_times"][:datasets_size] if "departure_times" in data else None
    remaining_capacities = data["remaining_capacities"][:datasets_size] if "remaining_capacities" in data else None

    collate_fn = collate_func_with_sample if what == "train" else None

    dataset = DataSet(node_coords, dist_matrices, node_demands, total_capacities, service_times,
                      time_windows, tour_lens, departure_times, remaining_capacities, via_depots)
    if ddp:
        sampler = DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None

    dataset = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle,
                         collate_fn=collate_fn, sampler=sampler)
    return dataset


def collate_func_with_sample(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    For instances of TOUR-TSP of graph size N (i.e. nb_nodes=N+1 including return to beginning node),
    this function also takes care of sampling a SUB-problem (PATH-TSP) of size 3 to N+1
    """
    nb_nodes = len(l_dataset_items[0].dist_matrices)
    begin_idx = np.random.randint(0, nb_nodes - 3)  # between _ included and nb_nodes + 1 excluded

    l_dataset_items_new = []
    for d in l_dataset_items:
        d_new = {}
        for k, v in d.items():
            if type(v) == np.int64:
                v_ = v
            elif k == "total_capacities":
                v_ = v
            elif k == "departure_times" or k == "remaining_capacities":
                v_ = v[begin_idx:begin_idx+1]
            elif len(v.shape) == 1 or k == "time_windows":
                v_ = v[begin_idx:, ...]
            else:
                v_ = v[begin_idx:, begin_idx:]
            d_new.update({k+'_s': v_})
        l_dataset_items_new.append({**d, **d_new})

    return default_collate(l_dataset_items_new)

