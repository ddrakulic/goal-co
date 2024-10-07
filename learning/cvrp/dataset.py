"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import numpy
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

    def __init__(self, node_coords, dist_matrices, node_demands, total_capacities, tour_lens=None,
                 remaining_capacities=None, via_depots=None, distance_constraints=None, remaining_distance_constraints=None):
        assert (node_coords is not None) ^ (dist_matrices is not None)
        self.node_coords = node_coords
        self.dist_matrices = dist_matrices
        self.node_demands = node_demands
        self.total_capacities = total_capacities
        self.remaining_capacities = remaining_capacities
        self.remaining_distance_constraints = remaining_distance_constraints
        self.distance_constraints = distance_constraints
        self.via_depots = via_depots
        self.tour_lens = tour_lens

    def __len__(self):
        return len(self.node_demands)

    def __getitem__(self, item):

        if self.remaining_capacities is not None:
            via_depots = self.via_depots[item]
            remaining_capacities = self.remaining_capacities[item].astype(numpy.float32)
        else:
            via_depots = numpy.array([])
            remaining_capacities = numpy.array([])

        if self.remaining_distance_constraints is not None:
            remaining_distance_constraints = self.remaining_distance_constraints[item].astype(numpy.float32)
        else:
            remaining_distance_constraints = numpy.array([])

        if self.distance_constraints is not None:
            distance_constraints = self.distance_constraints[item].astype(numpy.float32)
        else:
            distance_constraints = None

        if self.dist_matrices is None:
            dist_matrix = squareform(pdist(self.node_coords[item], metric='euclidean'))
        else:
            dist_matrix = self.dist_matrices[item]

            # memmap must be converted to "pure" numpy
            if isinstance(dist_matrix, np.memmap):
                dist_matrix = np.array(dist_matrix)

        dist_matrix = np.stack([dist_matrix, dist_matrix.transpose()]).transpose(1, 2, 0)

        # scale matrix values to [0, 1]
        norm_factor = dist_matrix.max()
        dist_matrix = dist_matrix / norm_factor

        if self.tour_lens is not None:
            tour_lens = self.tour_lens[item] / norm_factor
        else:
            tour_lens = np.array([])

        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.dist_matrices = torch.Tensor(dist_matrix)
        item_dict.node_demands = torch.Tensor(self.node_demands[item])
        item_dict.total_capacities = torch.tensor([self.total_capacities[item]]).int()
        item_dict.remaining_capacities = torch.Tensor(remaining_capacities).int()

        if distance_constraints is not None:
            # data for dcvrp
            item_dict.distance_constraints = distance_constraints
            item_dict.remaining_distance_constraints = torch.Tensor(remaining_distance_constraints).float()

        item_dict.tour_lens = tour_lens
        item_dict.via_depots = torch.Tensor(via_depots).long()
        return item_dict


def load_dataset(filename, batch_size, datasets_size, shuffle=False, drop_last=False, what="test", ddp=False):
    data = np.load(filename)
    if what == "train":
        assert data["is_training_dataset"]

    node_coords = data["node_coords"][:datasets_size] if "node_coords" in data else None
    dist_matrices = data["dist_matrices"][:datasets_size] if "dist_matrices" in data else None

    demands = data["node_demands"][:datasets_size]
    total_capacities = data["total_capacities"][:datasets_size]
    tour_lens = data["tour_lens"][:datasets_size]

    remaining_capacities = data["remaining_capacities"][:datasets_size] \
        if "remaining_capacities" in list(data.keys()) else None

    distance_constraints = data["distance_constraints"][:datasets_size] \
        if "distance_constraints" in list(data.keys()) else None

    remaining_distance_constraints = data["remaining_distances"][:datasets_size]\
        if "remaining_distances" in list(data.keys()) else None

    # in training dataset we have via_depots data
    via_depots = data["via_depots"][:datasets_size] if "via_depots" in list(data.keys()) else None

    # Do not use collate function in test dataset
    collate_fn = collate_func_with_sample if what == "train" else None
    dataset = DataSet(node_coords, dist_matrices, demands, total_capacities, remaining_capacities=remaining_capacities,
                      tour_lens=tour_lens, via_depots=via_depots, distance_constraints=distance_constraints,
                      remaining_distance_constraints=remaining_distance_constraints)

    if ddp:
        sampler = DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None

    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, collate_fn=collate_fn,
                         sampler=sampler)
    return dataloader


def collate_func_with_sample(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    For instances of TOUR-CVRP of graph size N (i.e. nb_nodes=N+1 including return to beginning node),
    this function also takes care of sampling a SUB-problem (PATH-TSP) of size 3 to N+1
    """
    nb_nodes = len(l_dataset_items[0].dist_matrices)
    begin_idx = np.random.randint(0, nb_nodes - 3)  # between _ included and nb_nodes + 1 excluded

    l_dataset_items_new = []
    for d in l_dataset_items:
        d_new = {}
        for k, v in d.items():
            if type(v) == numpy.float32:
                v_ = v
            elif k == "remaining_capacities":
                v_ = v[begin_idx:begin_idx+1]
            elif k == "remaining_distance_constraints":
                v_ = v[begin_idx:]
            elif k == "total_capacities":
                v_ = v
            elif k == "dist_matrices":
                v_ = v[begin_idx:, begin_idx:]
            elif k == "via_depots" or k == "node_demands":
                v_ = v[begin_idx:]
            else:
                v_ = v
            d_new.update({k + '_s': v_})
        l_dataset_items_new.append({**d, **d_new})

    return default_collate(l_dataset_items_new)

