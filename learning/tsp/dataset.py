"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import torch
import os
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from scipy.spatial.distance import pdist, squareform
import numpy as np
from torch.utils.data.dataloader import default_collate


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, node_coords=None, dist_matrices=None, objective_values=None):
        assert (node_coords is not None) ^ (dist_matrices is not None)
        self.node_coords = node_coords
        self.dist_matrices = dist_matrices
        self.objective_values = objective_values

    def __len__(self):
        if self.node_coords is not None:
            return len(self.node_coords)
        else:
            return len(self.dist_matrices)

    def __getitem__(self, item):

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

        if self.objective_values is not None:
            tour_lens = self.objective_values[item] / norm_factor
        else:
            tour_lens = np.array([])

        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.dist_matrices = torch.Tensor(dist_matrix)
        item_dict.tour_lens = tour_lens
        return item_dict


def load_dataset(path, batch_size, datasets_size, shuffle=False, drop_last=False, what="test", ddp=False, num_nodes=101):
    filename_extension = path[-3:]
    if filename_extension == "npz":
        data = np.load(path)

        if what == "train" and not data["is_training_dataset"]:
            exit("Nodes must be reordered during the training")
        node_coords = data["node_coords"][:datasets_size] if "node_coords" in data else None
        dist_matrices = data["dist_matrices"][:datasets_size] if "dist_matrices" in data else None
        if "tour_lens" in data:
            objective_values = data["tour_lens"][:datasets_size]
        elif "objective_values" in data:
            objective_values = data["objective_values"][:datasets_size]
        else:
            objective_values = None
    else:
        node_coords = None
        dist_matrices = np.memmap(os.path.join(path, "dist_matrices.dat"), dtype=np.float64, mode='r',
                                  shape=(datasets_size, num_nodes, num_nodes))
        if os.path.exists(os.path.join(path, "tour_lens.dat")):
            objective_values = np.memmap(os.path.join(path, "tour_lens.dat"), dtype=np.float64, mode='r',
                                  shape=(datasets_size,))
        else:
            objective_values = None

    # Do not use collate function in test dataset
    collate_fn = collate_func_with_sample if what == "train" else None

    dataset = DataSet(node_coords, dist_matrices, objective_values)
    if ddp:
        sampler = DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None

    dataset = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, collate_fn=collate_fn,
                         sampler=sampler)
    return dataset


def collate_func_with_sample(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    For instances of TOUR-TSP of graph size N (i.e. nb_nodes=N+1 including return to beginning node),
    this function also takes care of sampling a SUB-problem (PATH-TSP) of size 3 to N+1
    """
    nb_nodes = len(l_dataset_items[0].dist_matrices)
    subproblem_size = np.random.randint(4, nb_nodes + 1)  # between _ included and nb_nodes + 1 excluded
    begin_idx = np.random.randint(nb_nodes - subproblem_size + 1)

    l_dataset_items_new = []
    for d in l_dataset_items:
        d_new = {}
        for k, v in d.items():
            if k == "dist_matrices":
                v_ = v[begin_idx:begin_idx + subproblem_size, begin_idx:begin_idx + subproblem_size]
            else:
                v_ = v
            d_new.update({k + '_s': v_})
        l_dataset_items_new.append({**d, **d_new})

    return default_collate(l_dataset_items_new)
