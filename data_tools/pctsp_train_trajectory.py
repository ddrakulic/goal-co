"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""
import argparse
import os
import pickle
import numpy
from scipy.spatial.distance import pdist, squareform

SCALE = 1e7


def prepare_one_instance(node_coords, prizes, penalties, tour, from_solver=False):
    tour = tour[:-1]
    solution_len = len(tour)
    # for supervised training, keep instances with the same sol length, e.g. 50, to maximize number of optimization steps
    if from_solver and solution_len != 50:
        return None
    all_node_idx = set([i for i in range(len(node_coords))])
    gt_nodes_idx = set(tour)
    remaining_nodes = all_node_idx.difference(gt_nodes_idx)
    solution_plus_remaining = numpy.array(tour.tolist() + list(remaining_nodes) + [0])

    # reorder nodes  and prices
    node_coords = node_coords[solution_plus_remaining]
    prizes = prizes[solution_plus_remaining]
    penalties = penalties[solution_plus_remaining]

    return node_coords, prizes, penalties, solution_len
