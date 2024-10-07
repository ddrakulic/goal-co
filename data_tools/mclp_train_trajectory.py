"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import numpy as np

def fill_array(array, max_len):
    remaining = max_len - len(array)
    if remaining == 0:
        a = np.array(array).astype(np.int32).tolist()
    else:
        a = np.concatenate([np.array(array).astype(np.int32), np.full(remaining, -1)]).tolist()
    return a


def prepare_one_instance(dist_matrix, radius, solution):
    # we need to count how many location is covered with each facility

    # cover_nodes = list of N lists, K-th list contains list of nodes that covers facility in K
    problem_size = dist_matrix.shape[0]
    cover_nodes = [list()] * problem_size
    all_covered_nodes = list()
    for facility_node in solution:
        covered_nodes = np.where(dist_matrix[facility_node] <= radius)[0].tolist()
        cover_nodes[facility_node] = covered_nodes
        all_covered_nodes.extend(covered_nodes)
    num_covered = len(set(all_covered_nodes))
    max_len = max([len(i) for i in cover_nodes])
    trim_cover_nodes = list()
    for node in cover_nodes:
        trim_cover_nodes.append(fill_array(node, max_len))

    return np.array(trim_cover_nodes), num_covered
