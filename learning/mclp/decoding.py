"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""
from dataclasses import dataclass
import numpy
import torch
from torch import Tensor
from torch.nn import Module
import numpy as np
from utils.data_manipulation import prepare_graph_data


@dataclass
class MCLPSubPb:
    """
    In decoding, we successively apply model on progressively smaller sub-problems.
    In each sub-problem, we keep track of the indices of each node in the original full-problem.
    """
    dist_matrices: Tensor
    radiuses: Tensor
    num_facilities: Tensor
    covered_nodes: Tensor
    already_selected: Tensor


def decode(problem_name: str, data: list, net: Module, beam_size: int = 1, knns: int = 0, sample: bool = False) -> tuple[Tensor, numpy.array]:

    if beam_size == 1:
        rewards, selected_items = greedy_decoding(data, net, knns, sample)
    else:
        raise NotImplementedError
    return rewards, selected_items.cpu().numpy()


def greedy_decoding(data: list, net: Module, knns: int, sample: bool):
    dist_matrices, num_facilities, radiuses, _, _, _, _ = data
    bs, problem_size, _ = dist_matrices.shape
    covered_nodes = torch.zeros((bs, problem_size), device=dist_matrices.device)
    already_selected = torch.zeros((bs, problem_size), device=dist_matrices.device)

    sub_problem = MCLPSubPb(dist_matrices, radiuses, num_facilities, covered_nodes, already_selected)
    for dec_pos in range(problem_size):
        idx_selected = greedy_decoding_step(sub_problem, net, knns, sample)
        sub_problem.already_selected[torch.arange(bs), idx_selected.squeeze(-1)] = 1.
        # find covered nodes, dist(node, selected_node) <= radius
        covered = sub_problem.dist_matrices[torch.arange(bs), idx_selected.squeeze(-1)] <= sub_problem.radiuses.unsqueeze(-1)
        sub_problem.covered_nodes[covered] = 1.

        # todo: this works just for datasets with same num_facilities!
        completed = torch.count_nonzero(sub_problem.already_selected, dim=-1) == sub_problem.num_facilities

        if completed.all():
            break

    # compute rewards
    rewards = sub_problem.covered_nodes.sum(dim=-1)

    return rewards, sub_problem.already_selected


def greedy_decoding_step(sub_problem: MCLPSubPb, net: Module, knns: int, sample: bool) -> (Tensor, MCLPSubPb):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    if sample:
        probs = torch.softmax(scores, dim=-1)
        selected_nodes = torch.tensor([np.random.choice(np.arange(probs.shape[1]),
                                                       p=prob.cpu().numpy()) for prob in probs]).to(probs.device)[:, None]
    else:
        selected_nodes = torch.argmax(scores, dim=1, keepdim=True)

    return selected_nodes


def prepare_input_and_forward_pass(sub_problem: MCLPSubPb, net: Module, knns: int) -> Tensor:
    bs, num_nodes, _ = sub_problem.dist_matrices.shape
    if 0 < knns < num_nodes:
        # find K nearest neighbors of the current node
        raise NotImplementedError
    else:
        data = [sub_problem.dist_matrices, sub_problem.num_facilities, sub_problem.radiuses,
                sub_problem.covered_nodes, sub_problem.already_selected, None, None]
        node_features, matrices, problem_data = prepare_graph_data(data, "mclp")
        scores = net(node_features, matrices, problem_data)
    return scores
