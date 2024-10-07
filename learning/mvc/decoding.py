"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from dataclasses import dataclass, asdict
import torch
import numpy as np
from torch import Tensor
from torch.nn import Module
from utils.data_manipulation import prepare_graph_data


@dataclass
class MVCSubPb:
    problem_name: str
    adj_matrices: Tensor
    original_idxs: Tensor

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


def decode(problem_name: str, problem_data: Tensor, net: Module, beam_size: int = 1, knns: int = -1) -> Tensor:
    adj_matrices = problem_data[0]
    if beam_size == 1:
        num_selected_nodes, selected_items = greedy_decoding(adj_matrices, net, knns)
    else:
        raise NotImplementedError
    return num_selected_nodes, selected_items


def greedy_decoding(adj_matrices: Tensor, net: Module, knns):
    bs, problem_size, _ = adj_matrices.shape
    original_idxs = torch.tensor(list(range(problem_size)), device=adj_matrices.device)[None, :].repeat(bs, 1)

    sub_problem = MVCSubPb("mvc", adj_matrices, original_idxs)
    selected_nodes = torch.full((bs, problem_size), -1, dtype=torch.long, device=adj_matrices.device)
    for dec_pos in range(problem_size):
        idx_selected, sub_problem = greedy_decoding_step(sub_problem, net, knns)
        if torch.count_nonzero(idx_selected.flatten() == -1) == bs:
            # all tours are done!
            break
        selected_nodes[:, dec_pos] = idx_selected

    num_selected = torch.count_nonzero(selected_nodes != -1, axis=-1).float()
    # check is there is no remaining edges
    assert sub_problem.adj_matrices.sum() == 0

    return num_selected, selected_nodes


def greedy_decoding_step(sub_problem: MVCSubPb, net: Module, knns: Tensor) -> (Tensor, MVCSubPb):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)

    idx_selected = torch.argmax(scores, dim=1, keepdim=True)
    # if all possible choices are masked (= -inf), than instance is solved
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected)
    idx_selected_original[scores.max(dim=-1)[0] == -np.inf] = -1
    return idx_selected_original.squeeze(1), reformat_subproblem_for_next_step(sub_problem, idx_selected)


def prepare_input_and_forward_pass(sub_problem: MVCSubPb, net: Module, knns: int) -> Tensor:
    bs, num_nodes, _ = sub_problem.adj_matrices.shape
    data = [sub_problem.adj_matrices, None, None]
    node_features, adj_matrix, problem_data = prepare_graph_data(data, "mvc")
    scores = net(node_features, adj_matrix, problem_data)
    return scores


def reformat_subproblem_for_next_step(sub_problem: MVCSubPb, idx_selected: Tensor) -> MVCSubPb:
    # Example: current_subproblem: [a b c d e] => (model selects d) => next_subproblem: [d b c e]
    bs, subpb_size, _ = sub_problem.adj_matrices.shape
    is_selected = torch.arange(
        subpb_size, device=sub_problem.adj_matrices.device).unsqueeze(dim=0).repeat(bs, 1) == idx_selected.repeat(1,
                                                                                                                  subpb_size)
    next_adj_matrices = sub_problem.adj_matrices[~is_selected].reshape((bs, -1, subpb_size))
    next_adj_matrices = next_adj_matrices.transpose(1, 2)[~is_selected].reshape((bs, -1, subpb_size-1))

    next_original_idxs = sub_problem.original_idxs[~is_selected].reshape((bs, -1))
    return MVCSubPb("mvc", next_adj_matrices, next_original_idxs)
