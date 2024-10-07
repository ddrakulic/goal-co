"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from learning.reformat_subproblems import remove_origin_and_reorder_matrix, remove_origin_and_reorder_tensor
from utils.misc import compute_tour_lens
from utils.data_manipulation import prepare_routing_data


@dataclass
class OPSubPb:
    problem_name: str
    dist_matrices: Tensor
    node_values: Tensor
    upper_bounds: Tensor
    original_idxs: Tensor


def decode(problem_name: str, problem_data: list, net: Module, beam_size: int, knns: int) -> tuple[Tensor, Tensor]:
    dist_matrices, node_values, upper_bounds, _ = problem_data
    if beam_size == 1:
        tours, collected_rewards = greedy_decoding_loop(dist_matrices, node_values, upper_bounds, net, knns)
    else:
        raise NotImplementedError

    distances = compute_tour_lens(tours, dist_matrices[..., 0])
    
    assert torch.all(distances <= upper_bounds + 1e-4)

    return collected_rewards, tours


def greedy_decoding_loop(dist_matrices: Tensor, node_values: Tensor, upper_bounds: Tensor, net: Module,
                         knns: int, sample=False) -> tuple[Tensor, Tensor]:
    bs, num_nodes, _, _ = dist_matrices.shape
    original_idxs = torch.tensor(list(range(num_nodes)), device=dist_matrices.device)[None, :].repeat(bs, 1)
    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=dist_matrices.device)
    collected_rewards = torch.zeros(bs, device=dist_matrices.device)
    sub_problem = OPSubPb("op", dist_matrices, node_values, upper_bounds, original_idxs)
    for dec_pos in range(1, num_nodes - 1):
        idx_selected, sub_problem = greedy_decoding_step(sub_problem, net, knns)
        paths[:, dec_pos] = idx_selected
        collected_rewards += node_values[torch.arange(bs), idx_selected]
        if torch.count_nonzero(idx_selected.flatten() == -1) == bs:
            # all tours are done!
            break

    return paths, collected_rewards


def greedy_decoding_step(sub_problem: OPSubPb, net: Module, knns: int) -> (Tensor, OPSubPb):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    idx_selected = torch.argmax(scores, dim=1, keepdim=True)  # (b, 1)
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected)
    idx_selected_original[scores.max(dim=-1)[0] == -np.inf] = -1
    return idx_selected_original.squeeze(1), reformat_subproblem_for_next_step(sub_problem, idx_selected)


def prepare_input_and_forward_pass(sub_problem: OPSubPb, net: Module, knns: int) -> Tensor:
    bs, num_nodes = sub_problem.node_values.shape
    if 0 < knns < num_nodes:
        raise NotImplementedError
    else:
        data = [sub_problem.dist_matrices, sub_problem.node_values, sub_problem.upper_bounds, None]
        node_features, matrices, problem_data = prepare_routing_data(data, "op")
        scores = net(node_features, matrices, problem_data)
    return scores


def reformat_subproblem_for_next_step(sub_problem: OPSubPb, idx_selected: Tensor) -> OPSubPb:
    # Example: current_subproblem: [a b c d e] => (model selects d) => next_subproblem: [d b c e]
    bs, subpb_size, _, _ = sub_problem.dist_matrices.shape
    is_selected = torch.arange(
        subpb_size, device=sub_problem.dist_matrices.device).unsqueeze(dim=0).repeat(bs, 1) == idx_selected.repeat(1, subpb_size)

    next_original_idxs = remove_origin_and_reorder_tensor(sub_problem.original_idxs, is_selected)
    next_node_values = remove_origin_and_reorder_tensor(sub_problem.node_values, is_selected)

    next_upper_bounds = sub_problem.upper_bounds - sub_problem.dist_matrices[:, 0, :, 0][is_selected]

    next_dist_matrices = remove_origin_and_reorder_matrix(sub_problem.dist_matrices, is_selected)

    return OPSubPb("op", next_dist_matrices, next_node_values, next_upper_bounds, next_original_idxs)