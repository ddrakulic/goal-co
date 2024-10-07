"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from dataclasses import dataclass
import torch
import numpy as np
from torch import Tensor
from torch.nn import Module
from utils.data_manipulation import prepare_graph_data


@dataclass
class KPSubPb:
    weights: Tensor
    values: Tensor
    remaining_capacities: Tensor
    scale: Tensor
    original_idxs: Tensor


def decode(problem_name: str, problem_data: list, net: Module, beam_size: int = 1,
           knns: int = -1, sample: bool = False) -> tuple[Tensor, Tensor]:
    weights, values, total_capacities, scale, _, _ = problem_data
    if beam_size == 1:
        rewards, selected_items = decoding(weights, values, total_capacities, scale, net, knns)
    else:
        raise NotImplementedError
    return rewards, selected_items


def decoding(weights: Tensor, values: Tensor, capacities: Tensor, scale: Tensor,
             net: Module, knns: Tensor):
    bs, problem_size = values.shape
    original_idxs = torch.tensor(list(range(problem_size)), device=values.device)[None, :].repeat(bs, 1)

    sub_problem = KPSubPb(weights, values, capacities, scale, original_idxs)
    selected_items = torch.full((bs, problem_size), -1, dtype=torch.long, device=values.device)
    for dec_pos in range(problem_size):
        scores, idx_selected, sub_problem = decoding_step(sub_problem, net, knns)
        if torch.count_nonzero(idx_selected.flatten() == -1) == bs:
            # all tours are done!
            break
        selected_items[:, dec_pos] = idx_selected

    # trick: add 0 at the end of weights and values, for "selected nodes" with index -1
    selected_items[selected_items == -1] = problem_size
    weights = torch.cat([weights, torch.zeros(bs, 1, device=weights.device)], dim=-1)
    values = torch.cat([values, torch.zeros(bs, 1, device=values.device)], dim=-1)
    assert torch.all(torch.gather(weights, 1, selected_items).sum(dim=1) <= capacities)

    rewards = torch.gather(values, 1, selected_items).sum(dim=1)

    return rewards, selected_items


def decoding_step(sub_problem: KPSubPb, net: Module, knns: int) -> (Tensor, KPSubPb):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    idx_selected = torch.argmax(scores, dim=1, keepdim=True)
    # if all possible choices are masked (= -inf), then instance is solved
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected)
    idx_selected_original[scores.max(dim=-1)[0] == -np.inf] = -1
    return scores, idx_selected_original.squeeze(1), reformat_subproblem_for_next_step(sub_problem, idx_selected)


def prepare_input_and_forward_pass(sub_problem: KPSubPb, net: Module, knns: int) -> Tensor:
    bs, num_nodes = sub_problem.weights.shape
    if 0 < knns < num_nodes:
        ratios = sub_problem.values / sub_problem.weights
        knn_indices = torch.topk(ratios, k=knns, dim=-1)[1]

        knn_node_values = torch.gather(sub_problem.values, 1, knn_indices)
        knn_node_weights = torch.gather(sub_problem.weights, 1, knn_indices)
        data = [knn_node_weights, knn_node_values, sub_problem.remaining_capacities, sub_problem.scale, None, None]
        node_features, matrices, problem_data = prepare_graph_data(data, "kp")
        knn_scores = net(node_features, matrices, problem_data)
        # create result tensor for scores with all -inf elements
        scores = torch.full((bs, num_nodes), -np.inf, device=knn_scores.device)
        # and put computed scores for KNNs
        scores = torch.scatter(scores, 1, knn_indices, knn_scores)
    else:
        data = [sub_problem.weights, sub_problem.values, sub_problem.remaining_capacities,
                sub_problem.scale, None, None]
        node_features, matrices, problem_data = prepare_graph_data(data, "kp")
        scores = net(node_features, matrices, problem_data)
    return scores


def reformat_subproblem_for_next_step(sub_problem: KPSubPb, idx_selected: Tensor) -> KPSubPb:
    # Example: current_subproblem: [a b c d e] => (model selects d) => next_subproblem: [d b c e]
    subpb_size = sub_problem.weights.shape[1]
    bs = sub_problem.weights.shape[0]
    is_selected = torch.arange(
        subpb_size, device=sub_problem.weights.device).unsqueeze(dim=0).repeat(bs, 1) == idx_selected.repeat(1,
                                                                                                             subpb_size)
    # remaining items = the rest
    next_capacities = sub_problem.remaining_capacities - sub_problem.weights[is_selected]
    next_values = sub_problem.values[~is_selected].reshape((bs, -1))
    next_weights = sub_problem.weights[~is_selected].reshape((bs, -1))
    next_original_idxs = sub_problem.original_idxs[~is_selected].reshape((bs, -1))
    return KPSubPb(next_weights, next_values, next_capacities, sub_problem.scale, next_original_idxs)
