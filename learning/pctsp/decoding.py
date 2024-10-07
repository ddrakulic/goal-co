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
from utils.data_manipulation import prepare_routing_data


@dataclass
class PCTSPSubProblem:
    problem_name: str
    dist_matrices: Tensor
    node_prizes: Tensor
    node_penalties: Tensor
    remaining_to_collect: Tensor
    original_idxs: Tensor


def decode(problem_name: str, problem_data: list, net: Module, beam_size: int = 1, knns: int = -1, sample: bool = False) -> tuple[Tensor, Tensor]:
    dist_matrices, node_prizes, node_penalties, min_collected_prizes, _, _ = problem_data

    if beam_size == 1:
        tours, objectives = decoding_loop(problem_name, dist_matrices, node_prizes, node_penalties, min_collected_prizes,
                                          net, knns, sample)
    else:
        raise NotImplementedError

    return objectives, tours.cpu().numpy()


def decoding_loop(problem_name: str, dist_matrices: Tensor, node_prizes: Tensor, node_penalties: Tensor,
                  min_collected_prizes: Tensor, net: Module, knns: int, sample: bool) -> tuple[Tensor, Tensor]:
    bs, num_nodes, _, _ = dist_matrices.shape
    original_idxs = torch.arange(num_nodes, device=dist_matrices.device)[None, :].repeat(bs, 1)

    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=dist_matrices.device)
    done_instances = torch.full((bs, ), False, device=dist_matrices.device)
    sub_problem = PCTSPSubProblem(problem_name, dist_matrices, node_prizes, node_penalties, min_collected_prizes,
                                  original_idxs)

    # we sum all penalties and then add tour len and substract penalty of selected node
    penalties = node_penalties.sum(dim=-1)
    path_lengths = torch.zeros(bs, device=dist_matrices.device)

    for dec_pos in range(1, num_nodes - 1):
        idx_selected, sub_problem = decoding_step(sub_problem, net, knns, sample)
        done_instances[idx_selected == num_nodes - 1] = True
        done_instances[sub_problem.remaining_to_collect <= 0] = True
        paths[:, dec_pos] = idx_selected
        paths[done_instances, dec_pos] = 0
        penalties -= node_penalties[torch.arange(bs), paths[:, dec_pos]]
        path_lengths += dist_matrices.squeeze(-1)[torch.arange(bs), paths[:, dec_pos - 1], paths[:, dec_pos]]

        if torch.all(done_instances):
            break
    objective_values = path_lengths + penalties
    return paths, objective_values


def decoding_step(sub_problem: PCTSPSubProblem, net: Module, knns: int, sample: bool) -> (Tensor, PCTSPSubProblem):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    if sample:
        probs = torch.softmax(scores, dim=-1)
        selected_nodes = torch.tensor([np.random.choice(np.arange(probs.shape[1]),
                                                        p=prob.cpu().numpy()) for prob in probs]).to(probs.device)[:, None]
    else:
        selected_nodes = torch.argmax(scores, dim=1, keepdim=True)  # (b, 1)
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, selected_nodes)
    return idx_selected_original.squeeze(1), reformat_subproblem_for_next_step(sub_problem, selected_nodes)


def prepare_input_and_forward_pass(sub_problem: PCTSPSubProblem, net: Module, knns: int) -> Tensor:
    bs, num_nodes = sub_problem.node_prizes.shape
    if 0 < knns < num_nodes:
        raise NotImplementedError
    else:
        data = [sub_problem.dist_matrices, sub_problem.node_prizes, sub_problem.node_penalties,
                sub_problem.remaining_to_collect, None, None]
        node_features, matrices, problem_data = prepare_routing_data(data, "pctsp")

        scores = net(node_features, matrices, problem_data)

    return scores


def reformat_subproblem_for_next_step(sub_problem: PCTSPSubProblem, idx_selected: Tensor) -> PCTSPSubProblem:
    # Example: current_subproblem: [a b c d e] => (model selects d) => next_subproblem: [d b c e]
    bs, subpb_size, _, _ = sub_problem.dist_matrices.shape
    is_selected = torch.arange(
        subpb_size, device=sub_problem.dist_matrices.device).unsqueeze(dim=0).repeat(bs, 1) == idx_selected.repeat(1, subpb_size)

    selected_prizes = sub_problem.node_prizes[torch.arange(bs), idx_selected.squeeze(-1)]
    next_remaining_to_collect = sub_problem.remaining_to_collect - selected_prizes

    next_original_idxs = remove_origin_and_reorder_tensor(sub_problem.original_idxs, is_selected)
    next_node_prizes = remove_origin_and_reorder_tensor(sub_problem.node_prizes, is_selected)
    next_node_penalties = remove_origin_and_reorder_tensor(sub_problem.node_penalties, is_selected)

    next_dist_matrices = remove_origin_and_reorder_matrix(sub_problem.dist_matrices, is_selected)

    return PCTSPSubProblem("pctsp", next_dist_matrices, next_node_prizes, next_node_penalties,
                           next_remaining_to_collect, next_original_idxs)