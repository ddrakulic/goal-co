"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import copy
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from learning.reformat_subproblems import remove_origin_and_reorder_tensor, remove_origin_and_reorder_matrix
from utils.data_manipulation import prepare_data
from utils.misc import compute_tour_lens

@dataclass
class VRPSubPb:
    problem_name: str
    dist_matrices: Tensor
    node_demands: Tensor
    remaining_capacities: Tensor
    remaining_distances: Tensor
    original_idxs: Tensor

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


def reconstruct_tours(paths: Tensor, via_depots: Tensor) -> list[list[int]]:
    bs = paths.shape[0]
    complete_paths = [[0] for _ in range(bs)]
    for pos in range(1, paths.shape[1]):
        nodes_to_add = paths[:, pos].tolist()
        for instance in (via_depots[:, pos]).nonzero().squeeze(-1).cpu().numpy():
            if pos == 1: # during the sampling, model can choose "via-depot" action
                continue
            complete_paths[instance].append(0)
        for instance in range(bs):
            complete_paths[instance].append(nodes_to_add[instance])

    return complete_paths


def decode(problem: str, data: list, net: Module, beam_size: int = 1, knns: int = -1,
           make_tours: bool = True, sample: bool = False) -> tuple[Tensor, list[list[int]]]:
    if beam_size == 1:
        paths, via_depots = decoding_loop(problem, data, net, knns, sample)
    else:
        paths, via_depots = beam_search_decoding_loop(problem, data, net, beam_size, knns)

    bs, num_nodes, _, _ = data[0].shape
    assert paths.sum(dim=1).sum() == paths.shape[0] * .5 * (num_nodes - 2) * (num_nodes - 1)

    if make_tours:
        paths = reconstruct_tours(paths, via_depots)
    else:
        paths = None
    tour_lens = list()
    for b in range(bs):
        tour_lens.append(compute_tour_lens(torch.tensor(paths[b])[None, :], data[0][b, ..., 0][None, :]))
    # double check, compute tour lens
    return torch.stack(tour_lens).squeeze(-1), paths


def decoding_loop(problem_name: str, problem_data: list, net: Module,
                  knns: int, sample: bool) -> tuple[Tensor, Tensor]:
    bs, num_nodes, _, _ = problem_data[0].shape

    if problem_name == "dcvrp":
        dist_matrices, node_demands, total_capacities, _, _, distance_constraints, _ = problem_data
        remaining_distances = distance_constraints
    else:
        dist_matrices, node_demands, total_capacities, _, _, _ = problem_data
        remaining_distances = None

    original_idxs = torch.tensor(list(range(num_nodes)), device=dist_matrices.device)[None, :].repeat(bs, 1)
    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=dist_matrices.device)
    via_depots = torch.full((bs, num_nodes), False, dtype=torch.bool, device=dist_matrices.device)

    sub_problem = VRPSubPb(problem_name, dist_matrices, node_demands, total_capacities, remaining_distances, original_idxs)
    for dec_pos in range(1, num_nodes - 1):
        idx_selected, via_depot, sub_problem = decoding_step(sub_problem, net, knns, sample)
        paths[:, dec_pos] = idx_selected
        via_depots[:, dec_pos] = via_depot

    assert (sub_problem.remaining_capacities >= 0).all()
    if problem_name == "dcvrp":
        assert (sub_problem.remaining_distances >= 0).all()

    return paths, via_depots


def decoding_step(sub_problem: VRPSubPb, net: Module, knns: int, sample: bool) -> (Tensor, VRPSubPb):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    if sample:
        probs = torch.softmax(scores, dim=-1)
        selected_nodes = torch.tensor([np.random.choice(np.arange(probs.shape[1]),
                                                        p=prob.cpu().numpy()) for prob in probs]).to(probs.device)[:, None]
    else:
        selected_nodes = torch.argmax(scores, dim=1, keepdim=True)

    idx_selected = torch.div(selected_nodes, 2, rounding_mode='trunc')
    via_depot = (selected_nodes % 2 == 1)
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected)

    new_subproblem, via_depot = reformat_subproblem_for_next_step(sub_problem, idx_selected, via_depot)

    return idx_selected_original.squeeze(1), via_depot.squeeze(1), new_subproblem


def prepare_input_and_forward_pass(sub_problem: VRPSubPb, net: Module, knns: int) -> Tensor:
    # find K nearest neighbors of the current node
    bs, num_nodes, _, num_features = sub_problem.dist_matrices.shape


    if 0 < knns < num_nodes:
        knn_indices = torch.topk(sub_problem.dist_matrices[:, :-1, 0, 0], k=knns - 1, dim=-1, largest=False).indices
        # and add it manually

        knn_indices = torch.cat([knn_indices, torch.full([bs, 1], num_nodes - 1, device=knn_indices.device)], dim=-1)

        knn_node_demands = torch.gather(sub_problem.node_demands, 1, knn_indices)

        knn_dist_matrices = torch.gather(sub_problem.dist_matrices, 1,
                                         knn_indices[..., None, None].repeat(1, 1, num_nodes, num_features))
        knn_dist_matrices = torch.gather(knn_dist_matrices, 2,
                                         knn_indices[:, None, :, None].repeat(1, knns, 1, num_features))

        knn_dist_matrices = (knn_dist_matrices /
                             knn_dist_matrices.reshape(bs, -1).amax(dim=-1)[:, None, None, None].repeat(1, knns, knns, num_features))

        data = [knn_dist_matrices, knn_node_demands, sub_problem.remaining_capacities[:, 0][..., None],
                sub_problem.remaining_capacities[:, -1][..., None], None, None]
        node_features, edge_features, problem_data = prepare_data(data, sub_problem.problem_name)

        knn_scores = net(node_features, edge_features, problem_data)  # (b, seq)

        # create result tensor for scores with all -inf elements
        scores = torch.full((bs, 2 * num_nodes), -np.inf, device=node_features.device)
        double_knn_indices = torch.zeros([knn_indices.shape[0], 2 * knn_indices.shape[1]], device=knn_indices.device,
                                         dtype=torch.int64)
        double_knn_indices[:, 0::2] = 2 * knn_indices
        double_knn_indices[:, 1::2] = 2 * knn_indices + 1

        # and put computed scores for KNNs
        scores = torch.scatter(scores, 1, double_knn_indices, knn_scores)

    else:
        data = [sub_problem.dist_matrices, sub_problem.node_demands, sub_problem.remaining_capacities[:, 0].unsqueeze(-1),
                sub_problem.remaining_capacities[:, -1].unsqueeze(-1), None, None]
        if sub_problem.problem_name == "dcvrp":
            data.append(sub_problem.remaining_distances[:, -1][:, None, None])

        node_features, edge_features, problem_data = prepare_data(data, sub_problem.problem_name)

        scores = net(node_features, edge_features, problem_data)  # (b, seq)
    return scores


def beam_search_decoding_loop(problem_name: str, problem_data: list,
                              net: Module, beam_size: int, knns: int) -> tuple[Tensor, Tensor]:

    if problem_name == "dcvrp":
        dist_matrices, node_demands, total_capacities, _, _, distance_constraints, _ = problem_data
        remaining_distances = distance_constraints
    else:
        dist_matrices, node_demands, total_capacities, _, _, _ = problem_data
        remaining_distances = None

    orig_distances = copy.deepcopy(dist_matrices)

    bs, num_nodes, _, _ = dist_matrices.shape  # (including repetition of begin=end node)
    device = dist_matrices.device
    original_idxs = torch.tensor(list(range(num_nodes)), device=device)[None, :].repeat(bs, 1)
    paths = torch.zeros((bs * beam_size, num_nodes), dtype=torch.long, device=device)
    via_depots = torch.full((bs * beam_size, num_nodes), False, dtype=torch.bool, device=device)

    probabilities = torch.zeros((bs, 1), device=device)
    tour_lens = torch.zeros(bs * beam_size, device=device)

    sub_problem = VRPSubPb(problem_name, dist_matrices, node_demands, total_capacities,
                           remaining_distances, original_idxs)

    for dec_pos in range(1, num_nodes - 1):
        idx_selected, via_depot, batch_in_prev_input, probabilities, sub_problem =\
            beam_search_decoding_step(sub_problem, net, probabilities, bs, beam_size, knns)

        paths = paths[batch_in_prev_input]
        via_depots = via_depots[batch_in_prev_input]

        paths[:, dec_pos] = idx_selected
        via_depots[:, dec_pos] = via_depot
        tour_lens = tour_lens[batch_in_prev_input]

        # compute lenghts for direct edges
        tour_lens[~via_depots[:, dec_pos]] +=(
            orig_distances[batch_in_prev_input, paths[:, dec_pos], paths[:, dec_pos - 1], 0][~via_depots[:, dec_pos]])

        # compute lenghts for edges via depot
        tour_lens[via_depots[:, dec_pos]] += (
            orig_distances[batch_in_prev_input, paths[:, dec_pos-1], 0, 0][via_depots[:, dec_pos]] +
            orig_distances[batch_in_prev_input, 0, paths[:, dec_pos], 0][via_depots[:, dec_pos]])

        orig_distances = orig_distances[batch_in_prev_input]

    tour_lens += orig_distances[batch_in_prev_input, paths[:, dec_pos-1], 0, 0]

    tour_lens = tour_lens.reshape(bs, -1)
    paths = paths.reshape(bs, -1, num_nodes)
    via_depots = via_depots.reshape(bs, -1, num_nodes)
    min_tour_lens = torch.argmin(tour_lens, dim=1)
    return paths[torch.arange(bs), min_tour_lens], via_depots[torch.arange(bs), min_tour_lens]


def beam_search_decoding_step(sub_problem: VRPSubPb, net: Module, prev_probabilities: Tensor,
                              test_batch_size: int, beam_size: int, knns: int) -> (Tensor, VRPSubPb):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    num_nodes = sub_problem.dist_matrices.shape[1]
    num_instances = sub_problem.dist_matrices.shape[0] // test_batch_size
    candidates = torch.softmax(scores, dim=1)

    # repeat 2*num_nodes -> for each node we have two scores - direct edge and via depot
    probabilities = (prev_probabilities.repeat(1, 2 * num_nodes) + torch.log(candidates)).reshape(test_batch_size, -1)

    k = min(beam_size, probabilities.shape[1] - 2)
    topk_values, topk_indexes = torch.topk(probabilities, k, dim=1)
    batch_in_prev_input = ((num_instances * torch.arange(test_batch_size, device=probabilities.device)).unsqueeze(dim=1) +\
                           torch.div(topk_indexes, 2 * num_nodes, rounding_mode="floor")).flatten()
    topk_values = topk_values.flatten()
    topk_indexes = topk_indexes.flatten()
    sub_problem.original_idxs = sub_problem.original_idxs[batch_in_prev_input]
    sub_problem.node_demands = sub_problem.node_demands[batch_in_prev_input]
    sub_problem.remaining_capacities = sub_problem.remaining_capacities[batch_in_prev_input]
    sub_problem.dist_matrices = sub_problem.dist_matrices[batch_in_prev_input]

    selected_nodes = torch.remainder(topk_indexes, 2 * num_nodes).unsqueeze(dim=1)
    idx_selected = torch.div(selected_nodes, 2, rounding_mode='trunc')
    via_depot = (selected_nodes % 2 == 1)

    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected)
    new_subproblem, via_depot = reformat_subproblem_for_next_step(sub_problem, idx_selected, via_depot)

    return idx_selected_original.squeeze(1), via_depot.squeeze(1), batch_in_prev_input,\
        topk_values.unsqueeze(dim=1), new_subproblem


def reformat_subproblem_for_next_step(sub_problem: VRPSubPb, idx_selected: Tensor, via_depot: Tensor)\
        -> tuple[VRPSubPb, Tensor]:
    # Example: current_subproblem: [a b c d e] => (model selects d) => next_subproblem: [d b c e]

    bs, subpb_size, _, num_features = sub_problem.dist_matrices.shape
    is_selected = (torch.arange(subpb_size, device=sub_problem.dist_matrices.device).unsqueeze(dim=0).repeat(bs, 1) ==
                   idx_selected.repeat(1, subpb_size))

    next_demands = remove_origin_and_reorder_tensor(sub_problem.node_demands, is_selected)
    next_original_idxs = remove_origin_and_reorder_tensor(sub_problem.original_idxs, is_selected)

    selected_demands = sub_problem.node_demands[is_selected].unsqueeze(dim=1)

    # update current capacities
    remaining_capacities = sub_problem.remaining_capacities[:, -1].unsqueeze(-1) - selected_demands

    # recompute capacities
    total_capacities = sub_problem.remaining_capacities[:, 0:1]
    remaining_capacities[via_depot.bool()] = (total_capacities - selected_demands)[via_depot.bool()]

    next_current_capacities = torch.cat([sub_problem.remaining_capacities, remaining_capacities], dim=-1)

    if sub_problem.problem_name == "dcvrp":
        remaining_distances = (sub_problem.remaining_distances[:, -1] -
                               sub_problem.dist_matrices[:, 0, :, 0][is_selected])[:, None]

        remaining_distances[via_depot.bool()] = (sub_problem.remaining_distances[:, 0:1][via_depot.bool()] -
                                                 sub_problem.dist_matrices[:, -1, :, 0][is_selected][:, None][via_depot.bool()])
        next_remaining_distances = torch.cat([sub_problem.remaining_distances, remaining_distances], dim=-1)

    else:
        next_remaining_distances = None

    next_dist_matrices = remove_origin_and_reorder_matrix(sub_problem.dist_matrices, is_selected)

    new_subproblem = VRPSubPb(sub_problem.problem_name, next_dist_matrices, next_demands, next_current_capacities,
                              next_remaining_distances, next_original_idxs)

    return new_subproblem, via_depot