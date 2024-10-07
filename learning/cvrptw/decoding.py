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
from learning.reformat_subproblems import remove_origin_and_reorder_matrix
from utils.data_manipulation import prepare_routing_data


@dataclass
class CVRPTWSubPb:
    travel_times: Tensor
    node_demands: Tensor
    service_times: Tensor
    time_windows: Tensor
    departure_times: Tensor
    remaining_capacities: Tensor
    original_idxs: Tensor
    total_capacities: Tensor


def reconstruct_tours(paths: Tensor, via_depots: Tensor):
    bs = paths.shape[0]
    complete_paths = [[0] for _ in range(bs)]
    for pos in range(1, paths.shape[1]):
        nodes_to_add = paths[:, pos].tolist()
        for instance in (via_depots[:, pos]).nonzero().squeeze(-1).cpu().numpy():
            complete_paths[instance].append(0)
        for instance in range(bs):
            complete_paths[instance].append(nodes_to_add[instance])

    return complete_paths


def decode(problem_name: str, data: list, net: Module, beam_size: int, knns: int,
           sample=False, make_tours: bool = False) -> tuple[Tensor, Tensor]:
    travel_times, node_demands, total_capacities, _, _, service_times, time_windows, _, _ = data
    if beam_size == 1:
        paths, via_depots, tour_lengths = greedy_decoding_loop(travel_times, node_demands, service_times, time_windows,
                                                               total_capacities, net, knns)
    else:
        raise NotImplementedError
    if make_tours:
        tours = reconstruct_tours(paths, via_depots)
    else:
        tours = None

    return tour_lengths, tours


def greedy_decoding_loop(travel_times: Tensor, node_demands: Tensor, service_times: Tensor, time_windows: Tensor,
                         total_capacities: Tensor, net: Module, knns: int) -> tuple[Tensor, Tensor, Tensor]:
    bs, num_nodes, _, _ = travel_times.shape
    original_idxs = torch.tensor(list(range(num_nodes)), device=travel_times.device)[None, :].repeat(bs, 1)
    departure_times = torch.zeros(bs, dtype=torch.int32, device=travel_times.device)
    via_depots = torch.full((bs, num_nodes), False, dtype=torch.bool, device=travel_times.device)
    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=travel_times.device)
    paths[:, -1] = num_nodes - 1

    lenghts = torch.zeros(bs, device=travel_times.device)

    sub_problem = CVRPTWSubPb(travel_times, node_demands, service_times, time_windows, departure_times.unsqueeze(-1),
                              total_capacities, original_idxs, total_capacities)

    for dec_pos in range(1, num_nodes - 1):
        idx_selected, via_depot, sub_problem = greedy_decoding_step(sub_problem, net, knns)
        paths[:, dec_pos] = idx_selected
        via_depots[:, dec_pos] = via_depot

        # compute lenghts for direct edges
        direct_travel_times = travel_times[~via_depots[:, dec_pos], paths[~via_depots[:, dec_pos], dec_pos-1],
            paths[~via_depots[:, dec_pos], dec_pos], 0]
        # compute lenghts for edges via depot
        to_depots_travel_times = travel_times[via_depots[:, dec_pos], paths[via_depots[:, dec_pos], dec_pos-1],
            paths[via_depots[:, dec_pos], 0], 0]
        from_depot_travel_times = travel_times[via_depots[:, dec_pos], paths[via_depots[:, dec_pos], 0],
            paths[via_depots[:, dec_pos], dec_pos], 0]
        lenghts[~via_depots[:, dec_pos]] += direct_travel_times
        lenghts[via_depots[:, dec_pos]] += to_depots_travel_times + from_depot_travel_times

        arrival_times = torch.zeros(bs, dtype=torch.int32, device=travel_times.device)
        arrival_times[~via_depot] =\
            torch.max(departure_times[~via_depot] + direct_travel_times,
                      time_windows[torch.arange(bs), idx_selected, 0][~via_depot])
        arrival_times[via_depot] = \
            torch.max(from_depot_travel_times, time_windows[torch.arange(bs), idx_selected, 0][via_depot])

        departure_times = arrival_times + service_times[torch.arange(bs), idx_selected]
        # check no1: are departure times well computed
        assert torch.all(sub_problem.departure_times[..., -1] == departure_times)
        # check no2: is the time window constraint is satisfied
        assert torch.all(arrival_times <= time_windows[torch.arange(bs), idx_selected, 1])

    lenghts += travel_times[torch.arange(bs), paths[:, -2], paths[:, -1], 0]
    # check no3: is capacity constraint is satisfied
    assert sub_problem.remaining_capacities.min() >= 0
    # check no4: are all nodes visited
    assert paths.sum(dim=1).sum() == paths.shape[0] * .5 * (num_nodes - 1) * num_nodes

    return paths, via_depots, lenghts


def prepare_input_and_forward_pass(sub_problem: CVRPTWSubPb, net: Module, knns: int) -> Tensor:
    # find K nearest neighbors of the current node
    bs, num_nodes, node_dim, num_features = sub_problem.travel_times.shape

    if 0 < knns < num_nodes:

        knn_indices = torch.topk(sub_problem.travel_times[:, :-1, 0, 0], k=knns - 1, dim=-1, largest=False).indices
        # and add it manually

        knn_indices = torch.cat([knn_indices, torch.full([bs, 1], num_nodes - 1, device=knn_indices.device)], dim=-1)

        knn_node_demands = torch.gather(sub_problem.node_demands, 1, knn_indices)
        knn_service_times = torch.gather(sub_problem.service_times, 1, knn_indices)
        knn_time_windows = torch.gather(sub_problem.time_windows, 1, knn_indices.unsqueeze(-1).repeat(1, 1, 2))
        knn_travel_times = torch.gather(sub_problem.travel_times, 1,
                                        knn_indices[..., None, None].repeat(1, 1, num_nodes, num_features))
        knn_travel_times = torch.gather(knn_travel_times, 2,
                                        knn_indices[:, None, :, None].repeat(1, knns, 1, num_features))

        # scale knn_dist_matrices to have the same distribution as in the training data
        knn_dist_matrices = (knn_travel_times /
                             knn_travel_times.reshape(bs, -1).amax(dim=-1)[:, None, None, None].repeat(1, knns, knns, num_features))

        data = [knn_travel_times, knn_node_demands, sub_problem.total_capacities,
                sub_problem.remaining_capacities[:, -1][..., None], None, knn_service_times,
                knn_time_windows, sub_problem.departure_times[:, -1][..., None], None]

        knn_node_features, knn_travel_times, problem_data = prepare_routing_data(data, "cvrptw")

        knn_scores = net(knn_node_features, knn_dist_matrices, problem_data)  # (b, seq)

        # create result tensor for scores with all -inf elements
        scores = torch.full((bs, 2 * num_nodes), -np.inf, device=knn_node_features.device)
        double_knn_indices = torch.zeros([knn_indices.shape[0], 2 * knn_indices.shape[1]], device=knn_indices.device,
                                         dtype=torch.int64)
        double_knn_indices[:, 0::2] = 2 * knn_indices
        double_knn_indices[:, 1::2] = 2 * knn_indices + 1

        # and put computed scores for KNNs
        scores = torch.scatter(scores, 1, double_knn_indices, knn_scores)

    else:
        data = [sub_problem.travel_times, sub_problem.node_demands, sub_problem.total_capacities,
                sub_problem.remaining_capacities[:, -1].unsqueeze(-1), None, sub_problem.service_times,
                sub_problem.time_windows, sub_problem.departure_times[:, -1].unsqueeze(-1), None]
        node_features, travel_times, problem_data = prepare_routing_data(data, "cvrptw")
        scores = net(node_features, travel_times, problem_data)
    if sub_problem.node_demands[:, 0].sum() == 0:
        # if we are at the beginning, mask via depot edges
        scores[:, 1::2] = -np.inf
    return scores

def greedy_decoding_step(sub_problem: CVRPTWSubPb, net: Module, knns: int) -> (Tensor, CVRPTWSubPb):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    selected_nodes = torch.argmax(scores, dim=1, keepdim=True)
    idx_selected = torch.div(selected_nodes, 2, rounding_mode='trunc')
    via_depot = (selected_nodes % 2 == 1)
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected)

    new_subproblem, via_depot = reformat_subproblem_for_next_step(sub_problem, idx_selected, via_depot)
    return idx_selected_original.squeeze(1), via_depot.squeeze(1), new_subproblem


def reformat_subproblem_for_next_step(sub_problem: CVRPTWSubPb, idx_selected: Tensor, via_depot: Tensor) -> CVRPTWSubPb:
    # Example: current_subproblem: [a b c d e] => (model selects d) => next_subproblem: [d b c e]
    bs, subpb_size, _, num_features = sub_problem.travel_times.shape

    is_selected = torch.arange(subpb_size, device=sub_problem.travel_times.device).unsqueeze(dim=0).repeat(bs, 1) ==\
                  idx_selected.repeat(1, subpb_size)

    # next begin node = just-selected node
    selected_demands = sub_problem.node_demands[is_selected].unsqueeze(dim=1)
    selected_time_windows = sub_problem.time_windows[is_selected].unsqueeze(dim=1)
    selected_service_times = sub_problem.service_times[is_selected].unsqueeze(dim=1)
    selected_original_idx = sub_problem.original_idxs[is_selected].unsqueeze(dim=1)

    # remaining nodes = the rest, minus current first node
    remaining_demands = sub_problem.node_demands[~is_selected].reshape((bs, -1))[:, 1:]
    remaining_time_windows = sub_problem.time_windows[~is_selected].reshape((bs, -1, 2))[:, 1:, :]
    remaining_service_times = sub_problem.service_times[~is_selected].reshape((bs, -1))[:, 1:]
    remaining_original_idxs = sub_problem.original_idxs[~is_selected].reshape((bs, -1))[:, 1:]

    # concatenate
    next_demands = torch.cat([selected_demands, remaining_demands], dim=1)
    next_time_windows = torch.cat([selected_time_windows, remaining_time_windows], dim=1)
    next_service_times = torch.cat([selected_service_times, remaining_service_times], dim=1)
    next_original_idxs = torch.cat([selected_original_idx, remaining_original_idxs], dim=1)

    # update current capacities
    remaining_capacities = sub_problem.remaining_capacities[:, -1].unsqueeze(dim=1) - selected_demands

    # if we reach max capacity -> add edge in via depot
    via_depot[remaining_capacities < 0] = True

    # update departure times
    next_departure_times = torch.max(
        sub_problem.departure_times[..., -1].unsqueeze(-1) + sub_problem.travel_times[:, 0, :, 0][is_selected].unsqueeze(-1),
        selected_time_windows[..., 0]) + selected_service_times

    # for via_depot edges, departure time
    next_departure_times[via_depot] = (torch.max(sub_problem.travel_times[:, -1, :, 0][is_selected].unsqueeze(-1),
                                                 selected_time_windows[..., 0]) + selected_service_times)[via_depot]

    next_departure_times = torch.cat([sub_problem.departure_times, next_departure_times], dim=-1)

    # recompute capacities
    remaining_capacities[via_depot] = (sub_problem.total_capacities - selected_demands)[via_depot]

    next_remaining_capacities = torch.cat([sub_problem.remaining_capacities, remaining_capacities], dim=-1)

    next_travel_times = remove_origin_and_reorder_matrix(sub_problem.travel_times, is_selected)

    new_subproblem = CVRPTWSubPb(next_travel_times, next_demands, next_service_times, next_time_windows,
                                 next_departure_times, next_remaining_capacities, next_original_idxs,
                                 sub_problem.total_capacities)

    return new_subproblem, via_depot
