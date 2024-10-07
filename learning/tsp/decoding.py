"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from dataclasses import dataclass, asdict
import copy
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from learning.reformat_subproblems import remove_origin_and_reorder_matrix, remove_origin_and_reorder_tensor
from utils.data_manipulation import prepare_data
from utils.misc import compute_tour_lens


@dataclass
class TSPSubProblem:
    problem_name: str
    dist_matrices: Tensor
    original_idxs: Tensor

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


def decode(problem_name: str, problem_data: list, net: Module, beam_size: int = 1, knns: int = -1,
           sample: bool = False) -> tuple[Tensor, Tensor]:
    dist_matrices = problem_data[0]
    if len(dist_matrices.shape) == 3:
        dist_matrices = dist_matrices.unsqueeze(-1)

    if beam_size == 1:
        tours = decoding_loop(problem_name, dist_matrices, net, knns, sample)
    else:
        tours = beam_search_decoding_loop(problem_name, dist_matrices, net, beam_size, knns)

    num_nodes = dist_matrices.shape[1]

    assert tours.sum(dim=1).sum() == tours.shape[0] * .5 * (num_nodes - 1) * num_nodes

    # compute distances by using (original) distance matrix
    if problem_name == "tsp":
        objective_values = compute_tour_lens(tours, dist_matrices[..., 0])
    elif problem_name == "sop":
        objective_values = compute_tour_lens(tours[..., :-1], dist_matrices[..., 0])
    elif problem_name == "trp":
        batch_idx = torch.arange(len(tours))
        dist_matrices = dist_matrices[..., 0]
        traveling_distances = torch.stack([dist_matrices[batch_idx, tours[batch_idx, idx], tours[batch_idx, idx + 1]]
                                           for idx in range(tours.shape[1] - 1)]).transpose(0, 1)
        objective_values = traveling_distances[:, :-1].cumsum(dim=-1).sum(dim=-1)
    else:
        raise NotImplementedError
    return objective_values, tours.cpu().numpy()


def decoding_loop(problem_name: str, dist_matrices: Tensor, net: Module, knns: int, sample: bool) -> Tensor:
    bs, num_nodes, _, _ = dist_matrices.shape

    original_idxs = torch.tensor(list(range(num_nodes)), device=dist_matrices.device)[None, :].repeat(bs, 1)
    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=dist_matrices.device)
    paths[:, -1] = num_nodes - 1
    sub_problem = TSPSubProblem(problem_name, dist_matrices, original_idxs)
    for dec_pos in range(1, num_nodes - 1):
        idx_selected, sub_problem = decoding_step(sub_problem, net, knns, sample)
        paths[:, dec_pos] = idx_selected
    return paths


def prepare_input_and_forward_pass(sub_problem: TSPSubProblem, net: Module, knns: int) -> Tensor:
    # find K nearest neighbors of the current node
    bs, num_nodes, _, num_edge_features = sub_problem.dist_matrices.shape

    if 0 < knns < num_nodes:
        # sort node by distance from the origin (ignore the target node)
        dist_matrices = sub_problem.dist_matrices
        _, sorted_nodes_idx = torch.sort(dist_matrices[:, 0, :-1, 0], dim=-1)

        # select KNNs
        knn_indices = sorted_nodes_idx[:, :knns-1]
        # and add the target at the end
        knn_indices = torch.cat([knn_indices,
                                 torch.full([bs, 1], num_nodes - 1, device=knn_indices.device)], dim=-1)

        knn_dist_matrices = torch.gather(dist_matrices, 1, knn_indices[..., None, None].repeat(1, 1, num_nodes, 2))
        knn_dist_matrices = torch.gather(knn_dist_matrices, 2, knn_indices[:, None, :, None].repeat(1, knns, 1, 2))

        knn_dist_matrices = (knn_dist_matrices / knn_dist_matrices.amax(dim=-1).amax(dim=-1).amax(dim=-1)[:, None, None, None].repeat(1, knns, knns, 2))

        node_features, edge_features, problem_data = prepare_data([knn_dist_matrices, None], sub_problem.problem_name)

        knn_scores = net(node_features, edge_features, problem_data)
        # create result tensor for scores with all -inf elements
        scores = torch.full((bs, num_nodes), -np.inf, device=knn_scores.device)
        # and put computed scores for KNNs
        scores = torch.scatter(scores, 1, knn_indices, knn_scores)
    else:
        node_features, edge_features, problem_data = prepare_data([sub_problem.dist_matrices, None],
                                                                  sub_problem.problem_name)
        scores = net(node_features, edge_features, problem_data)

    return scores

def decoding_step(sub_problem: TSPSubProblem, net: Module, knns: int, sample: bool) -> (Tensor, TSPSubProblem):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    if sample:
        probs = torch.softmax(scores, dim=-1)
        selected_nodes = torch.tensor([np.random.choice(np.arange(probs.shape[1]),
                                                        p=prob.cpu().numpy()) for prob in probs]).to(probs.device)[:, None]
    else:
        selected_nodes = torch.argmax(scores, dim=1, keepdim=True)  # (b, 1)
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, selected_nodes)
    return idx_selected_original.squeeze(1), reformat_subproblem_for_next_step(sub_problem, selected_nodes)

def beam_search_decoding_loop(problem_name: str, dist_matrices: Tensor, net: Module, beam_size: int, knns: int) -> Tensor:
    bs, num_nodes, _, _ = dist_matrices.shape  # (including repetition of begin=end node)

    orig_distances = copy.deepcopy(dist_matrices)

    original_idxs = torch.tensor(list(range(num_nodes)), device=dist_matrices.device)[None, :].repeat(bs, 1)
    paths = torch.zeros((bs * beam_size, num_nodes), dtype=torch.long, device=dist_matrices.device)
    paths[:, -1] = num_nodes - 1

    probabilities = torch.zeros((bs, 1), device=dist_matrices.device)
    tour_lens = torch.zeros(bs * beam_size, 1, device=dist_matrices.device)

    sub_problem = TSPSubProblem(problem_name, dist_matrices, original_idxs)
    for dec_pos in range(1, num_nodes - 1):

        idx_selected_original, batch_in_prev_input, probabilities, sub_problem =\
            beam_search_decoding_step(sub_problem, net, probabilities, bs, beam_size, knns)

        paths = paths[batch_in_prev_input]
        paths[:, dec_pos] = idx_selected_original
        tour_lens = tour_lens[batch_in_prev_input]

        # these are distances between normalized! coordinates (!= real tour lengths)
        tour_lens += orig_distances[batch_in_prev_input, paths[:, dec_pos-1], paths[:, dec_pos], 0].unsqueeze(-1)
        orig_distances = orig_distances[batch_in_prev_input]

    tour_lens += orig_distances[batch_in_prev_input, paths[:, dec_pos-1], 0, 0].unsqueeze(-1)

    distances = tour_lens.reshape(bs, -1)
    paths = paths.reshape(bs, -1, num_nodes)
    return paths[torch.arange(bs), torch.argmin(distances, dim=1)]

def beam_search_decoding_step(sub_problem: TSPSubProblem, net: Module, prev_probabilities: Tensor, test_batch_size: int,
                              beam_size: int, knns: int) -> (Tensor, TSPSubProblem):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    num_nodes = sub_problem.dist_matrices.shape[1]
    num_instances = sub_problem.dist_matrices.shape[0] // test_batch_size
    candidates = torch.softmax(scores, dim=1)

    probabilities = (prev_probabilities.repeat(1, num_nodes) + torch.log(candidates)).reshape(test_batch_size, -1)

    k = min(beam_size, probabilities.shape[1] - 2)
    topk_values, topk_indexes = torch.topk(probabilities, k, dim=1)
    batch_in_prev_input = ((num_instances * torch.arange(test_batch_size, device=probabilities.device)).unsqueeze(dim=1) +\
                           torch.div(topk_indexes, num_nodes, rounding_mode="floor")).flatten()
    topk_values = topk_values.flatten()
    topk_indexes = topk_indexes.flatten()
    sub_problem.original_idxs = sub_problem.original_idxs[batch_in_prev_input]
    sub_problem.dist_matrices = sub_problem.dist_matrices[batch_in_prev_input]
    idx_selected = torch.remainder(topk_indexes, num_nodes).unsqueeze(dim=1)
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected).squeeze(-1)

    return idx_selected_original, batch_in_prev_input, topk_values.unsqueeze(dim=1), \
           reformat_subproblem_for_next_step(sub_problem, idx_selected)

def reformat_subproblem_for_next_step(sub_problem: TSPSubProblem, idx_selected: Tensor) -> TSPSubProblem:
    # Example: current_subproblem: [a b c d e] => (model selects d) => next_subproblem: [d b c e]
    bs, subpb_size, _, _ = sub_problem.dist_matrices.shape
    is_selected = (torch.arange(subpb_size, device=sub_problem.dist_matrices.device)[None, ...].repeat(bs, 1) ==
                   idx_selected.repeat(1, subpb_size))

    next_original_idxs = remove_origin_and_reorder_tensor(sub_problem.original_idxs, is_selected)
    next_dist_matrices = remove_origin_and_reorder_matrix(sub_problem.dist_matrices, is_selected)
    return TSPSubProblem(sub_problem.problem_name, next_dist_matrices, next_original_idxs)
