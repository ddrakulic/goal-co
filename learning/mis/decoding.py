from dataclasses import dataclass, asdict
import torch
import numpy as np
from torch import Tensor
from torch.nn import Module
from utils.data_manipulation import prepare_graph_data


@dataclass
class MISSubPb:
    """
    In decoding, we successively apply model on progressively smaller sub-problems.
    In each sub-problem, we keep track of the indices of each node in the original full-problem.
    """
    problem_name: str
    adj_matrices: Tensor
    can_be_selected: Tensor
    original_idxs: Tensor

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


def decode(problem_name: str, data: list, net: Module, beam_size: int = 1, knns: int = -1,
           sample: bool = False) -> tuple[Tensor, Tensor]:
    if beam_size == 1:
        num_selected_nodes, selected_items = greedy_decoding(data, net, knns, sample)
    else:
        raise NotImplementedError
    return num_selected_nodes, selected_items.cpu().numpy()


def greedy_decoding(data: list, net: Module, knns: int, sample: bool):
    adj_matrices, can_be_selected, _, _ = data
    bs, problem_size, _ = adj_matrices.shape
    original_idxs = torch.tensor(list(range(problem_size)), device=adj_matrices.device)[None, :].repeat(bs, 1)
    sub_problem = MISSubPb("mis", adj_matrices, can_be_selected, original_idxs)
    selected_nodes = torch.full((bs, problem_size), -1, dtype=torch.long, device=adj_matrices.device)
    for dec_pos in range(problem_size):
        idx_selected, sub_problem = greedy_decoding_step(sub_problem, net, knns, sample)
        if torch.count_nonzero(idx_selected.flatten() == -1) == bs:
            # all tours are done!
            break
        selected_nodes[:, dec_pos] = idx_selected

    num_selected = torch.count_nonzero(selected_nodes != -1, dim=-1).float()
    # todo: implement checking is there is no remaining edges
    # assert sub_problem.adj_matrices.sum() == 0

    return num_selected, selected_nodes


def greedy_decoding_step(sub_problem: MISSubPb, net: Module, knns: Tensor, sample: bool) -> (Tensor, MISSubPb):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    if sample:
        probs = torch.softmax(scores, dim=-1)
        # if instance is finished, all scores will be infinity, so sampling is not possible,
        # hack, provide flat distribution, anyway at this point result is ignored
        probs[probs.sum(dim=-1).isnan()] = torch.softmax(torch.ones(scores.shape[1], device=scores.device), dim=0)
        selected_nodes = torch.tensor([np.random.choice(np.arange(probs.shape[1]),
                                                        p=prob.cpu().numpy()) for prob in probs]).to(probs.device)[:, None]
    else:
        selected_nodes = torch.argmax(scores, dim=1, keepdim=True)
    # if all possible choices are masked (= -inf), then instance is solved
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, selected_nodes)
    idx_selected_original[scores.max(dim=-1)[0] == -np.inf] = -1
    return idx_selected_original.squeeze(1), reformat_subproblem_for_next_step(sub_problem, selected_nodes)


def prepare_input_and_forward_pass(sub_problem: MISSubPb, net: Module, knns: int) -> Tensor:
    bs, num_nodes, _ = sub_problem.adj_matrices.shape
    data = [sub_problem.adj_matrices, sub_problem.can_be_selected, None, None]
    node_features, adj_matrix, problem_data = prepare_graph_data(data, "mis")
    scores = net(node_features, adj_matrix, problem_data)
    return scores


def reformat_subproblem_for_next_step(sub_problem: MISSubPb, idx_selected: Tensor) -> MISSubPb:
    # Example: current_subproblem: [a b c d e] => (model selects d) => next_subproblem: [d b c e]
    bs, subpb_size, _ = sub_problem.adj_matrices.shape
    is_selected = torch.arange(
        subpb_size,
        device=sub_problem.adj_matrices.device).unsqueeze(dim=0).repeat(bs, 1) == idx_selected.repeat(1, subpb_size)

    can_be_selected = sub_problem.can_be_selected
    can_be_selected[sub_problem.adj_matrices[torch.arange(bs), idx_selected.squeeze(-1)] == 1.] = False

    next_adj_matrices = sub_problem.adj_matrices[~is_selected].reshape((bs, -1, subpb_size))
    next_adj_matrices = next_adj_matrices.transpose(1, 2)[~is_selected].reshape((bs, -1, subpb_size-1))
    next_can_be_selected = can_be_selected[~is_selected].reshape((bs, -1))
    next_original_idxs = sub_problem.original_idxs[~is_selected].reshape((bs, -1))
    return MISSubPb("mis", next_adj_matrices, next_can_be_selected, next_original_idxs)
