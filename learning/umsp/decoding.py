"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from dataclasses import dataclass, asdict
import torch
from torch import Tensor
from torch.nn import Module
from utils.data_manipulation import prepare_graph_data


@dataclass
class UPMSSubProblem:
    problem_name: str
    execution_times: Tensor
    machine_states: Tensor
    scales: Tensor

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


def decode(problem_name: str, problem_data: list, net: Module, beam_size: int = 1,
           knns: int = -1, sample: bool = False) -> tuple[Tensor, Tensor]:
    execution_times, _, _, scales, _ = problem_data
    if beam_size == 1:
        makespans, solutions = greedy_decoding(execution_times, scales, net, knns)
    else:
        raise NotImplementedError
    return makespans, solutions


def greedy_decoding(execution_times: Tensor, scales: Tensor, net: Module, knns: int = -1) -> tuple[Tensor, Tensor]:
    bs, num_jobs, num_machines = execution_times.shape
    solutions = torch.zeros((bs, num_jobs, num_machines), dtype=torch.int32, device=execution_times.device)
    machine_states = torch.zeros((bs, num_machines), device=execution_times.device)
    sub_problem = UPMSSubProblem("umsp", execution_times, machine_states, scales)
    for dec_pos in range(num_jobs - 1):
        selected_machine_idx, sub_problem = greedy_decoding_step(sub_problem, net, knns)
        solutions[:, dec_pos][torch.arange(bs), selected_machine_idx] = 1

    last_choice = (sub_problem.machine_states + sub_problem.execution_times.squeeze(1)).argmin(dim=-1)
    solutions[:, -1][torch.arange(bs), last_choice] = 1
    # check if all jobs are allocated
    assert (solutions.sum(dim=-1) == 1).all()
    makespans = (execution_times * solutions).sum(-2).max(axis=-1)[0]
    return makespans, solutions


def greedy_decoding_step(sub_problem: UPMSSubProblem, net: Module, knns: int = -1) -> (Tensor, UPMSSubProblem):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    selected_machine_idx = torch.argmax(scores, dim=1, keepdim=True)
    return selected_machine_idx.squeeze(1), reformat_subproblem_for_next_step(sub_problem, selected_machine_idx)


def prepare_input_and_forward_pass(sub_problem: UPMSSubProblem, net: Module, knns: int = -1) -> Tensor:
    bs, num_jobs, _ = sub_problem.execution_times.shape
    if 0 < knns < num_jobs:
        raise "Not implemented"
    else:
        data = [sub_problem.execution_times, sub_problem.machine_states, None, sub_problem.scales, None]
        node_features, edge_features, problem_data = prepare_graph_data(data, "upms")
        scores = net(node_features, edge_features, problem_data)
    return scores


def reformat_subproblem_for_next_step(sub_problem: UPMSSubProblem, selected_machine_idx: Tensor) -> UPMSSubProblem:
    # remove first job and update machine states (add execution time of selected jobs)
    bs = sub_problem.execution_times.shape[0]

    processing_times = sub_problem.execution_times
    selected_processing_times = processing_times[:, 0][torch.arange(bs), selected_machine_idx.squeeze(-1)]
    # update machine states
    sub_problem.machine_states[torch.arange(bs), selected_machine_idx.squeeze(-1)] += selected_processing_times

    new_execution_times = sub_problem.execution_times[:, 1:]

    return UPMSSubProblem("umsp", new_execution_times, sub_problem.machine_states, sub_problem.scales)

