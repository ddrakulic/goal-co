"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import numpy as np
from dataclasses import dataclass, asdict
import torch
from torch import Tensor
from torch.nn import Module
from utils.data_manipulation import prepare_graph_data


@dataclass
class JSSPSubProblem:
    """
    In decoding, we successively apply model on progressively smaller sub-problems.
    In each sub-problem, we keep track of the indices of each node in the original full-problem.
    """
    problem_name: str
    job_idxs: Tensor
    ranks: Tensor
    execution_times: Tensor
    task_on_machines: Tensor
    precedences: Tensor
    jobs_tasks: Tensor
    task_availability_times: Tensor
    machine_availability_times: Tensor
    scales: Tensor

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


def decode(problem_name: str, data: list, net: Module, beam_size: int = 1, knns: int = -1,
           sample: bool = False) -> tuple[Tensor, Tensor]:

    if beam_size == 1:
        makespans, solutions = greedy_decoding(problem_name, data, net, knns, sample)
    else:
        raise NotImplementedError

    return makespans, solutions.cpu().numpy()


def greedy_decoding(problem_name: str, data: list, net: Module,
                    knns: int = -1, sample: bool = False) -> tuple[Tensor, Tensor]:
    (execution_times, task_on_machines, precedences, jobs_tasks, job_availability_times, machine_availability_times,
        scales, _) = data
    bs, num_tasks, num_machines = task_on_machines.shape
    num_jobs = num_tasks // num_machines

    # solution: for each machine, N tasks with (job_id, rank, start_time, end_time)
    solutions = torch.full((bs, num_machines, num_jobs, 4), -1., device=execution_times.device)
    num_solutions_on_machines = torch.zeros(bs, num_machines, device=execution_times.device)
    job_idxs = torch.arange(num_jobs, device=execution_times.device)
    job_idxs = job_idxs[:, None].repeat(1, num_machines).flatten()[None, :].repeat(bs, 1)
    ranks = torch.arange(num_machines, device=execution_times.device)
    ranks = ranks[None, :].repeat(num_jobs, 1).flatten().repeat(bs, 1)

    sub_problem = JSSPSubProblem(problem_name, job_idxs, ranks, execution_times, task_on_machines, precedences,
                                 jobs_tasks, job_availability_times, machine_availability_times, scales)

    for dec_pos in range(num_tasks):
        (selected_job_idxs, selected_ranks, selected_machine_idx, start_times,
         end_times, sub_problem) = greedy_decoding_step(sub_problem, net, knns, sample)

        result = torch.stack([selected_job_idxs, selected_ranks, start_times, end_times]).t()
        solutions[torch.arange(bs), selected_machine_idx.int(),
            num_solutions_on_machines[torch.arange(bs), selected_machine_idx].int()] = result
        num_solutions_on_machines[torch.arange(bs), selected_machine_idx] += 1

    makespans = solutions.reshape(bs, -1, 4)[..., -1].max(dim=-1)[0]

    return makespans, solutions


def greedy_decoding_step(sub_problem: JSSPSubProblem, net: Module, knns: int = -1,
                         sample: bool = False) -> (Tensor, JSSPSubProblem):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    if sample:
        probs = torch.softmax(scores, dim=-1)
        selected_nodes = torch.tensor([np.random.choice(np.arange(probs.shape[1]),
                                                        p=prob.cpu().numpy()) for prob in probs]).to(probs.device)[:, None]
    else:
        selected_nodes = torch.argmax(scores, dim=1, keepdim=True)

    (selected_job_idxs, selected_ranks, selected_machine_idxs, start_times,
     end_times, new_subproblem) = reformat_subproblem_for_next_step(sub_problem, selected_nodes)

    return selected_job_idxs, selected_ranks, selected_machine_idxs, start_times, end_times, new_subproblem


def prepare_input_and_forward_pass(sub_problem: JSSPSubProblem, net: Module, knns: int = -1) -> Tensor:
    bs, num_tasks = sub_problem.execution_times.shape
    if 0 < knns < num_tasks:
        num_machines = sub_problem.machine_availability_times.shape[-1]
        sorted_nodes_idx = torch.sort(sub_problem.ranks, dim=-1).indices
        indices_smallest_ranks = sorted_nodes_idx[:, :knns]

        execution_times = torch.gather(sub_problem.execution_times, 1, indices_smallest_ranks)
        task_availability_times = torch.gather(sub_problem.task_availability_times, 1, indices_smallest_ranks)

        precedences = torch.gather(sub_problem.precedences, 1,
                                   indices_smallest_ranks.unsqueeze(dim=-1).repeat(1, 1, num_tasks))
        precedences = torch.gather(precedences, 2,
                                   indices_smallest_ranks.unsqueeze(dim=-2).repeat(1, knns, 1))

        jobs_tasks = torch.gather(sub_problem.jobs_tasks, 1,
                                  indices_smallest_ranks.unsqueeze(dim=-1).repeat(1, 1, num_tasks))
        jobs_tasks = torch.gather(jobs_tasks, 2,
                                  indices_smallest_ranks.unsqueeze(dim=-2).repeat(1, knns, 1))

        task_on_machines = torch.gather(sub_problem.task_on_machines, 1,
                                        indices_smallest_ranks.unsqueeze(dim=-1).repeat(1, 1, num_machines))

        data = [execution_times, task_on_machines, precedences, jobs_tasks, task_availability_times,
                sub_problem.machine_availability_times, sub_problem.scales, None]
        node_features, edge_features, problem_data = prepare_graph_data(data, sub_problem.problem_name)
        knn_scores = net(node_features, edge_features, problem_data)
        scores = torch.full((bs, num_tasks), -np.inf, device=knn_scores.device)
        # and put computed scores for KNNs
        scores = torch.scatter(scores, 1, indices_smallest_ranks, knn_scores)
    else:
        data = [sub_problem.execution_times, sub_problem.task_on_machines, sub_problem.precedences,
                sub_problem.jobs_tasks, sub_problem.task_availability_times, sub_problem.machine_availability_times,
                sub_problem.scales, None]
        node_features, edge_features, problem_data = prepare_graph_data(data, sub_problem.problem_name)
        scores = net(node_features, edge_features, problem_data)
    return scores


def reformat_subproblem_for_next_step(sub_problem: JSSPSubProblem, selected_task_idx: Tensor) -> (
        Tensor, Tensor, Tensor, Tensor, Tensor, JSSPSubProblem):
    # remove first job and update machine states (add execution time of selected jobs)
    bs, num_tasks, num_machines = sub_problem.task_on_machines.shape

    is_selected = (torch.arange(num_tasks, device=sub_problem.execution_times.device)[None, ...].repeat(bs, 1) ==
                   selected_task_idx.repeat(1, num_tasks))

    selected_job_idxs = sub_problem.job_idxs[is_selected]
    selected_ranks = sub_problem.ranks[is_selected]

    selected_tasks_execution_times = sub_problem.execution_times[is_selected]
    selected_machine_idxs = torch.nonzero(sub_problem.task_on_machines[is_selected])[..., 1]

    # remove data of selected task
    new_execution_times = sub_problem.execution_times[~is_selected].reshape(bs, num_tasks-1)
    new_job_idxs = sub_problem.job_idxs[~is_selected].reshape(bs, num_tasks-1)
    new_ranks = sub_problem.ranks[~is_selected].reshape(bs, num_tasks - 1)
    new_task_on_machines = sub_problem.task_on_machines[~is_selected].reshape(bs, num_tasks-1, num_machines)

    # remove rows and columns of selected tasks from precedences matrices
    new_precedences = sub_problem.precedences[~is_selected].reshape(bs, num_tasks-1, num_tasks)
    new_precedences = new_precedences.transpose(1, 2)[~is_selected].reshape(bs, num_tasks-1, num_tasks-1)
    new_precedences = new_precedences.transpose(1, 2)

    new_jobs_tasks = sub_problem.jobs_tasks[~is_selected].reshape(bs, num_tasks - 1, num_tasks)
    new_jobs_tasks = new_jobs_tasks.transpose(1, 2)[~is_selected].reshape(bs, num_tasks - 1, num_tasks - 1)
    new_jobs_tasks = new_jobs_tasks.transpose(1, 2)

    # start time max(task_availability, machine_availability)
    task_availability_times = sub_problem.task_availability_times[is_selected]
    machine_availability_times = sub_problem.machine_availability_times[torch.arange(bs), selected_machine_idxs]

    task_start_times = torch.max(task_availability_times, machine_availability_times)
    task_end_times = task_start_times + selected_tasks_execution_times
    # update task and machine availability times
    new_machine_availability_times = sub_problem.machine_availability_times
    new_machine_availability_times[torch.arange(bs), selected_machine_idxs] = task_end_times

    new_task_availability_times = sub_problem.task_availability_times[~is_selected].reshape(bs, num_tasks-1)
    # todo: reimplement with vectorization
    for b in range(bs):
        new_task_availability_times[b][new_job_idxs[b] == selected_job_idxs[b]] = task_end_times[b]

    return (selected_job_idxs, selected_ranks, selected_machine_idxs, task_start_times, task_end_times,
            JSSPSubProblem(sub_problem.problem_name, new_job_idxs, new_ranks, new_execution_times, new_task_on_machines,
                           new_precedences, new_jobs_tasks, new_task_availability_times, new_machine_availability_times,
                           sub_problem.scales))

