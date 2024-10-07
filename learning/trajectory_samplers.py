"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from learning import decoding_fn
from learning.data_generators import generate_instances
from learning.tsp.dataset import DataSet as TSPLikeDataSet
from learning.pctsp.dataset import DataSet as PCTSPDataSet
from learning.cvrp.dataset import DataSet as CVRPLikeDataSet
from learning.mclp.dataset import DataSet as MCLPDataSet
from learning.jssp.dataset import DataSet as JSSPLikeDataSet
from learning.mis.dataset import DataSet as MISDataSet
from data_tools.cvrp_train_trajectory import prepare_one_instance as prepare_one_cvrp_instance
from data_tools.pctsp_train_trajectory import prepare_one_instance as prepare_one_pctsp_instance
from data_tools.mclp_train_trajectory import prepare_one_instance as prepare_one_mclp_instance
from data_tools.jssp_train_trajectory import prepare_one_instance as prepare_one_jssp_instance




def generate_dataset(net, problem, batch_size, num_samples, problem_size, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("---- Sampling trajectories ----- ")
    net.eval()
    batch_of_objectives, batch_of_coords, batch_of_dist_matrices = list(), list(), list()
    batch_of_demands, batch_of_capacities = list(), list()
    batch_of_remaining_capacities, batch_of_via_depots = list(), list()
    batch_of_distance_constraints, batch_of_remaining_distances = list(), list()
    batch_of_prizes, batch_of_penalties = list(), list()
    batch_of_prizes_to_collect, batch_of_solution_lens = list(), list()
    batch_of_radiuses, batch_of_solutions, batch_of_covering_nodes = list(), list(), list()
    batch_of_num_facilities, batch_of_scales = list(), list()
    batch_of_execution_times, batch_of_task_on_machines, batch_of_precedences = list(), list(), list()
    batch_of_job_task_matrix, batch_of_task_availability_times, batch_of_machine_availability_times = list(), list(), list()
    batch_of_adj_matrices = list()

    for _ in tqdm(range(batch_size)):
        instance = generate_instances(problem, problem_size, dataset=dataset)
        # create a batch of the same instance
        if problem in ["trp", "sop"]:
            batch_of_instances = [torch.tensor(instance["dist_matrices"], device=device).repeat(num_samples, 1, 1, 1), None]
        elif problem == "pctsp":
            batch_of_instances = [torch.tensor(instance["dist_matrices"],
                                          device=device).repeat(num_samples, 1, 1, 1),
                             torch.tensor(instance["node_prizes"], device=device).repeat(num_samples, 1),
                             torch.tensor(instance["node_penalties"], device=device).repeat(num_samples, 1),
                             torch.tensor(instance["min_collected_prize"], device=device).repeat(num_samples),
                             None, None]
        elif problem in ["ocvrp", "sdcvrp"]:
            batch_of_instances = [
                torch.tensor(instance["dist_matrices"], device=device).repeat(num_samples, 1, 1, 1),
                torch.tensor(instance["node_demands"], device=device).repeat(num_samples, 1),
                torch.tensor(instance["capacity"], device=device).repeat(num_samples, 1),
                None, None, None]
        elif problem == "mclp":
            batch_of_instances = [
                torch.tensor(instance["dist_matrices"], device=device).repeat(num_samples, 1, 1),
                torch.tensor(instance["radiuses"], device=device).repeat(num_samples),
                torch.tensor(instance["num_facilities"], device=device).repeat(num_samples),
                None, None, None, None]
        elif problem == "ossp":
            _instance = instance["instances"]
            bs, num_jobs, num_machines, _ = _instance.shape

            execution_times = torch.tensor(_instance[..., 1].reshape(1, -1), device=device)
            precedences = torch.zeros([num_jobs * num_machines, num_jobs * num_machines], device=device)[None, ...]
            jobs_tasks = torch.zeros([num_jobs * num_machines, num_jobs * num_machines], device=device)[None, ...]
            task_availability_times = torch.zeros(num_machines * num_jobs, device=device)[None, ...]
            machine_availability_times = torch.zeros(num_machines, device=device)[None, ...]
            machine_idx = torch.tensor(_instance[..., 0].reshape(1, -1), device=device)

            task_on_machines = torch.zeros(num_jobs * num_machines, num_machines, device=device)[None, ...]
            for i in range(num_jobs * num_machines):
                task_on_machines[0, i, machine_idx[0, i]] = 1.
            scales = torch.tensor(instance["scales"], device=device)

            batch_of_instances = [execution_times.repeat(num_samples, 1),
                                  task_on_machines.repeat(num_samples, 1, 1),
                                  precedences.repeat(num_samples, 1, 1),
                                  jobs_tasks.repeat(num_samples, 1, 1),
                                  task_availability_times.repeat(num_samples, 1),
                                  machine_availability_times.repeat(num_samples, 1),
                                  scales[None, None].repeat(num_samples, 1),
                                  None]
        elif problem == "mis":
            batch_of_instances = [torch.tensor(instance["adj_matrices"], device=device).repeat(num_samples, 1, 1),
                                  torch.full(instance["adj_matrices"].shape[0:2], True,
                                              device=device).repeat(num_samples, 1),
                                  None, None]
        else:
            raise NotImplementedError

        # sample k trajectories and chose the best
        with torch.no_grad():
            sampled_objective_values, sampled_trajectories = decoding_fn[problem](problem, batch_of_instances, net,
                                                                                  sample=True)
        if problem in ["mclp", "mis"]:
            best_sampled_trajectory_index = torch.argmax(sampled_objective_values)
        else:
            best_sampled_trajectory_index = torch.argmin(sampled_objective_values)

        best_sampled_trajectory = sampled_trajectories[best_sampled_trajectory_index]
        best_objective = sampled_objective_values[best_sampled_trajectory_index].cpu().numpy()
        batch_of_objectives.append(best_objective)

        if problem == "trp":
            coords = instance["node_coords"][0][best_sampled_trajectory]
            batch_of_coords.append(coords)
        elif problem == "sop":
            dist_matrices = instance["dist_matrices"][0][best_sampled_trajectory]
            dist_matrices = dist_matrices[:, best_sampled_trajectory][..., 0]
            batch_of_dist_matrices.append(dist_matrices)
        elif problem == "pctsp":
            solution_tour = best_sampled_trajectory[:torch.where(best_sampled_trajectory[1:] == 0)[0][0] + 2]
            node_coords, prizes, penalties, solution_len = prepare_one_pctsp_instance(instance["node_coords"][0][:-1],
                                                                                      instance["node_prizes"][0][:-1],
                                                                                      instance["node_penalties"][0][:-1],
                                                                                      solution_tour)
            batch_of_coords.append(node_coords)
            batch_of_prizes.append(prizes)
            batch_of_penalties.append(penalties)
            batch_of_prizes_to_collect.append(instance["min_collected_prize"][0])
            batch_of_solution_lens.append(solution_len)

        elif problem in ["ocvrp", "sdcvrp"]:
            tour_len, coords, demands, ordered_route, remaining_capacity, via_depots = \
                prepare_one_cvrp_instance(instance["node_coords"][0][:-1], instance["node_demands"][0][:-1],
                                          instance["capacity"][0], best_sampled_trajectory, problem,
                                          best_objective)
            batch_of_coords.append(coords)
            batch_of_demands.append(demands)
            batch_of_capacities.append(instance["capacity"][0])
            batch_of_remaining_capacities.append(remaining_capacity)
            batch_of_via_depots.append(via_depots)
        elif problem == "mclp":
            solutions = np.where(best_sampled_trajectory)[0]
            covering_nodes, num_covered = prepare_one_mclp_instance(instance["dist_matrices"][0],
                                                                    instance["radiuses"][0], solutions)
            assert num_covered == best_objective
            batch_of_dist_matrices.append(instance["dist_matrices"][0])
            batch_of_radiuses.append(instance["radiuses"][0])
            batch_of_num_facilities.append(instance["num_facilities"][0])
            batch_of_solutions.append(solutions)
            batch_of_covering_nodes.append(covering_nodes)
        elif problem == "ossp":
            _instance = instance["instances"][0]
            num_jobs, num_machines, _ = _instance.shape
            machine_idx = torch.arange(num_machines)[:, None].repeat(1, num_machines).flatten()
            solution = np.concatenate([machine_idx[:, None],
                                       best_sampled_trajectory.reshape(-1, 4).astype(np.int32)], axis=1)

            execution_times = _instance[..., 1].flatten()
            task_on_machines = _instance[..., 0].flatten()
            (execution_times, task_on_machines, _, job_task_matrix, task_availability_times,
             machine_availability_times) = prepare_one_jssp_instance(num_jobs, num_machines, execution_times,
                                                                     task_on_machines, solution)
            batch_of_execution_times.append(execution_times)
            batch_of_task_on_machines.append(task_on_machines)
            batch_of_job_task_matrix.append(job_task_matrix)
            batch_of_task_availability_times.append(task_availability_times)
            batch_of_machine_availability_times.append(machine_availability_times)
            batch_of_scales.append(np.array(instance["scales"])[None])
        elif problem == "mis":
            num_nodes = instance["adj_matrices"][0].shape[1]
            nodes_in_the_solution = best_sampled_trajectory[best_sampled_trajectory != -1]
            other_nodes = set(np.arange(num_nodes)).difference(nodes_in_the_solution)
            instance_order = np.concatenate([nodes_in_the_solution, list(other_nodes)], axis=0)
            adj_matrix = instance["adj_matrices"][0][instance_order][:, instance_order]
            batch_of_solution_lens.append(len(nodes_in_the_solution))
            batch_of_adj_matrices.append(adj_matrix)
        else:
            raise NotImplementedError

    if problem == "trp":
        dataset = TSPLikeDataSet(np.stack(batch_of_coords))
    elif problem == "sop":
        dataset = TSPLikeDataSet(dist_matrices=np.stack(batch_of_dist_matrices))
    elif problem in ["ocvrp", "sdcvrp"]:
        dataset = CVRPLikeDataSet(np.stack(batch_of_coords), None, np.stack(batch_of_demands),
                                  np.stack(batch_of_capacities), via_depots=np.stack(batch_of_via_depots),
                                  remaining_capacities=np.stack(batch_of_remaining_capacities))
    elif problem == "dcvrp":
        dataset = CVRPLikeDataSet(np.stack(batch_of_coords), None, np.stack(batch_of_demands),
                                  np.stack(batch_of_capacities), via_depots=np.stack(batch_of_via_depots),
                                  remaining_capacities=np.stack(batch_of_remaining_capacities),
                                  distance_constraints=np.stack(batch_of_distance_constraints),
                                  remaining_distance_constraints=np.stack(batch_of_remaining_distances))
    elif problem == "pctsp":
        dataset = PCTSPDataSet(np.stack(batch_of_coords), np.stack(batch_of_prizes),
                               np.stack(batch_of_penalties), np.stack(batch_of_prizes_to_collect),
                               np.stack(batch_of_objectives), np.stack(batch_of_solution_lens))
    elif problem == "mclp":
        # we need again to trim covering nodes data, to have same dimensions
        max_len = max([bcv.shape[1] for bcv in batch_of_covering_nodes])
        batch_of_trim_covering_nodes = list()
        for b in batch_of_covering_nodes:
            missing = max_len - b.shape[1]
            if missing == 0:
                batch_of_trim_covering_nodes.append(b)
            else:
                batch_of_trim_covering_nodes.append(
                    np.concatenate([b, np.full((b.shape[0], missing), -1, dtype=np.int32)], axis=1))
        dataset = MCLPDataSet(np.stack(batch_of_dist_matrices), np.stack(batch_of_num_facilities),
                              np.stack(batch_of_radiuses), np.stack(batch_of_solutions),
                              np.stack(batch_of_trim_covering_nodes))
    elif problem == "ossp":
        dataset = JSSPLikeDataSet(num_jobs, num_machines,
                                  np.stack(batch_of_execution_times),
                                  np.stack(batch_of_task_on_machines),
                                  None,
                                  np.stack(batch_of_job_task_matrix),
                                  np.stack(batch_of_task_availability_times),
                                  np.stack(batch_of_machine_availability_times),
                                  np.stack(batch_of_scales))
    elif problem == "mis":
        dataset = MISDataSet(np.stack(batch_of_adj_matrices), np.stack(batch_of_solution_lens))
    else:
        raise NotImplementedError

    objectives = np.stack(batch_of_objectives)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader, objectives
