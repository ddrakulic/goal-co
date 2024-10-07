import torch
import numpy as np
from torch import Tensor


def prepare_batch(data, problem, device, sample=True):

    ks = "_s" if sample else ""
    if problem in ["tsp", "trp", "sop"]:
        dist_matrices = data[f"dist_matrices{ks}"].to(device)
        optimal_values = data[f"tour_lens{ks}"].to(device)
        batch_of_data = [dist_matrices, optimal_values]
    elif problem == "pctsp":
        dist_matrices = data[f"dist_matrices{ks}"].to(device)
        node_prices = data[f"node_prizes{ks}"].to(device)
        node_penalties = data[f"node_penalties{ks}"].to(device)
        soluton_lengths = data.get(f"solution_lengths{ks}", None)
        min_collected_prices = data[f"min_collected_prizes{ks}"].to(device)
        optimal_values = data[f"optimal_values{ks}"].to(device)
        batch_of_data = [dist_matrices, node_prices, node_penalties, min_collected_prices, soluton_lengths,
                         optimal_values]
    elif problem in ["cvrp", "sdcvrp", "ocvrp"]:
        dist_matrices = data[f"dist_matrices{ks}"].to(device)
        node_demands = data[f"node_demands{ks}"].to(device)
        total_capacities = data[f"total_capacities{ks}"].to(device)
        remaining_capacities = data[f"remaining_capacities{ks}"].to(device)
        via_depots = data[f"via_depots{ks}"].to(device)
        optimal_values = data[f"tour_lens{ks}"].to(device)
        batch_of_data = [dist_matrices, node_demands, total_capacities, remaining_capacities, via_depots, optimal_values]
    elif problem == "cvrptw":
        dist_matrices = data[f"dist_matrices{ks}"].to(device)
        node_demands = data[f"node_demands{ks}"].to(device)
        total_capacities = data[f"total_capacities{ks}"].to(device)
        remaining_capacities = data[f"remaining_capacities{ks}"].to(device)
        via_depots = data[f"via_depots{ks}"].to(device)
        service_times = data[f"service_times{ks}"].to(device)
        time_windows = data[f"time_windows{ks}"].to(device)
        departure_times = data[f"departure_times{ks}"].to(device)
        optimal_values = data[f"tour_lens{ks}"].to(device)
        batch_of_data = [dist_matrices, node_demands, total_capacities, remaining_capacities,
                         via_depots, service_times, time_windows, departure_times, optimal_values]
    elif problem == "op":
        dist_matrices = data[f"dist_matrices{ks}"].to(device)
        optimal_values = data[f"collected_rewards{ks}"].to(device)
        node_values = data[f"node_values{ks}"].to(device)
        upper_bounds = data[f"upper_bounds{ks}"].to(device)
        batch_of_data = [dist_matrices, node_values, upper_bounds, optimal_values]
    elif problem in ["kp"]:
        weights = data[f"weights{ks}"].to(device)
        values = data[f"values{ks}"].to(device)
        optimal_values = data[f"optimal_values{ks}"].to(device)
        remaining_capacities = data[f"remaining_capacities{ks}"].to(device)
        scale = data[f"scale{ks}"].to(device)
        if "solution_probs_s" in data:
            solutions = data["solution_probs_s"]
        elif "solutions_s" in data:
            solutions = data["solutions_s"]
        else:
            solutions = None

        if solutions is not None:
            solutions = solutions.to(device)
        batch_of_data = [weights, values, remaining_capacities, scale, solutions, optimal_values]
    elif problem == "mclp":
        batch_of_data = [data["dist_matrices"].to(device),
                         data["num_facilities"].to(device),
                         data["radiuses"].to(device),
                         data["covering_nodes"].to(device),
                         data["selected_nodes"].to(device),
                         data["solutions"].to(device),
                         data["objective_values"].to(device)]
    elif problem == "mvc":
        adj_matrices = data[f"adj_matrices{ks}"].to(device)
        optimal_values = data[f"optimal_values"].to(device)
        solution_probs = data.get("solution_probs_s", None)
        if solution_probs is not None:
            solution_probs = solution_probs.to(device)
        batch_of_data = [adj_matrices, solution_probs, optimal_values]
    elif problem == "mis":
        adj_matrices = data["adj_matrices"].to(device)
        optimal_values = data["optimal_values"].to(device)
        can_be_selected = data["can_be_selected"].to(device)
        solution_probs = torch.full((adj_matrices.shape[0], adj_matrices.shape[1]), -np.inf).to(device)
        for b in range(solution_probs.shape[0]):
            solution_probs[b, :data["optimal_values"][b]] = 1.

        batch_of_data = [adj_matrices, can_be_selected, solution_probs, optimal_values]
    elif problem == "upms":
        processing_times = data[f"execution_times{ks}"].to(device)
        solutions = data[f"solutions{ks}"].to(device)
        machine_states = data[f"machine_states{ks}"].to(device)
        scale = data[f"scales{ks}"].to(device)
        optimal_values = data[f"optimal_values{ks}"].to(device)
        batch_of_data = [processing_times, machine_states, solutions, scale, optimal_values]
    elif problem in ["jssp", "ossp"]:
        execution_times = data[f"execution_times{ks}"].to(device)
        jobs_on_machines = data[f"task_on_machines{ks}"].to(device)
        precedencies = data[f"precedencies{ks}"].to(device)
        jobs_tasks = data[f"jobs_tasks{ks}"].to(device)
        task_availability_times = data[f"task_availability_times{ks}"].to(device)
        machine_availability_times = data[f"machine_availability_times{ks}"].to(device)
        scales = data[f"scales{ks}"].to(device)
        optimal_values = data.get("optimal_values", None)

        if optimal_values is not None:
            optimal_values = optimal_values.to(device)
        batch_of_data = [execution_times, jobs_on_machines, precedencies, jobs_tasks, task_availability_times,
                         machine_availability_times, scales, optimal_values]
    else:
        raise NotImplementedError

    return batch_of_data


def prepare_data(data, problem, subproblem_size=None):
    if problem in ["tsp", "trp", "sop", "cvrp", "ocvrp", "sdcvrp", "cvrptw", "op", "pctsp"]:
        if subproblem_size is None:
            subproblem_size = 0
        return prepare_routing_data(data, problem, subproblem_size)
    elif problem in ["mvc", "mis", "kp", "upms", "jssp", "ossp", "mclp"]:
        return prepare_graph_data(data, problem, subproblem_size)


def prepare_routing_data(data, problem, subproblem_size=0) -> tuple[Tensor, Tensor, dict]:
    bs, num_nodes, _, _ = data[0].shape

    if problem == "tsp":
        dist_matrices, _ = data
        node_features = None
        dist_matrices = dist_matrices[:, subproblem_size:, subproblem_size:]
        edge_features = dist_matrices
        problem_data = {"problem_name": problem, "loss": "single_cross_entropy", "is_multitype": False,
                        "seq_len_per_type": None}
    elif problem == "trp":
        dist_matrices, _ = data
        node_features = None
        # data are the same as for tsp, we do not need the last element=start point and second distance dimension
        dist_matrices = dist_matrices[:, subproblem_size:-1, subproblem_size:-1, 0:1]

        edge_features = dist_matrices
        problem_data = {"problem_name": problem, "dist_matrices": dist_matrices,
                        "loss": "single_cross_entropy", "is_multitype": False, "seq_len_per_type": None}
    elif problem == "sop":
        dist_matrices, _ = data
        node_features = None
        # data are the same as for tsp, we do not need the last element=start point and second distance dimension
        dist_matrices = dist_matrices[:, subproblem_size:-1, subproblem_size:-1, 0:1]
        precedence_matrices = torch.zeros(dist_matrices[..., 0:1].shape, device=dist_matrices.device)
        precedence_matrices[dist_matrices[..., 0:1] < 0] = 1
        edge_features = torch.cat([dist_matrices, precedence_matrices], dim=-1)
        problem_data = {"problem_name": problem, "dist_matrices": dist_matrices,
                        "loss": "single_cross_entropy", "is_multitype": False, "seq_len_per_type": None}
    elif problem == "pctsp":
        dist_matrices, node_prices, node_penalties, min_collected_prices, solution_lengths, _ = data
        num_nodes = node_prices.shape[1] - subproblem_size
        collected_so_far = node_prices[:, :subproblem_size].sum(dim=-1)
        remaining_to_collect = min_collected_prices - collected_so_far
        node_features = torch.cat([node_prices[:, subproblem_size:].unsqueeze(-1),
                                   node_penalties[:, subproblem_size:].unsqueeze(-1),
                                   remaining_to_collect[:, None, None].repeat(1, num_nodes, 1)], dim=-1)
        edge_features = dist_matrices[:, subproblem_size:, subproblem_size:]
        problem_data = {"problem_name": problem, "remaining_to_collect": remaining_to_collect,
                        "last_step": subproblem_size == solution_lengths, "num_nodes": num_nodes,
                        "loss": "single_cross_entropy", "is_multitype": False, "seq_len_per_type": None}
    elif problem in ["cvrp", "sdcvrp", "ocvrp"]:
        dist_matrices, node_demands, total_capacities, remaining_capacities, via_depots, _ = data
        edge_features = dist_matrices[:, subproblem_size:, subproblem_size:]
        node_demands = node_demands[:, subproblem_size:]
        remaining_capacities = remaining_capacities[:, subproblem_size]
        if via_depots is not None:
            via_depots = via_depots[:, subproblem_size:]
        num_nodes = edge_features.shape[1]
        node_features = torch.cat([(node_demands / total_capacities)[..., None],
            (remaining_capacities[..., None] / total_capacities)[..., None].repeat(1, num_nodes, 1)], dim=-1)

        problem_data = {"problem_name": problem, "node_demands": node_demands,
                        "remaining_capacities": remaining_capacities, "via_depots": via_depots,
                        "loss": "single_cross_entropy", "is_multitype": False, "seq_len_per_type": None}
    elif problem == "cvrptw":
        # for CVRP, we need to scale capacities and demands (divide with total capacity)
        (travel_times, node_demands, total_capacities, remaining_capacities, via_depots,
            service_times, time_windows, departure_times, _) = data
        # distance matrices are not normalized for TW problems
        dist_matrices = travel_times[:, subproblem_size:, subproblem_size:]
        bs, num_nodes, _, _ = dist_matrices.shape

        time_norm_term = travel_times.reshape(bs, -1).amax(axis=-1)
        # bs, num_nodes, num_nodes
        edge_features = dist_matrices / time_norm_term[:, None, None, None].repeat(1, num_nodes, num_nodes, 2)
        # bs, num_nodes, 2
        time_windows_norm = time_windows[:, subproblem_size:] / time_norm_term[:, None, None].repeat(1, num_nodes, 2)
        # bs, 1
        departure_times_norm = (departure_times[:, subproblem_size] / time_norm_term)[..., None]
        # bs, num_nodes
        service_times_norm = service_times[:, subproblem_size:] / time_norm_term[..., None]

        # bs, num_nodes
        node_demands_norm = node_demands[:, subproblem_size:] / total_capacities
        # bs, 1
        remaining_capacities_norm = remaining_capacities[:, subproblem_size][..., None] / total_capacities

        node_features = torch.cat([node_demands_norm[..., None],
                                   remaining_capacities_norm[:, None, :].repeat(1, num_nodes, 1),
                                   service_times_norm[..., None],
                                   time_windows_norm, departure_times_norm[:, None, :].repeat(1, num_nodes, 1)], dim=-1)

        if via_depots is not None:
            via_depots = via_depots[:, subproblem_size:]

        problem_data = {"problem_name": problem, "travel_times": travel_times[:, subproblem_size:, subproblem_size:],
                        "node_demands": node_demands[:, subproblem_size:],
                        "remaining_capacities": remaining_capacities[:, subproblem_size],
                        "time_windows": time_windows[:, subproblem_size:],
                        "departure_times": departure_times[:, subproblem_size],
                        "via_depots": via_depots, "loss": "single_cross_entropy", "is_multitype": False,
                        "seq_len_per_type": None}
    elif problem == "op":
        dist_matrices, node_prizes, upper_bounds, _ = data
        dist_matrices = dist_matrices[:, subproblem_size:, subproblem_size:]
        node_prizes = node_prizes[:, subproblem_size:]
        edge_features = dist_matrices

        node_features = torch.cat([node_prizes[..., None],
                                   upper_bounds[:, None, None].repeat(1, node_prizes.shape[1], 1)], dim=-1)

        problem_data = {"problem_name": problem, "dist_matrices": dist_matrices, "upper_bounds": upper_bounds,
                        "loss": "single_cross_entropy", "is_multitype": False, "seq_len_per_type": None}
    else:
        raise NotImplementedError

    return node_features, edge_features, problem_data


def prepare_graph_data(data, problem, subproblem_size=0):
    if problem == "mvc":
        if subproblem_size is not None:
            assert NotImplementedError
        adj_matrices, solution_probs, _ = data
        node_features = None
        edge_features = adj_matrices[..., None]
        problem_data = {"problem_name": problem, "solution_probs": solution_probs, "loss": "multi_cross_entropy",
                        "is_multitype": False, "seq_len_per_type": None}
    elif problem == "mis":
        adj_matrices, can_be_selected, solution_probs, _ = data
        node_features = None
        edge_features = adj_matrices[:, subproblem_size:, subproblem_size:][..., None]
        if solution_probs is not None:
            # it is training, create solution probabilities and can_be_selected
            solution_probs = solution_probs[:, subproblem_size:]
            if subproblem_size > 0:
                can_be_selected = adj_matrices[:, :subproblem_size].sum(axis=1) == 0.
                can_be_selected = can_be_selected[:, subproblem_size:]
                assert (torch.count_nonzero(can_be_selected, dim=-1) > 0).all()
        problem_data = {"problem_name": problem, "solution_probs": solution_probs, "can_be_selected": can_be_selected,
                        "loss": "multi_cross_entropy", "is_multitype": False, "seq_len_per_type": None}
    elif problem == "kp":
        if subproblem_size is not None:
            assert NotImplementedError
        weights, values, remaining_capacities, scale, _, solution_probs = data
        weights_norm = weights / scale[..., None]
        values_norm = values / scale[..., None]
        remaining_capacities_norm = remaining_capacities / scale
        edge_features = None
        node_features = torch.cat([weights_norm[..., None], values_norm[..., None],
                                   remaining_capacities_norm[..., None].repeat(1, weights.shape[1])[..., None]],
                                  dim=-1)
        problem_data = {"problem_name": problem, "solution_probs": solution_probs, "weights": weights, "values": values,
                        "remaining_capacities": remaining_capacities, "loss": "multi_cross_entropy",
                        "is_multitype": False, "seq_len_per_type": None}
    elif problem == "upms":
        if subproblem_size is not None:
            assert NotImplementedError
        processing_times, machine_states, solutions, scales, _ = data

        # scale data
        processing_times = processing_times / scales[:, :, None]
        machine_states = (machine_states - machine_states.min(axis=-1)[0][..., None]) / scales

        node_features = machine_states[..., None]
        edge_features = processing_times[..., None]

        problem_data = {"problem_name": problem, "solutions": solutions, "num_jobs": edge_features.shape[1],
                        "num_machines": edge_features.shape[2], "loss": "single_cross_entropy", "is_multitype": True,
                        "seq_len_per_type": [processing_times.shape[1], processing_times.shape[2]]}
    elif problem == "jssp":
        if subproblem_size is not None:
            assert NotImplementedError
        (processing_times, task_on_machines, precedencies, jobs_tasks, task_availability_times,
         machine_availability_times, scales, _) = data
        bs, num_tasks, num_machines = task_on_machines.shape

        # normalize times
        execution_times = processing_times / scales
        # normalize machine/job availability
        min_values = torch.cat([task_availability_times, machine_availability_times], dim=-1).min(dim=-1).values
        task_availability_times = (task_availability_times - min_values.unsqueeze(-1)) / scales
        machine_availability_times = (machine_availability_times - min_values.unsqueeze(-1)) / scales

        node_features = [torch.cat([execution_times.unsqueeze(-1),
                                    task_availability_times.unsqueeze(-1)], dim=-1),
                         torch.cat([machine_availability_times.unsqueeze(-1)], dim=-1)]

        edge_features = [torch.cat([jobs_tasks.unsqueeze(-1), precedencies.unsqueeze(-1)], dim=-1),
                         task_on_machines.unsqueeze(-1)]

        problem_data = {"problem_name": problem, "num_tasks": num_tasks, "num_machines": num_machines,
                        "precedencies": precedencies, "loss": "single_cross_entropy", "is_multitype": True,
                        "seq_len_per_type": [task_availability_times.shape[1], machine_availability_times.shape[1]]}
    elif problem == "ossp":
        # same as JSSP but without precedent constraints
        (processing_times, task_on_machines, _, jobs_tasks, task_availability_times,
         machine_availability_times, scales, _) = data
        bs, num_tasks, num_machines = task_on_machines.shape

        if len(task_availability_times.shape) == 3:
            # training phase, sample subproblems from data
            processing_times = processing_times[:, subproblem_size:]
            task_on_machines = task_on_machines[:, subproblem_size:]
            jobs_tasks = jobs_tasks[:, subproblem_size:, subproblem_size:]
            task_availability_times = task_availability_times[:, subproblem_size, subproblem_size:]
            machine_availability_times = machine_availability_times[:, subproblem_size]

        # normalize times
        execution_times = processing_times / scales
        # normalize machine/job availability
        min_values = torch.cat([task_availability_times, machine_availability_times], dim=-1).min(dim=-1).values
        task_availability_times = (task_availability_times - min_values.unsqueeze(-1)) / scales
        machine_availability_times = (machine_availability_times - min_values.unsqueeze(-1)) / scales

        node_features = [torch.cat([execution_times.unsqueeze(-1),
                                    task_availability_times.unsqueeze(-1)], dim=-1),
                         torch.cat([machine_availability_times.unsqueeze(-1)], dim=-1)]

        edge_features = [jobs_tasks.unsqueeze(-1), task_on_machines.unsqueeze(-1)]

        problem_data = {"problem_name": problem, "num_tasks": num_tasks, "num_machines": num_machines,
                        "loss": "single_cross_entropy", "is_multitype": True,
                        "seq_len_per_type": [task_availability_times.shape[1], machine_availability_times.shape[1]]}
    elif problem == "mclp":
        dist_matrices, num_facilities, radiuses, covered_nodes, already_selected, solutions, _ = data
        bs, num_nodes, _ = dist_matrices.shape

        if solutions is not None:
            solutions = solutions[:, subproblem_size:]
            already_selected = already_selected[:, subproblem_size]
            covered_nodes = covered_nodes[:, subproblem_size]

        # todo: add scale to the data!
        edge_features = (dist_matrices / 100).unsqueeze(-1)

        node_features = torch.cat([covered_nodes[..., None],
                                   (radiuses / 100)[..., None].repeat(1, covered_nodes.shape[1])[..., None]], dim=-1)
        problem_data = {"problem_name": problem, "covered_nodes": covered_nodes,
                        "already_selected": already_selected, "solutions": solutions, "loss": "multi_cross_entropy",
                        "seq_len_per_type": None, "is_multitype": False}
    else:
        raise NotImplementedError

    return node_features, edge_features, problem_data


def create_ground_truth(bs, problem_data, device):
    problem = problem_data["problem_name"]
    if problem in ["tsp", "trp", "op", "sop"]:
        ground_truth = torch.ones((bs, ), dtype=torch.long, device=device)
    elif problem in ["cvrp", "cvrptw", "sdcvrp", "ocvrp"]:
        assert "via_depots" in problem_data
        ground_truth = torch.full((bs,), 2, dtype=torch.long, device=device)
        # update ground truth for edges via depot (data[-1])
        ground_truth[problem_data["via_depots"][:, 1] == 1.] += 1
    elif problem in ["mvc", "mis", "kp"]:
        assert "solution_probs" in problem_data
        ground_truth = torch.softmax(problem_data["solution_probs"], dim=-1)
    elif problem == "pctsp":
        ground_truth = torch.ones((bs,), dtype=torch.long, device=device)
        ground_truth[problem_data["last_step"]] = problem_data["num_nodes"] - 1
    elif problem == "upms":
        ground_truth = torch.nonzero(problem_data["solutions"][:, 0])[:, 1]
    elif problem in ["jssp", "ossp"]:
        ground_truth = torch.zeros((bs,), dtype=torch.long, device=device)
    elif problem == "mclp":
        assert "solutions" in problem_data
        solutions = problem_data["solutions"]
        bs, num_nodes = problem_data["already_selected"].shape
        ground_truth = torch.full((bs, num_nodes), -np.inf, device=problem_data["solutions"].device)
        ground_truth.scatter_(1, solutions, torch.ones_like(solutions, dtype=torch.float))
        ground_truth = torch.softmax(ground_truth, dim=-1)

    return ground_truth
