import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx


def generate_locations(problem_size, num_instances, repeat_origin=True):
    coords = np.random.rand(num_instances, problem_size, 2).astype(np.float32)

    if repeat_origin:
        coords = np.concatenate([coords, coords[:, 0:1, :]], axis=1)

    dist_matrices = np.stack([squareform(pdist(c, metric='euclidean')).astype(np.float32) for c in coords])
    return coords, dist_matrices

def generate_instances(problem, problem_size, num_instances=1, dataset=None):
    data = dict()

    if problem == "trp":
        coords, dist_matrices = generate_locations(problem_size, num_instances)
        dist_matrices = np.concatenate([dist_matrices[...,None], dist_matrices.transpose(0, 2, 1)[..., None]], axis=-1)
        data["node_coords"] = coords
        data["dist_matrices"] = dist_matrices
    elif problem == "sop":
        max_val = 1000
        # add dummy node, it will be removed in data_manipulation (same code as for tsp, there that is destination)
        dist_matrices = np.random.randint(low=0, high=max_val, size=(num_instances, problem_size+1, problem_size+1))

        for distance_matrix in dist_matrices:
            for i in range(2, problem_size):
                precedences = np.random.choice(i // 2, np.random.randint(0, i // 2), replace=False)
                distance_matrix[i][precedences] = -1
                np.fill_diagonal(distance_matrix, 0)

        data["dist_matrices"] = (dist_matrices[..., None] / max_val).astype(np.float32)

    elif problem == "pctsp":
        coords, dist_matrices = generate_locations(problem_size + 1, num_instances)
        # distribution taken from MDAM paper
        # https://github.com/liangxinedu/MDAM/blob/19b0bf813fb2dbec2fcde9e22eb50e04675400cd/problems/pctsp/problem_pctsp.py#L115
        penalty_max = 0.12
        penalties = np.random.rand(num_instances, problem_size) * penalty_max
        prizes = np.random.rand(num_instances, problem_size) * 4. / float(problem_size)
        penalties = np.concatenate([np.zeros((num_instances, 1)),
                                    penalties,
                                    np.zeros((num_instances, 1))], axis=1).astype(np.float32)
        prizes = np.concatenate([np.zeros((num_instances, 1)),
                                 prizes,
                                 np.zeros((num_instances, 1))], axis=1).astype(np.float32)

        data["node_coords"] = coords
        data["dist_matrices"] = dist_matrices[..., None]
        data["node_prizes"] = prizes
        data["node_penalties"] = penalties
        data["min_collected_prize"] = np.array([1.] * num_instances).astype(np.float32)
    elif problem in ["ocvrp", "sdcvrp", "dcvrp"]:
        coords, dist_matrices = generate_locations(problem_size + 1, num_instances)

        dist_matrices = np.concatenate([dist_matrices[..., None], dist_matrices.transpose(0, 2, 1)[..., None]], axis=-1)
        node_demands = np.random.randint(1, 10, size=(num_instances, problem_size))
        node_demands = np.concatenate([np.zeros((num_instances, 1)),
                                       node_demands,
                                       np.zeros((num_instances, 1))], axis=1).astype(np.float32)

        data["node_coords"] = coords
        data["dist_matrices"] = dist_matrices
        data["node_demands"] = node_demands
        data["capacity"] = np.array([50.] * num_instances).astype(np.float32)
        if problem == "dcvrp":
            data["distance_constraint"] = np.stack([(0.2 + 2 * dm.max()).item()
                                                    for dm in dist_matrices])[..., None].astype(np.float32)
    elif problem == "mclp":
        _, dist_matrices = generate_locations(problem_size, num_instances, repeat_origin=False)
        data["dist_matrices"] = (100 * dist_matrices).astype(np.int32)
        data["radiuses"] = np.array([10] * num_instances).astype(np.float32)
        data["num_facilities"] = np.array([10] * num_instances).astype(np.int32)
    elif problem == "ossp":
        # by default, number machines is 10. So, if problem size is 100 => 10x10
        # For different jssp size, should be reimplemented
        num_machines = 10
        assert problem_size % num_machines == 0
        num_jobs = problem_size // num_machines
        # exec times are from [2, 100]
        max_exec_val = 100
        exec_times = np.random.randint(2, max_exec_val, size=[num_instances, num_jobs, num_machines])
        machine_idxs = np.array([[np.random.permutation(range(num_machines)) for _ in range(num_jobs)]
                                 for __ in range(num_instances)])

        instance = np.stack([machine_idxs, exec_times], axis=-1)
        data["instances"] = instance
        data["scales"] = max_exec_val
    elif problem == "mis":
        # generate erdos renyi graph with p=0.15
        batch_of_adjacency_matrices = list()
        for i in range(num_instances):
            seed = np.random.randint(100000)
            G = nx.erdos_renyi_graph(n=problem_size, p=0.15, seed=seed)
            nodes = list(G.nodes)
            num_nodes = len(nodes)
            edges = list(G.edges)
            adjacency_matrix = np.zeros((num_nodes, num_nodes))
            for edge in edges:
                adjacency_matrix[edge[0]][edge[1]] = 1.
                adjacency_matrix[edge[1]][edge[0]] = 1.
            batch_of_adjacency_matrices.append(adjacency_matrix)
        data["adj_matrices"] = np.stack(batch_of_adjacency_matrices).astype(np.float32)
    return data
