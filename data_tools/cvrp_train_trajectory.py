"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""


import numpy
import numpy as np
from scipy.spatial.distance import pdist, squareform
FACTOR = 10000.


def prepare_one_instance(coords, demands, capacity, route, type, cost_from_solver=None):
    problem_size = coords.shape[0]
    route_with_subtours, subroute = list(), list()
    via_depot = [0] * problem_size

    for node_idx in route[1:]:
        if node_idx == 0:
            route_with_subtours.append(subroute)
            subroute = list()
        else:
            subroute.append(node_idx)
    route_with_subtours.append(subroute)
    # order subtours by remaining capacity

    route_remaining_capacities = list()
    for _route in route_with_subtours:
        tour_capacity = capacity
        for node_idx in _route:
            tour_capacity -= demands[node_idx]
        route_remaining_capacities.append(tour_capacity)

    route_idxs = np.argsort(route_remaining_capacities)

    route_ordered_by_remaining_capacity = [0]
    route_current_capacity = [capacity]
    for num_tour in route_idxs:
        route_ordered_by_remaining_capacity.extend(route_with_subtours[num_tour])
        first = True
        for node in route_with_subtours[num_tour]:
            if first:
                route_current_capacity.append(capacity - demands[node])
                first = False
            else:
                route_current_capacity.append(route_current_capacity[-1] - demands[node])

    W = squareform(pdist(coords, metric='euclidean'))

    cost = 0
    if type == "ocvrp":
        for i in range(0, len(route) - 1):
            if route[i + 1] != 0:
                cost += W[route[i], route[i + 1]]
    else:
        for i in range(0, len(route)-1):
            cost += W[route[i], route[i + 1]]

    if cost_from_solver is not None:
        assert numpy.isclose(cost, cost_from_solver, atol=1e-4)

    route_ordered_by_remaining_capacity.append(0)
    for i in range(1, len(route_current_capacity)):
        if route_current_capacity[i] > route_current_capacity[i - 1]:
            via_depot[i] = 1
    via_depot.append(0)

    coords = coords[route_ordered_by_remaining_capacity]
    demands = demands[route_ordered_by_remaining_capacity]

    return cost, coords, demands, route_ordered_by_remaining_capacity, route_current_capacity, via_depot
