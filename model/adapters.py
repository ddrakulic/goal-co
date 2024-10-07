"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import torch
from torch import nn
from torch.nn import Linear

class NodeAdapter(nn.Module):
    def __init__(self, dim_emb, adapter_low_dim=8, is_finetuning=False):
        super().__init__()

        self.tsp = nn.Parameter(torch.ones(2, adapter_low_dim))
        self.cvrp = nn.Parameter(torch.ones(4, adapter_low_dim))
        self.cvrptw = nn.Parameter(torch.ones(8, adapter_low_dim))
        self.op = nn.Parameter(torch.ones(4, adapter_low_dim))
        self.kp = nn.Parameter(torch.ones(3, adapter_low_dim))

        self.upms = nn.Parameter(torch.ones(2, adapter_low_dim))
        self.jssp1 = nn.Parameter(torch.ones(2, adapter_low_dim))
        self.jssp2 = nn.Parameter(torch.ones(1, adapter_low_dim))

        # tuning
        if is_finetuning:
            self.sop = nn.Parameter(torch.ones(1, adapter_low_dim))
            self.trp = nn.Parameter(torch.ones(1, adapter_low_dim))
            self.pctsp = nn.Parameter(torch.ones(5, adapter_low_dim))
            self.ocvrp = nn.Parameter(torch.ones(4, adapter_low_dim))
            self.sdcvrp = nn.Parameter(torch.ones(4, adapter_low_dim))
            self.dcvrp = nn.Parameter(torch.ones(5, adapter_low_dim))
            self.mclp = nn.Parameter(torch.ones(2, adapter_low_dim))
            self.ossp1 = nn.Parameter(torch.ones(2, adapter_low_dim))
            self.ossp2 = nn.Parameter(torch.ones(1, adapter_low_dim))

        self.input_node_projection = nn.Linear(adapter_low_dim, dim_emb)

    def forward(self, node_features, node_rand_emb, problem_data):
        problem_name = problem_data["problem_name"]

        if problem_name in ["mvc", "mis"]:
            node_emb = node_rand_emb
        elif problem_name in "tsp":
            node_emb = node_rand_emb

            # for tsp, only features are origin and destination tokens
            node_orig_dest_emb = self.input_node_projection(self.tsp)
            node_emb[:, 0, :] = node_rand_emb[:, 0, :] + node_orig_dest_emb[0]
            node_emb[:, -1, :] = node_rand_emb[:, -1, :] + node_orig_dest_emb[1]
        elif problem_name in ["trp", "sop"]:
            node_emb = node_rand_emb
            # for trp, only feature is origin tokens
            node_orig_emb = self.input_node_projection(getattr(self, problem_name))
            node_emb[:, 0, :] = node_rand_emb[:, 0, :] + node_orig_emb[0]

        elif problem_name in ["cvrp", "cvrptw", "op", "ocvrp", "sdcvrp", "dcvrp", "pctsp"]:
            params = getattr(self, problem_name)
            node_proto_emb = node_features @ params[:-2]

            # add origin and destination tokens
            node_proto_emb[:, 0, :] = node_proto_emb[:, 0, :] + params[-2]
            node_proto_emb[:, -1, :] = node_proto_emb[:, -1, :] + params[-1]
            node_emb = node_rand_emb + self.input_node_projection(node_proto_emb)

        elif problem_name in ["jssp", "ossp"]:
            if problem_name == "jssp":
                params = [self.jssp1, self.jssp2]
            else:
                params = [self.ossp1, self.ossp2]

            # Low rank projections
            task_proto_emb = node_features[0] @ params[0]
            machine_proto_emb = node_features[1] @ params[1]

            # Projection to embedding space
            task_emb = self.input_node_projection(task_proto_emb)
            machine_emb = self.input_node_projection(machine_proto_emb)
            node_emb = node_rand_emb + torch.cat([task_emb, machine_emb], dim=1)
        elif problem_name == "upms":
            node_proto_emb = self.upms
            # Low rank projections
            machine_proto_emb = node_features @ node_proto_emb[0:1]

            # Projection to embedding space
            machine_emb = self.input_node_projection(machine_proto_emb)
            task_emb = torch.zeros(machine_emb.shape[0], problem_data["num_jobs"], machine_emb.shape[2],
                                   device=machine_emb.device)
            task_origin_emb = self.input_node_projection(node_proto_emb[1])
            task_emb[:, 0, :] = task_emb[:, 0, :] + task_origin_emb
            node_emb = node_rand_emb + torch.cat([task_emb, machine_emb], dim=1)
        else:
            node_proto_emb = node_features @ getattr(self, problem_name)
            node_emb = node_rand_emb + self.input_node_projection(node_proto_emb)

        return node_emb


class EdgeAdapter(nn.Module):
    def __init__(self, dim_emb, activation="relu", adapter_low_dim=4, is_finetuning=False):
        super().__init__()

        # two-dimensional input for routing problems, can handle both symmetric and asymmetric versions
        # in symmetric version, both dimensions are same (distance from A to B and from B to A)
        self.tsp = nn.Parameter(torch.ones(2, adapter_low_dim))
        self.cvrp = nn.Parameter(torch.ones(2, adapter_low_dim))
        self.cvrptw = nn.Parameter(torch.ones(2, adapter_low_dim))
        self.op = nn.Parameter(torch.ones(2, adapter_low_dim))
        self.mvc = nn.Parameter(torch.ones(1, adapter_low_dim))
        self.upms = nn.Parameter(torch.ones(1, adapter_low_dim))
        self.jssp1 = nn.Parameter(torch.ones(2, adapter_low_dim))
        self.jssp2 = nn.Parameter(torch.ones(1, adapter_low_dim))

        # tuning
        if is_finetuning:
            self.trp = nn.Parameter(torch.ones(1, adapter_low_dim))
            self.sop = nn.Parameter(torch.ones(2, adapter_low_dim))
            self.pctsp = nn.Parameter(torch.ones(1, adapter_low_dim))
            self.ocvrp = nn.Parameter(torch.ones(2, adapter_low_dim))
            self.dcvrp = nn.Parameter(torch.ones(2, adapter_low_dim))
            self.sdcvrp = nn.Parameter(torch.ones(2, adapter_low_dim))
            self.mis = nn.Parameter(torch.ones(1, adapter_low_dim))
            self.mclp = nn.Parameter(torch.ones(1, adapter_low_dim))
            self.ossp1 = nn.Parameter(torch.ones(1, adapter_low_dim))
            self.ossp2 = nn.Parameter(torch.ones(1, adapter_low_dim))


        self.input_edge_projection = nn.Linear(adapter_low_dim, dim_emb)

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        else:
            self.activation = None

    def forward(self, edge_features, problem_data):
        problem_name = problem_data["problem_name"]
        if edge_features is None:
            return None
        elif problem_name in ["jssp", "ossp"]:
            if problem_name == "jssp":
                params = [self.jssp1, self.jssp2]
            else:
                params = [self.ossp1, self.ossp2]

            edge_proto_emb1 = edge_features[0] @ params[0]
            edge_proto_emb2 = edge_features[1] @ params[1]

            edge_emb1 = self.input_edge_projection(edge_proto_emb1)
            edge_emb2 = self.input_edge_projection(edge_proto_emb2)

            if self.activation is not None:
                edge_emb1 = self.activation(edge_emb1)
                edge_emb2 = self.activation(edge_emb2)
            edge_emb = [edge_emb1, None, edge_emb2.transpose(1, 2), edge_emb2]

        elif problem_name == "upms":
            edge_proto_emb = edge_features @ self.upms
            edge_emb = self.input_edge_projection(edge_proto_emb)

            if self.activation is not None:
                edge_emb = self.activation(edge_emb)
            edge_emb = [None, None, edge_emb.transpose(1, 2), edge_emb]
        else:
            edge_proto_emb = edge_features @ getattr(self, problem_name)

            edge_emb = self.input_edge_projection(edge_proto_emb)
            if self.activation is not None:
                edge_emb = self.activation(edge_emb)

        return edge_emb


class OutputAdapter(nn.Module):
    def __init__(self, dim_emb, is_finetuning=False):
        super().__init__()
        self.tsp = Linear(dim_emb, 1)
        self.cvrp = Linear(dim_emb, 2)
        self.cvrptw = Linear(dim_emb, 2)
        self.op = Linear(dim_emb, 1)
        self.kp = Linear(dim_emb, 1)
        self.mvc = Linear(dim_emb, 1)
        self.upms = Linear(dim_emb, 1)
        self.jssp = Linear(dim_emb, 1)

        # tuning
        if is_finetuning:
            self.trp = Linear(dim_emb, 1)
            self.sop = Linear(dim_emb, 1)
            self.pctsp = Linear(dim_emb, 1)
            self.dcvrp = Linear(dim_emb, 2)
            self.ocvrp = Linear(dim_emb, 2)
            self.sdcvrp = Linear(dim_emb, 2)
            self.mis = Linear(dim_emb, 1)
            self.mclp = Linear(dim_emb, 1)
            self.ossp = Linear(dim_emb, 1)

    def forward(self, state, problem_data):
        if problem_data["problem_name"] == "upms":
            state = state[:, -problem_data["num_machines"]:]
        elif problem_data["problem_name"] in ["jssp", "ossp"]:
            state = state[:, :problem_data["num_tasks"]]
        scores = getattr(self, problem_data["problem_name"])(state).squeeze(-1)
        return scores
