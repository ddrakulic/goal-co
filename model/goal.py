import numpy
import numpy as np
import torch
from torch.nn import Module, Linear
from torch.nn.modules import ModuleList
from model.adapters import NodeAdapter, EdgeAdapter, OutputAdapter
from model.layer import Layer


class GOAL(Module):

    def __init__(self, dim_node_idx, dim_emb, num_layers, dim_ff, activation_ff, node_feature_low_dim,
                 edge_feature_low_dim, activation_edge_adapter, num_heads, attention_clip_value, is_finetuning=False):
        super().__init__()
        self.dim_node_idx = dim_node_idx

        self.node_idx_projection = Linear(dim_node_idx, dim_emb)
        self.is_finetuning = is_finetuning

        self.nb_layers = num_layers

        self.node_adapter = NodeAdapter(dim_emb, node_feature_low_dim, is_finetuning)

        self.edge_adapter = EdgeAdapter(dim_emb, activation_edge_adapter, edge_feature_low_dim, is_finetuning)

        self.output_adapter = OutputAdapter(dim_emb, is_finetuning)

        self.layers = ModuleList([Layer(dim_emb, dim_ff, num_heads, attention_clip_value, activation_ff)
                                  for _ in range(num_layers)])

    def get_device(self):
        return list(self.state_dict().items())[0][1].device

    def forward(self, node_features, edge_features, problem_data):
        """
        Forward of the model during the training (data are reordered and prepared to training)
        node features
            TSP None
           CVRP [ batch_size, num_nodes, (demand, current_capacity)]
        CVRP-TW [ batch_size, num_nodes, (demand, service_time, beg_time_windows, end_time_windows,
                  departure_times, remaining_capacity) ]
             OP [ batch_size, num_nodes, (node_value, upper_bound)]
             KP [ batch_size, num_nodes, (weight, value, remaining_capacity)]
           MWVC [ batch_size, num_nodes, (node_weight)]
            JSP [[batch_size, num_nodes, (job_feature)], [batch_size, num_nodes, (machine_feature, machine_state)]]
        """

        batch_size, seq_len, device = self.data_info(node_features, edge_features, problem_data)

        # for some problems we need a mask (e.g. in MVC mask all unconnected nodes)
        mask = self.create_mask(problem_data, edge_features)

        # node projections
        node_random_emb = self.node_idx_projection(torch.rand((batch_size, seq_len, self.dim_node_idx), device=device))

        # input adapters
        state = self.node_adapter(node_features, node_random_emb, problem_data)
        edge_emb = self.edge_adapter(edge_features, problem_data)

        # backbone
        for layer in self.layers:
            state = layer(state, edge_emb, mask, problem_data["is_multitype"], problem_data["seq_len_per_type"])

        # output adapter
        scores = self.output_adapter(state, problem_data)

        # masking infeasible actions
        scores = self.mask_infeasible_actions(scores, mask, problem_data)

        return scores.reshape(scores.shape[0], -1)

    @staticmethod
    def mask_infeasible_actions(scores, mask, problem_data):
        # masking
        if problem_data["problem_name"] == "tsp":
            # mask origin and destination
            scores[:, 0] = scores[:, -1] = -np.inf
        elif problem_data["problem_name"] == "trp":
                # mask origin and destination
                scores[:, 0] = -np.inf
        elif problem_data["problem_name"] == "sop":
            scores[:, 0] = -np.inf
            with_precedence = torch.count_nonzero((problem_data["dist_matrices"][..., 0:1].squeeze(-1)[..., 1:] < 0), dim=-1) > 0
            scores[with_precedence] = -np.inf
        elif problem_data["problem_name"] == "pctsp":
            scores[:, 0] = -np.inf
            scores[:, -1][problem_data["remaining_to_collect"] > 0] = -np.inf
        elif problem_data["problem_name"] in ["cvrp", "sdcvrp", "ocvrp"]:
            # mask origin and destination (x2 - direct edge and via depot)
            scores[:, 0, :] = scores[:, -1, :] = -np.inf
            # exclude all impossible edges (direct edges to nodes with capacity larger than available demand)
            scores[..., 0][problem_data["node_demands"] > problem_data["remaining_capacities"].unsqueeze(-1)] = -np.inf
        elif problem_data["problem_name"] == "dcvrp":
            # mask origin and destination (x2 - direct edge and via depot)
            scores[:, 0, :] = scores[:, -1, :] = -np.inf
            # exclude all impossible edges (direct edges to nodes with capacity larger than available demand)
            scores[..., 0][problem_data["node_demands"] > problem_data["remaining_capacities"].unsqueeze(-1)] = -np.inf
            scores[..., 0][problem_data["dist_matrices"][:, 0, :, 0].squeeze(-1) +
                           problem_data["dist_matrices"][:, :, -1, 0].squeeze(-1) -
                           problem_data["remaining_distances"] > 1e-6] = -np.inf
        elif problem_data["problem_name"] == "cvrptw":
            # mask origin and destination (x2 - direct edge and via depot)
            scores[:, 0, :] = scores[:, -1, :] = -np.inf
            # exclude all impossible edges:
            # direct edges to nodes with demands larger than available capacity
            scores[..., 0][problem_data["node_demands"] > problem_data["remaining_capacities"].unsqueeze(-1)] = -np.inf
            # plus direct edges to nodes with closing time later than arrival time
            scores[..., 0][
                torch.max(problem_data["departure_times"].unsqueeze(-1) + problem_data["travel_times"][:, 0, :, 0],
                          problem_data["time_windows"][..., 0])
                > problem_data["time_windows"][..., 1]] = -np.inf
        elif problem_data["problem_name"] == "op":
            scores[:, 0] = scores[:, -1] = -np.inf
            # op - mask all nodes with cost to go there and back to depot > current upperbound
            scores[problem_data["dist_matrices"][:, 0, :, 0].squeeze(-1) +
                   problem_data["dist_matrices"][:, :, -1, 0].squeeze(-1) -
                   problem_data["upper_bounds"].unsqueeze(-1) > 0] = -np.inf
        elif problem_data["problem_name"] == "mvc":
            scores[mask[..., 0]] = -np.inf
        elif problem_data["problem_name"] == "mis":
            scores[mask[..., 0]] = -np.inf
        elif problem_data["problem_name"] == "mclp":
            scores[mask[..., 0]] = -np.inf
        elif problem_data["problem_name"] == "kp":
            # KP- mask all items with weight > remaining capacity
            scores[problem_data["weights"] > problem_data["remaining_capacities"].unsqueeze(-1)] = -np.inf
        elif problem_data["problem_name"] == "multikp":
            # KP- mask all items with weight > remaining capacity
            filter = (problem_data["weights"][:, None, :].repeat(1, problem_data["remaining_capacities"].shape[1], 1) >
                      problem_data["remaining_capacities"][..., None])
            scores[filter] = -np.inf
        elif problem_data["problem_name"] == "jssp":
            tasks_with_precedences = problem_data["precedencies"].sum(axis=-1) > 0
            scores[tasks_with_precedences] = -np.inf
        return scores

    @staticmethod
    def data_info(node_features, edge_features, problem_data):

        if problem_data["problem_name"] in ["upms", "jssp", "ossp"]:
            if problem_data["problem_name"] in ["jssp", "ossp"]:
                edge_features = edge_features[1]
            batch_size = edge_features.shape[0]
            device = edge_features.device
            seq_len = edge_features.shape[1] + edge_features.shape[2]
        elif problem_data["problem_name"] == "multikp":
            batch_size = node_features[0].shape[0]
            device = node_features[0].device
            seq_len = node_features[0].shape[1] + node_features[1].shape[1]
        else:
            features = edge_features if edge_features is not None else node_features
            batch_size = features.shape[0]
            device = features.device
            seq_len = features.shape[1]

        return batch_size, seq_len, device

    @staticmethod
    def create_mask(problem_data, matrices):
        if problem_data["problem_name"] == "mvc":
            matrices = matrices.squeeze(-1)
            mask = torch.full(matrices.shape, False, device=matrices.device)
            mask[matrices.sum(axis=-1) == 0] = True
        elif problem_data["problem_name"] == "mis":
            mask = torch.full(matrices.squeeze(-1).shape, False, device=matrices.device)
            mask[~problem_data["can_be_selected"]] = True
        elif problem_data["problem_name"] == "mclp":
            mask = torch.full(matrices[..., 0].shape, False, device=matrices.device)
            if problem_data["already_selected"] is not None:
                already_selected = problem_data["already_selected"].long()
                for b in range(mask.shape[0]):
                    mask[b, already_selected[b] == 1] = True
        else:
            mask = None
        return mask

