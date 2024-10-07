"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import subprocess
import random
import numpy as np
import torch
from torch import Tensor
from scipy.spatial.distance import pdist, squareform


def get_params_to_log(args):
    args.update(
        {'commit_id': subprocess.run(
            ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE).stdout.decode('utf-8')[:-1]}
    )
    return args


def do_lr_decay(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        new_learning_rate = param_group['lr'] * decay_rate
        param_group['lr'] = new_learning_rate
    print("New learning rate {:.6f}".format(new_learning_rate))


def set_seed(seed: int):
    if seed is None:
        seed = random.randint(0, 1e5)
    print("Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)  # CAREFUL if doing sampling inside dataloaders!
    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed(seed)


def model_num_params(module):
    sum = 0
    for name, values in module.state_dict().items():
        dim = 1
        for val in values.shape:
            dim *= val
        sum += dim
    return sum


class EpochMetrics:
    # dict of metrics values over epoch
    # makes sure the same metric names are given for each update

    def __init__(self):
        self.metrics = None

    def update(self, d):
        d = {k: (v.item() if isinstance(v, Tensor) else v) for k, v in d.items()}
        if self.metrics is None:
            self.metrics = {kd: [vd] for kd, vd in d.items()}
        else:
            for (k, v), (kd, vd) in zip(self.metrics.items(), d.items()):
                assert k == kd
                v.append(vd)

    def get_means(self):
        return {k: np.mean(v) for k, v in self.metrics.items()}


def get_opt_gap(predicted_val, gt_value, problem_name):
    opt_gap = 100 * ((predicted_val - gt_value) / gt_value).mean().item()
    if problem_name == "op" or problem_name == "kp" or problem_name == "mclp" or problem_name == "mis" :
        opt_gap *= -1
    return opt_gap


def compute_tour_lens(paths: Tensor, dist_matrices: Tensor) -> Tensor:
    batch_idx = torch.arange(len(paths))
    distances = torch.sum(torch.stack([dist_matrices[batch_idx, paths[batch_idx, idx], paths[batch_idx, idx + 1]]
       for idx in range(paths.shape[1] - 1)]).transpose(0, 1), axis=1)

    return distances


def obs_to_network_input(obs, problem):
    if problem == "tsp":
        device = obs["node_coords"].device
        node_coords = obs["node_coords"]
        already_visited = obs["already_visited"]
        orig_dest = obs["orig_dest"]
        if len(node_coords.shape) == 2:
            # add "batch" dimension if not exist
            node_coords = node_coords.unsqueeze(0)
            already_visited = already_visited.unsqueeze(0)
            orig_dest = orig_dest.unsqueeze(0)

        matrices = torch.stack([torch.tensor(squareform(pdist(nodes, metric='euclidean')), device=device).float()
                                for nodes in node_coords.cpu().numpy()])

        node_features = None
        problem_data = dict()
        problem_data["problem_name"] = "tsp"
        problem_data["already_visited"] = already_visited
        problem_data["orig_dest"] = orig_dest

    elif problem == "mclp":
        device = obs["node_coords"].device
        num_nodes = obs["node_coords"].shape[1]
        matrices = torch.stack([torch.tensor(squareform(pdist(nodes, metric='euclidean')), device=device).float()
                                for nodes in obs["node_coords"].cpu().numpy()])
        node_features = torch.cat([obs["node_values"].unsqueeze(-1), obs["covered_nodes"].unsqueeze(-1),
                                   obs["radius"].unsqueeze(-1).unsqueeze(-1).repeat(1, num_nodes, 1)], dim=-1)
        problem_data = dict()
        problem_data["problem_name"] = "mclp"
        problem_data["already_selected"] = obs["selected_nodes"]

    elif problem == "mis":
        problem_data = dict()
        problem_data["problem_name"] = "mis"
        problem_data["can_be_selected"] = obs["can_be_selected"]
        node_features = torch.cat([obs["node_weights"].unsqueeze(-1), obs["can_be_selected"].unsqueeze(-1)], dim=-1)
        matrices = obs["adj_matrix"]

    elif problem == "maxbi":
        problem_data = dict()
        node_features = obs["node_data"].unsqueeze(-1)
        problem_data["problem_name"] = "maxbi"
        problem_data["origin_idx"] = obs["origin_idx"]
        matrices = obs["adj_matrix"]

    return matrices, node_features, problem_data


def load_backbone_parameters(net, path_to_pretrained_model):
    if torch.cuda.is_available():
        pretrained_model = torch.load(path_to_pretrained_model)
    else:
        pretrained_model = torch.load(path_to_pretrained_model, map_location="cpu")
    pretrained_state_dict = pretrained_model["net"]
    current_model_state_dict = net.state_dict()
    loaded_keys = 0
    for key, params in pretrained_state_dict.items():
        current_model_state_dict[key].copy_(pretrained_state_dict[key])
        loaded_keys += 1

    print("Loaded ", loaded_keys, "keys from", path_to_pretrained_model)


def save_model(checkpointer, label, module, epoch_done, best_current_val_metric, optimizer, complete=False):
    assert label in ["current", "best"]
    args = {"module": module}
    if not complete:
        args_ = {"optimizer": None, 'label': label}
    else:
        assert label == "current"
        other = {"epoch_done": epoch_done,
                 "best_current_val_metric": best_current_val_metric}

        args_ = {"optimizer": optimizer,
                 "label": label+"_FULL",
                 "other": other}

    checkpointer.save(**args, **args_)


def remove_model(checkpointer, label):
    assert label in ["current", "best"]
    checkpointer.delete(label)
