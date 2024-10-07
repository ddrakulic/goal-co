"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import os
import torch
from model.goal import GOAL
from utils.misc import set_seed, get_params_to_log
from utils.watcher import MetricsLogger


def setup_experiment(args):
    set_seed(args.seed)
    print(args)

    for d in [args.output_dir, os.path.join(args.output_dir, "models")]:
        os.makedirs(d, exist_ok=True)


def init_logger(args):
    aimstack_tracking_uri = None
    if args.backends[4] == '1':
        assert os.path.exists(args.aimstack_dir)
        aimstack_tracking_uri = args.aimstack_dir

    logger = MetricsLogger(backends=args.backends, exp_name=str(args.job_id), project_name=args.project_name,
                           exp_res_dir=args.output_dir,  aimstack_dir=aimstack_tracking_uri,
                           hps=get_params_to_log(vars(args)))
    return logger


def init_net_and_optimizer(args, is_tuning=False):

    net = GOAL(args.dim_node_idx, args.dim_emb, args.num_layers, args.dim_ff, args.activation_ff,
               args.node_feature_low_dim, args.edge_feature_low_dim, args.activation_edge_adapter,
               args.num_heads, args.attention_clip_value, is_tuning)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)

    return net, optimizer, scheduler


def setup_train_environment(args):

    net, optimizer, scheduler = init_net_and_optimizer(args)
    other = None

    if args.pretrained_model is not None:
        assert args.pretrained_model.split(".")[-1] == "current_FULL"
        checkpointer_filename = args.pretrained_model
    else:
        # check is there is current_FULL model -> automatic continuation of the training (same job id)
        checkpointer_filename = os.path.join(args.output_dir, "models", str(args.job_id) + ".current_FULL")

    if os.path.exists(checkpointer_filename):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpointer = torch.load(checkpointer_filename, map_location=device)
        net.load_state_dict(checkpointer["net"], strict=False)

        print("Load checkpointer model", checkpointer_filename)
        if checkpointer["optimizer"] is not None:
            optimizer.load_state_dict(checkpointer["optimizer"])

        other = checkpointer["other"]
        if "meta_learning_eps" in other:
            args.meta_learning_eps = other["meta_learning_eps"]

        # if we continue the training, the best model for current run does not exist
        best_model_filename = os.path.join(args.output_dir, "models", str(args.job_id) + ".best")
        if not os.path.exists(best_model_filename):
            torch.save({"net": checkpointer["net"], "optimizer": None, "other": None},
                       best_model_filename)
            print(best_model_filename, " params saved from pretrained model")
    else:
        print("Checkpointer file does not exist. Training from scratch")

    return net, optimizer, scheduler, other


def setup_tune_environment(args):
    net, optimizer, scheduler = init_net_and_optimizer(args, is_tuning=True)

    if torch.cuda.is_available():
        net = net.to("cuda")
    return net, optimizer, scheduler


def setup_test_environment(args):
    net = GOAL(args.dim_node_idx, args.dim_emb, args.num_layers, args.dim_ff, args.activation_ff,
               args.node_feature_low_dim, args.edge_feature_low_dim, args.activation_edge_adapter, args.num_heads,
               args.attention_clip_value)
    if torch.cuda.is_available():
        net = net.to("cuda")
    return net
