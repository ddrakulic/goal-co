"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import datetime
import torch.cuda
from torch.distributed import destroy_process_group
import os
from args import add_common_args, add_common_training_args
from learning.data_iterators import DataIterator
from learning.trainer import Trainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.exp import setup_experiment, setup_train_environment
from utils.misc import get_params_to_log
from utils.watcher import MetricsLogger


def run_distributed_training(args, _trainer):
    torch.cuda.empty_cache()
    device_index = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=1800))

    for state in _trainer.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = state[k].to(device_index)

    torch.cuda.set_device(device_index)
    # find_unused_parameters must be true, we can have unused parameters
    _trainer.model = DDP(_trainer.model.to(device_index), device_ids=[device_index], find_unused_parameters=True)

    data_iterator = DataIterator(args, ddp=True)
    _trainer.train_datasets = data_iterator.train_datasets
    _trainer.val_datasets = data_iterator.val_datasets
    _trainer.test_datasets = data_iterator.test_datasets
    # set logger only for gpu:0
    if device_index == 0:
        mlflow_tracking_uri, aimstack_tracking_uri = None, None
        if args.backends[4] == '1':
            assert os.path.exists(args.aimstack_dir)
            aimstack_tracking_uri = args.aimstack_dir

        logger = MetricsLogger(backends=args.backends, exp_name=str(args.job_id), project_name=args.project_name,
                               exp_res_dir=args.output_dir, mlflow_tracking_uri=mlflow_tracking_uri,
                               aimstack_dir=aimstack_tracking_uri, hps=get_params_to_log(vars(args)))

        _trainer.logger = logger
    _trainer.train()

    destroy_process_group()


def run_single_gpu_training(args, _trainer):
    data_iterator = DataIterator(args, ddp=False)
    _trainer.train_datasets = data_iterator.train_datasets
    _trainer.val_datasets = data_iterator.val_datasets
    _trainer.test_datasets = data_iterator.test_datasets
    aimstack_tracking_uri = None
    if args.backends[4] == '1':
        assert os.path.exists(args.aimstack_dir)
        aimstack_tracking_uri = args.aimstack_dir

    logger = MetricsLogger(backends=args.backends, exp_name=str(args.job_id), project_name=args.project_name,
                           exp_res_dir=args.output_dir, aimstack_dir=aimstack_tracking_uri,
                           hps=get_params_to_log(vars(args)))
    _trainer.logger = logger
    if torch.cuda.is_available():
        _trainer.model = _trainer.model.cuda()
        for state in _trainer.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    _trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_common_training_args(parser)
    args = parser.parse_args()

    setup_experiment(args)

    model, optimizer, scheduler, other = setup_train_environment(args)
    best_current_val_metric = other["best_current_val_metrics_per_problem"] if other is not None else None
    epoch_done = other["epoch_done"] if other is not None else 0

    num_gpus = torch.cuda.device_count()
    trainer = Trainer(args, model, optimizer, scheduler, epoch_done, best_current_val_metric, num_gpus)

    if num_gpus > 1:
        run_distributed_training(args, trainer)
    else:
        run_single_gpu_training(args, trainer)
