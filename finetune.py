"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""


import argparse
from args import add_common_args, add_common_training_args, add_common_self_supervised_tuning_args
from learning.tuner import Tuner
from learning.data_iterators import DataIterator
from utils.exp import setup_experiment, init_logger, setup_tune_environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_common_training_args(parser)
    add_common_self_supervised_tuning_args(parser)
    args = parser.parse_args()

    setup_experiment(args)
    net, optimizer, scheduler = setup_tune_environment(args)

    logger = init_logger(args)

    data_iterator = DataIterator(args)

    tuner = Tuner(args, net, data_iterator.train_datasets, data_iterator.test_datasets, optimizer, logger)
    tuner.finetune()
