"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import time
from args import add_common_args, add_common_training_args
from learning.data_iterators import DataIterator
from learning.tester import Tester
from utils.exp import setup_experiment, setup_test_environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_common_training_args(parser)
    args = parser.parse_args()

    setup_experiment(args)

    net = setup_test_environment(args)

    data_iterator = DataIterator(args, ddp=False)

    tester = Tester(args, net, data_iterator.test_datasets)
    tester.load_model(args.pretrained_model)

    start_time = time.time()
    tester.test()
    print("inference time", time.time() - start_time)
