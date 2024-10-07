"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from learning.validaton import validate_model
import torch

class Tester:

    def __init__(self, args, net, test_dataset, watcher=None):
        # supervisor for testing the model. can be called from training loop or completely independently

        self.problems = args.problems
        self.job_id = args.job_id
        self.net = net
        self.test_dataset = test_dataset
        self.watcher = watcher
        self.output_dir = args.output_dir
        self.debug = args.debug
        self.beam_size = args.beam_size
        self.knns = args.knns

    def load_model(self, path_to_model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path_to_model, map_location=device)
        self.net.load_state_dict(checkpoint["net"], strict=False)
        print("Loaded", path_to_model)

    def test(self, epoch_done=0):
        validate_model(self.net, self.test_dataset, epoch_done, self.watcher, self.debug, self.beam_size, self.knns,
                       what="test")
