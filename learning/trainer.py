"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import os
import time
import torch
from torch import nn
import numpy
from learning.validaton import validate_model
from utils.data_manipulation import prepare_batch, prepare_data, create_ground_truth
from utils.misc import EpochMetrics
from utils.multi_class_loss import CrossEntropyLoss as MultiClassEntropyLoss

DEBUG_NUM_BATCHES = 1


class Trainer:

    def __init__(self, args, model, optimizer, scheduler, epochs_done=0, best_current_val_metric=None, num_pus=1):
        self.problems = args.problems

        # optimizer is defined on the self.model
        self.model = model
        self.optimizer = optimizer

        self.train_batch_size = args.train_batch_size
        self.train_datasets = None
        self.val_datasets = None
        self.test_datasets = None
        self.epochs_done = epochs_done
        self.scheduler = scheduler
        self.update_lr_every_n_epoch = args.update_lr_every_n_epoch
        self.logger = None
        self.output_dir = args.output_dir
        self.job_id = args.job_id
        self.best_current_val_metrics_per_problem = best_current_val_metric
        self.debug = args.debug

        try:
            self.test_every = args.test_every if args.test_every > 0 else None
        except AttributeError:
            self.test_every = None
        self.nb_total_epochs = args.num_total_epochs
        self.val_every = args.val_every
        self.loss = {"single_cross_entropy": nn.CrossEntropyLoss(),
                     "multi_cross_entropy": MultiClassEntropyLoss()}

    def train(self):
        """
        train the model
        """
        if hasattr(self.model, "module"):
            device = self.model.module.get_device()
        else:
            device = self.model.get_device()

        assert self.train_datasets is not None and self.val_datasets is not None and self.test_datasets is not None

        for _ in range(self.epochs_done, self.nb_total_epochs):
            # Train one epoch
            start = time.time()
            self.model.train()
            epoch_metrics_train = EpochMetrics()

            self.multi_training_step(epoch_metrics_train, device)

            self.epochs_done += 1
            if self.epochs_done % self.update_lr_every_n_epoch == 0:
                self.scheduler.step()

            # logging, validation and testing only on device 0 (=None in case of cpu)
            if device.index == 0 or device.index is None:
                device_name = device.type
                if device.index is not None:
                    device_name = device_name + ":" + str(device.index)
                print("[EPOCH {:03d} {}] | Time: {:.3f}s ".format(self.epochs_done, device_name, time.time() - start))

                if self.logger is not None:
                    self.logger.record({f"{k}_train_" + device_name:
                                            v for k, v in epoch_metrics_train.metrics.items()}, self.epochs_done)
                    self.logger.record({"learning_rate": self.scheduler.get_last_lr()[0]}, self.epochs_done)

                if self.epochs_done % self.val_every == 0:
                    val_metrics_per_problem = validate_model(self.model, self.val_datasets, self.epochs_done, self.logger,
                                                             self.debug, what="val")

                    if self.best_current_val_metrics_per_problem is None:
                        self.best_current_val_metrics_per_problem = val_metrics_per_problem
                    else:
                        val_metrics_mean = (numpy.array(list(val_metrics_per_problem.values())) / numpy.array(
                            list(self.best_current_val_metrics_per_problem.values()))).mean()

                        # if val_metrics_mean < 1, there is an improvement
                        if val_metrics_mean < 1:
                            self.best_current_val_metrics_per_problem = val_metrics_per_problem
                            self.save_model(self.epochs_done, 'best')  # only model

                # test
                if self.test_every is not None:
                    if self.epochs_done % self.test_every == 0:
                        self.save_model(self.epochs_done, "current")
                        self.load_model("best")
                        validate_model(self.model, self.test_datasets, self.epochs_done, self.logger, self.debug, what="test")
                        self.load_model("current")

                # Save COMPLETE version of model (with optimizer, metric, iteration, current data) for possible reload.
                if device.index == 0 and not self.debug:
                    self.save_model(self.epochs_done, "current_FULL")

    def multi_training_step(self, epoch_metrics_train, device):

        for batch_num, all_data in enumerate(zip(*self.train_datasets.values())):

            for task in numpy.random.permutation(len(self.problems)):
                self.optimizer.zero_grad()

                problem = self.problems[task]
                data = all_data[task]

                batch_of_data = prepare_batch(data, problem, device, sample=True)
                node_features, edge_features, problem_data = prepare_data(batch_of_data, problem)
                ground_truth = create_ground_truth(self.train_batch_size, problem_data, device)

                # do forward
                output_scores = self.model(node_features, edge_features, problem_data)

                loss = self.loss[problem_data["loss"]](output_scores, ground_truth)

                epoch_metrics_train.update({"loss": loss})
                loss.backward()

                self.optimizer.step()

            if batch_num == DEBUG_NUM_BATCHES and self.debug:
                break

    def save_model(self, epochs_done: int, label: str):
        assert label in ["best", "current", "current_FULL"]
        path = os.path.join(os.path.join(self.output_dir, "models", (str(self.job_id) + "." + label)))

        try:
            # if net is a DataParallel object, save its module
            module = self.model.module
        except AttributeError:
            module = self.model

        if label == "best" or label == "current":
            chk = {"net": module.state_dict(),
                   "optimizer": None,
                   "other": None}
        else:
            other = dict()
            other["epoch_done"] = epochs_done
            other["best_current_val_metrics_per_problem"] = self.best_current_val_metrics_per_problem
            chk = {"net": module.state_dict(),
                   "optimizer": self.optimizer.state_dict(),
                   "other": other}

        torch.save(chk, path)
        print("Saved", path)

    def load_model(self, label: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        path_to_model = os.path.join(os.path.join(self.output_dir, "models", str(self.job_id) + "." + label))
        checkpoint = torch.load(path_to_model, map_location=device)
        try:
            # if net is a DataParallel object, load module
            self.model.module.load_state_dict(checkpoint["net"])
        except AttributeError:
            self.model.load_state_dict(checkpoint["net"])

        print("Loaded", path_to_model)
