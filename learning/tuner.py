"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import os
import time
import copy
import torch
import numpy
from torch import nn
from tqdm import tqdm
from learning import decoding_fn
from learning.data_generators import generate_instances
from learning.trajectory_samplers import generate_dataset
from learning.validaton import validate_model
from utils.data_manipulation import prepare_batch, prepare_data, create_ground_truth
from utils.misc import EpochMetrics
from utils.multi_class_loss import CrossEntropyLoss as MultiClassEntropyLoss

DEBUG_NUM_BATCHES = 1


class Tuner:

    def __init__(self, args, net, finetune_dataset, test_dataset, optimizer, logger=None):
        assert len(args.problems) == 1
        self.problem = args.problems[0]
        self.net = net
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finetune_dataset = finetune_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.debug = args.debug
        self.output_dir = args.output_dir
        self.job_id = args.job_id

        self.test_every = args.test_every if args.test_every > 0 else 1
        self.nb_total_epochs = args.num_total_epochs

        self.path_to_pretrained_model = args.pretrained_model

        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.val_batch_size = args.val_batch_size
        self.val_dataset_size = args.val_datasets_size

        if self.finetune_dataset[self.problem] is None:
            # there is no labeled training set, it is self supervised tuner
            self.self_supervised_tuning = True
            self.num_samples = args.num_samples
            self.problem_size = args.problem_size
            self.num_sampled_init_params = args.num_sampled_init_params
        else:
            self.self_supervised_tuning = False

        if self.problem in ["mis", "mclp"]:
            self.loss = MultiClassEntropyLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self.logger = logger

    def load_pretrained_params(self):
        assert self.path_to_pretrained_model is not None
        checkpointer = torch.load(self.path_to_pretrained_model, map_location=self.device)
        nb_loaded_params = 0
        # load all parameters
        for key, params in checkpointer["net"].items():
            if key in self.net.state_dict():
                self.net.state_dict()[key].copy_(params)
            nb_loaded_params += params.flatten().shape[0]

    def finetune(self):
        assert self.problem in ["tsp", "trp", "sop", "pctsp", "ocvrp", "sdcvrp", "mis", "mclp", "ossp"]

        if self.path_to_pretrained_model is not None:
            # load all parameters
            self.load_pretrained_params()
            print("Loaded all parameters from pretrained model.")
        else:
            print("Training from scratch")

        if not self.self_supervised_tuning:
            # tuning dataset in not provided -> self-supervised learning, we need to sample instances
            tuning_dataset = self.finetune_dataset[self.problem]
        else:
            # for some problems, performances are very different for different initialization, so chose the best
            if self.num_sampled_init_params > 1:
                self.find_the_best_initialization()

        # self.test_current_model(self.test_dataset, 0)
        opt_gaps = list()
        finetuning_metrics = EpochMetrics()
        logging_step = 1
        for epoch_done in range(1, self.nb_total_epochs + 1):
            # Train one epoch
            start = time.time()
            metrics = dict()
            if self.self_supervised_tuning:
                tuning_dataset, objectives = generate_dataset(self.net, self.problem, self.train_batch_size,
                                                              self.num_samples, self.problem_size, self.test_dataset)

                metrics.update({"sampled_objectives": objectives.mean()})

            for batch_num, data in enumerate(tuning_dataset):

                if batch_num == DEBUG_NUM_BATCHES and self.debug:
                    break
                loss = self.train_one_batch(data)

                metrics.update({"training_loss": loss})
                finetuning_metrics.update(metrics)
                self.logger.record({f'{k}_train': v for k, v in finetuning_metrics.metrics.items()}, logging_step)
                res = self.test_current_model(self.test_dataset, logging_step)
                self.save_model("_" + str(batch_num))
                logging_step += 1
                opt_gaps.append(res[self.problem])
            print("[EPOCH {:03d}] Time: {:.3f}s ".format(epoch_done, time.time() - start))
        print(opt_gaps)

    def train_one_batch(self, data):

        if self.problem in ["tsp", "trp", "sop", "sdcvrp", "ocvrp"]:
            problem_size = data["dist_matrices"].shape[1]
            max_subproblem_size = problem_size - 2
        elif self.problem == "mis":
            max_subproblem_size = torch.max(data["solution_lengths"]).item()
        elif self.problem == "mclp":
            problem_size = data["num_facilities"].max().item()
            max_subproblem_size = problem_size
        elif self.problem == "pctsp":
            problem_size = data["solution_lengths"].max().item()
            max_subproblem_size = problem_size
        elif self.problem == "ossp":
            max_subproblem_size = data["execution_times"].shape[1] - 2
        else:
            raise NotImplementedError

        data = prepare_batch(data, self.problem, self.device, sample=False)
        self.net.train()

        epoch_metrics = EpochMetrics()
        for subproblem_size in range(max_subproblem_size):
            self.optimizer.zero_grad()

            if self.problem == "pctsp" and torch.count_nonzero(subproblem_size == data[4]):
                # if there are done instances, remove them from the batch
                data =  [d[subproblem_size != data[4]] for d in data]
            elif self.problem == "mis" and torch.count_nonzero(subproblem_size == data[3]):
                data =  [d[subproblem_size != data[3]] for d in data]

            node_features, edge_features, problem_data = prepare_data(data, self.problem, subproblem_size)

            batch_size, device = self.bs_and_device(node_features, edge_features)
            ground_truth = create_ground_truth(batch_size, problem_data, device)

            # do forward
            output_scores = self.net(node_features, edge_features, problem_data)
            loss = self.loss(output_scores, ground_truth)
            loss.backward()
            self.optimizer.step()
            epoch_metrics.update({"training_loss": loss})

        return numpy.mean(epoch_metrics.metrics["training_loss"])

    def test_current_model(self, dataset, epoch_done, what="test"):
        return validate_model(self.net, dataset, epoch_done, self.logger, self.debug, what=what)

    def save_model(self, label=""):
        path = os.path.join(os.path.join(self.output_dir, "models", (str(self.job_id) + label + ".best")))

        try:
            # if net is a DataParallel object, save its module
            module = self.net.module
        except AttributeError:
            module = self.net

        chk = {"net": module.state_dict(),
               "optimizer": None,
               "other": None}
        torch.save(chk, path)
        print("Saved", path)

    @staticmethod
    def bs_and_device(node_features, edge_features):
        if node_features is not None:
            if type(node_features) == list:
                batch_size = node_features[0].shape[0]
                device = node_features[0].device
            else:
                batch_size = node_features.shape[0]
                device = node_features.device
        else:
            if type(edge_features) == list:
                batch_size = edge_features[0]
                device = edge_features[0].device
            else:
                batch_size = edge_features.shape[0]
                device = edge_features.device

        return batch_size, device

    def find_the_best_initialization(self):
        print("Sample", self.num_sampled_init_params, " random params and keep the best")
        best_objective = numpy.inf
        if self.problem in ["mis", "mclp"]:
            best_objective = -best_objective
        best_params = None
        random_instances = generate_instances(self.problem, num_instances=8, problem_size=self.problem_size)
        for _ in tqdm(range(self.num_sampled_init_params)):
            # reset parameters of output adapter
            getattr(self.net.output_adapter, self.problem).reset_parameters()

            if self.problem in ["trp", "sop"]:
                instance_data = [torch.tensor(random_instances["dist_matrices"], device=self.device), None]

            elif self.problem == "pctsp":
                instance_data = [torch.tensor(random_instances["dist_matrices"], device=self.device),
                                 torch.tensor(random_instances["node_prizes"], device=self.device),
                                 torch.tensor(random_instances["node_penalties"], device=self.device),
                                 torch.tensor(random_instances["min_collected_prize"], device=self.device),
                                 None, None]
            elif self.problem in ["ocvrp", "sdcvrp"]:
                instance_data = [
                    torch.tensor(random_instances["dist_matrices"], device=self.device),
                    torch.tensor(random_instances["node_demands"], device=self.device),
                    torch.tensor(random_instances["capacity"][:, None], device=self.device),
                    None, None, None]
            elif self.problem == "mclp":
                instance_data = [torch.tensor(random_instances["dist_matrices"], device=self.device),
                                 torch.tensor(random_instances["num_facilities"], device=self.device),
                                 torch.tensor(random_instances["radiuses"], device=self.device),
                                 None, None, None, None]
            elif self.problem == "ossp":
                instance = random_instances["instances"]
                bs, num_jobs, num_machines, _ = instance.shape

                precedences = torch.stack([torch.zeros([num_jobs * num_machines, num_jobs * num_machines],
                                                       device=self.device) for _ in range(bs)])
                jobs_tasks = torch.stack([torch.zeros([num_jobs * num_machines, num_jobs * num_machines],
                                                      device=self.device) for _ in range(bs)])
                task_availability_times = torch.stack([torch.zeros(num_machines * num_jobs,
                                                                   device=self.device) for _ in range(bs)])
                machine_availability_times = torch.stack([torch.zeros(num_machines, device=self.device)
                                              for _ in range(bs)])

                machine_idx = torch.tensor(instance[..., 0].reshape(bs, -1), device=self.device)

                task_on_machines = torch.stack([torch.zeros(num_jobs * num_machines, num_machines, device=self.device)
                                                for _ in range(bs)])
                for b in range(bs):
                    for i in range(num_jobs * num_machines):
                        task_on_machines[b, i, machine_idx[b, i]] = 1.

                instance_data = [torch.tensor(instance[..., 1].reshape(bs, -1), device=self.device),
                                 task_on_machines,
                                 precedences,
                                 jobs_tasks,
                                 task_availability_times,
                                 machine_availability_times,
                                 torch.tensor(random_instances["scales"], device=self.device),
                                 None]
            elif self.problem == "mis":
                instance_data = [torch.tensor(random_instances["adj_matrices"], device=self.device),
                                 torch.full((random_instances["adj_matrices"].shape[0:2]), True, device=self.device),
                                 None, None]
            else:
                raise NotImplementedError
            with torch.no_grad():
                objective_value, _ = decoding_fn[self.problem](self.problem, instance_data, self.net)

            if self.problem in ["mis", "mclp"]:
                if objective_value.mean() > best_objective:
                    best_objective = objective_value.mean()
                    best_params = copy.deepcopy(self.net.state_dict())
            else:
                if objective_value.mean() < best_objective:
                    best_objective = objective_value.mean()
                    best_params = copy.deepcopy(self.net.state_dict())

        assert best_params is not None
        self.net.load_state_dict(best_params)

        print("Done. Best initialization for output adapter is chosen, obj. value", best_objective.item())
