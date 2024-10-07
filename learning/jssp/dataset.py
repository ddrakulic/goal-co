"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import warnings
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import torch
import random
from torch.utils.data.dataloader import default_collate
import os


class DotDict(dict):
    def __init__(self, **kwds):
        super().__init__()
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, num_jobs, num_machines, execution_times, task_on_machines, precedencies, jobs_tasks,
                 task_availability_times, machine_availability_times, scales, optimal_values=None):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.execution_times = execution_times
        self.task_on_machines = task_on_machines
        self.scales = scales
        self.precedencies = precedencies
        self.jobs_tasks = jobs_tasks
        self.task_availability_times = task_availability_times
        self.machine_availability_times = machine_availability_times
        self.optimal_values = optimal_values

    def __len__(self):
        return len(self.execution_times)

    def __getitem__(self, item):
        # From list to tensors as a DotDict
        item_dict = DotDict()

        if self.precedencies is not None:
            precedences = torch.tensor(self.precedencies[item]).float()
            jobs_tasks = torch.tensor(self.jobs_tasks[item]).float()
        else:
            precedences = torch.zeros([self.num_jobs * self.num_machines, self.num_jobs * self.num_machines])
            jobs_tasks = torch.zeros([self.num_jobs * self.num_machines, self.num_jobs * self.num_machines])

            for job_id in range(self.num_jobs):
                precedences[job_id * self.num_machines:(job_id+1) * self.num_machines,
                             job_id * self.num_machines:(job_id+1) * self.num_machines] = (
                    torch.tril(torch.ones(self.num_machines, self.num_machines), diagonal=-1))
                jobs_tasks[job_id * self.num_machines:(job_id + 1) * self.num_machines,
                    job_id * self.num_machines:(job_id + 1) * self.num_machines] = (
                    torch.ones([self.num_machines, self.num_machines]))

        optimal_values = self.optimal_values[item] if self.optimal_values is not None else np.array([])
        task_availability_times = self.task_availability_times[
            item] if self.task_availability_times is not None else np.zeros(self.num_machines * self.num_jobs)
        machine_availability_times = self.machine_availability_times[
            item] if self.machine_availability_times is not None else np.zeros(self.num_machines)

        task_on_machines = torch.zeros(self.num_jobs * self.num_machines, self.num_machines)
        for i in range(self.num_jobs * self.num_machines):
            task_on_machines[i, self.task_on_machines[item][i]] = 1.

        item_dict.execution_times = torch.tensor(self.execution_times[item]).float()
        item_dict.precedencies = precedences.float()
        item_dict.jobs_tasks = jobs_tasks.float()
        item_dict.task_on_machines = task_on_machines
        item_dict.scales = torch.tensor(self.scales[item]).float()
        item_dict.task_availability_times = torch.tensor(task_availability_times).float()
        item_dict.machine_availability_times = torch.tensor(machine_availability_times).float()
        item_dict.optimal_values = optimal_values

        return item_dict


def load_dataset(path, batch_size, datasets_size, shuffle=False, drop_last=False, what="test", ddp=False):

    if what == "train":
        with open(os.path.join(path, "dataset.info")) as file:
            info = file.readline()
            _datasets_size, num_jobs, num_machines = info.split(",")
            _datasets_size, num_jobs, num_machines = int(_datasets_size), int(num_jobs), int(num_machines)
            if _datasets_size < datasets_size:
                warnings.warn("Required dataset size is smaller than total size")
            datasets_size = min(datasets_size, _datasets_size)

        execution_times = np.memmap(os.path.join(path, "execution_times.dat"), dtype=np.int32, mode="r",
                                    shape=(datasets_size, num_jobs*num_machines))
        task_on_machines = np.memmap(os.path.join(path, "task_on_machines.dat"), dtype=np.int32, mode="r",
                                     shape=(datasets_size, num_jobs * num_machines))
        precedencies = np.memmap(os.path.join(path, "precedencies.dat"), dtype=np.int32, mode="r",
                                 shape=(datasets_size, num_jobs * num_machines, num_jobs * num_machines))
        jobs_tasks = np.memmap(os.path.join(path, "jobs_tasks.dat"), dtype=np.int32, mode="r",
                               shape=(datasets_size, num_jobs * num_machines, num_jobs * num_machines))
        task_availability_times = np.memmap(os.path.join(path, "task_availability_times.dat"), dtype=np.int32, mode="r",
                                            shape=(datasets_size, num_jobs*num_machines, num_jobs*num_machines))
        machine_availability_times = np.memmap(os.path.join(path, "machine_availability_times.dat"), dtype=np.int32,
                                               mode="r", shape=(datasets_size, num_jobs*num_machines, num_machines))
        scales = np.memmap(os.path.join(path, "scales.dat"), dtype=np.int32, mode="r", shape=(datasets_size, 1))
        optimal_values = None
    else:
        data = np.load(path)
        execution_times = data["execution_times"][:datasets_size]
        task_on_machines = data["task_on_machines"][:datasets_size]
        scales = data["scales"][:datasets_size]
        optimal_values = data["optimal_values"][:datasets_size]
        num_jobs = data["num_jobs"].item()
        num_machines = data["num_machines"].item()
        precedencies, jobs_tasks, task_availability_times, machine_availability_times = None, None, None, None

    collate_fn = collate_func if what == "train" else None

    dataset = DataSet(num_jobs, num_machines, execution_times, task_on_machines, precedencies, jobs_tasks,
                      task_availability_times, machine_availability_times, scales, optimal_values)

    if ddp:
        sampler = DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None

    dataset = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, collate_fn=collate_fn,
                         sampler=sampler)
    return dataset


def collate_func(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    """
    num_tasks, num_machines = l_dataset_items[0]["task_on_machines"].shape
    num_removed_tasks = random.randint(0, num_tasks - 1)
    l_dataset_items_new = []
    for d in l_dataset_items:
        d_new = dict()

        d_new["precedencies_s"] = d["precedencies"][num_removed_tasks:, num_removed_tasks:]
        d_new["jobs_tasks_s"] = d["jobs_tasks"][num_removed_tasks:, num_removed_tasks:]
        d_new["task_on_machines_s"] = d["task_on_machines"][num_removed_tasks:]
        d_new["execution_times_s"] = d["execution_times"][num_removed_tasks:]
        d_new["task_availability_times_s"] = d["task_availability_times"][num_removed_tasks][num_removed_tasks:]
        d_new["machine_availability_times_s"] = d["machine_availability_times"][num_removed_tasks]
        d_new["scales_s"] = d["scales"]

        l_dataset_items_new.append({**d, **d_new})

    return default_collate(l_dataset_items_new)
