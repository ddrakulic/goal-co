"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import numpy

MACHINE_ID, JOB_ID, RANK, START_TIME, END_TIME = 0, 1, 2, 3, 4

def prepare_one_instance(num_machines, num_jobs, execution_times, task_on_machines, solution):
    num_tasks = num_jobs * num_machines
    sub_task_availability_times, sub_machine_availability_times = list(), list()
    sol_order_by_start_time = numpy.argsort(solution[:, START_TIME])
    task_idx_ordered_by_start_time = (num_machines * solution[..., JOB_ID] +
                                      solution[..., RANK])[sol_order_by_start_time]

    precedences = numpy.zeros([num_jobs * num_machines, num_jobs * num_machines], dtype=numpy.int32)
    job_task_matrix = numpy.zeros([num_jobs * num_machines, num_jobs * num_machines], dtype=numpy.int32)

    for machine_id in range(num_jobs):
        precedences[machine_id * num_jobs:(machine_id + 1) * num_jobs,
        machine_id * num_jobs:(machine_id + 1) * num_jobs] = (
            numpy.tril(numpy.ones(num_machines), k=-1))
        job_task_matrix[machine_id * num_jobs:(machine_id + 1) * num_jobs,
        machine_id * num_jobs:(machine_id + 1) * num_jobs] = (
            numpy.ones([num_machines, num_machines], dtype=numpy.int32))
    for sub_problem_size in range(num_tasks):

        sol_idx_to_keep = sol_order_by_start_time[sub_problem_size:]
        sol_idx_to_remove = sol_order_by_start_time[:sub_problem_size]

        task_idx_to_keep = (num_machines * solution[..., JOB_ID] + solution[..., RANK])[sol_idx_to_keep]

        assert (task_idx_ordered_by_start_time[sub_problem_size:] == task_idx_to_keep).all()
        task_available_times = numpy.zeros(num_tasks, dtype=numpy.int32)
        machine_available_times = numpy.zeros(num_machines, dtype=numpy.int32)

        already_allocated_tasks = solution[sol_idx_to_remove]
        # remaining_tasks = solutions[sol_idx_to_keep]

        for machine_id in range(num_machines):
            # find end time of last already allocated job on each machine
            if numpy.count_nonzero(already_allocated_tasks[..., MACHINE_ID] == machine_id) != 0:
                machine_available_times[machine_id] = already_allocated_tasks[already_allocated_tasks[...,
                MACHINE_ID] == machine_id][..., END_TIME].max()

        for job_id in range(num_jobs):
            if numpy.count_nonzero(already_allocated_tasks[..., JOB_ID] == job_id) != 0:
                task_available_times[job_id * num_machines: (job_id + 1) * num_machines] = (
                    already_allocated_tasks[already_allocated_tasks[..., JOB_ID] == job_id][..., END_TIME].max())

        # selected_idx = task_idx_to_keep[0]

        sub_task_availability_times.append(task_available_times[task_idx_ordered_by_start_time])
        sub_machine_availability_times.append(machine_available_times)

    execution_times = execution_times[task_idx_ordered_by_start_time]
    precedences = precedences[task_idx_ordered_by_start_time][:, task_idx_ordered_by_start_time]

    job_task_matrix = job_task_matrix[task_idx_ordered_by_start_time][:, task_idx_ordered_by_start_time]
    task_on_machines = task_on_machines[task_idx_ordered_by_start_time]
    task_availability_times = numpy.stack(sub_task_availability_times)
    machine_availability_times = numpy.stack(sub_machine_availability_times)

    return (execution_times, task_on_machines, precedences, job_task_matrix, task_availability_times,
            machine_availability_times)
