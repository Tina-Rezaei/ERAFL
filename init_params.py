import random
import os
import pickle
import numpy as np


def generate_random_time_budget(n, lower_bound, upper_bound):
    time_budget = dict()
    for i in range(1, n + 1):
        time_budget[i] = random.uniform(lower_bound, upper_bound)
    return time_budget


def required_computation_initialization(model_size):
    return int((500 / 40) * model_size)


def generate_random_epoch_number(lower_bound, upper_bound):
    return int(np.random.uniform(lower_bound, upper_bound))


def tasks_ids_initialization(number_of_tasks):
    return [i + 1 for i in range(number_of_tasks)]


def logspace_based_random_generator(lower_boundary, upper_boundary, chunk_number):

    # lower_power = int(np.log10(lower_boundary))
    # upper_power = int(np.log10(upper_boundary))
    # boundaries = np.logspace(start=lower_power, stop=upper_power, num=chunk_number)
    # chunk = (upper_boundary- lower_boundary)/chunk_number
    # boundaries = [i*chunk for i in range(0, upper_boundary)]

    boundaries = []
    boundary = lower_boundary
    counter = 0
    while boundary < upper_boundary:
        boundary = boundary + 2**counter
        boundaries.append(boundary)
        counter += 1

    chunks_representative = [np.random.uniform(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    random.shuffle(chunks_representative)
    return random.choice(chunks_representative)


def task_generator(params, load_budget):
    required_comp_list = []
    tasks_data_size = []
    tasks_model_size = []
    tasks_epoch_number = []
    tasks_computation_per_bit = []
    tasks_privacy_scores = []
    while True:
        data_size = logspace_based_random_generator(params['data_size_l'], params['data_size_u'], 100)

        model_size = logspace_based_random_generator(params['model_size_l'], params['model_size_u'], 100)

        epoch_number = generate_random_epoch_number(params['epoch_l'], params['epoch_u'])

        computation_per_bit = required_computation_initialization(model_size)

        privacy_score = random.choice([1, 2, 4, 8, 16])

        required_comp = data_size * epoch_number * computation_per_bit

        # if task load is approved then add all related parameters
        if required_comp < load_budget:
            tasks_data_size.append(data_size)
            tasks_model_size.append(model_size)
            tasks_epoch_number.append(epoch_number)
            tasks_computation_per_bit.append(computation_per_bit)
            required_comp_list.append(required_comp)
            tasks_privacy_scores.append(privacy_score)
        if sum(required_comp_list) > load_budget:
            tasks_data_size = {key: value for key, value in enumerate(tasks_data_size, 1)}
            tasks_model_size = {key: value for key, value in enumerate(tasks_model_size, 1)}
            tasks_epoch_number = {key: value for key, value in enumerate(tasks_epoch_number, 1)}
            tasks_computation_per_bit = {key: value for key, value in enumerate(tasks_computation_per_bit, 1)}
            tasks_privacy_scores = {key: value for key, value in enumerate(tasks_privacy_scores, 1)}
            return tasks_data_size, tasks_model_size, tasks_epoch_number, tasks_computation_per_bit, tasks_privacy_scores


def init_parameters(load_ratio, load_cycles, iter, params, params_path):
    load_budget = params['comp_rsc'] * load_cycles

    tasks_data_size, tasks_model_size, tasks_epoch_number, tasks_computation_per_bit, tasks_privacy_scores = task_generator(params, load_budget)

    number_of_tasks = len(tasks_data_size)

    with open(os.path.join(params_path, f'{load_ratio}', f'{iter}_data.pickle'), 'wb') as f:
        pickle.dump(tasks_data_size, f)

    with open(os.path.join(params_path, f'{load_ratio}', f'{iter}_model.pickle'), 'wb') as f:
        pickle.dump(tasks_model_size, f)

    with open(os.path.join(params_path, f'{load_ratio}', f'{iter}_computation.pickle'), 'wb') as f:
        pickle.dump(tasks_computation_per_bit, f)

    with open(os.path.join(params_path, f'{load_ratio}', f'{iter}_epoch.pickle'), 'wb') as f:
        pickle.dump(tasks_epoch_number, f)

    with open(os.path.join(params_path, f'{load_ratio}', f'{iter}_privacy_score.pickle'), 'wb') as f:
        pickle.dump(tasks_privacy_scores, f)

    tasks_time_budget = generate_random_time_budget(number_of_tasks, params['time_budget_l'], params['time_budget_u'])
    with open(os.path.join(params_path, f'{load_ratio}', f'{iter}_time.pickle'), 'wb') as f:
        pickle.dump(tasks_time_budget, f)

    tasks_ids = tasks_ids_initialization(number_of_tasks)

    # set no-offloading for all the tasks
    n_tasks = len(tasks_data_size)
    tasks_offloading_state = {i + 1: '' for i in range(n_tasks)}

    return [params['bandwidth'], params['backhaul'], params['comp_rsc']], [tasks_data_size, tasks_time_budget,
                                                                           tasks_computation_per_bit,
                                                                           tasks_epoch_number, tasks_model_size,
                                                                           tasks_privacy_scores,
                                                                           tasks_offloading_state, tasks_ids]
