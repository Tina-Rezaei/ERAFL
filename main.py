import os
import pickle
import copy
from ERAFL import erafl
from ERAS import eras
from BOS_model import bos_model
from EOS import eos
from General_components.drop_task import drop_task
from General_components.init_params import init_parameters
from validate_solution import check_constraints


def model_executor(constant_params, paras, model_name, model_path, iteration):
    range_params = copy.deepcopy(paras)
    model_mapper = {
        'ERAFL': erafl,
        'RAFS': eras,
        'BOS': bos_model
    }
    model_output_name = os.path.join(model_path, f'{model_name}_{iteration}')
    while True:
        model, status = model_mapper[model_name](constant_params, range_params)

        if status == 'optimal':
            with open(model_output_name, 'wb') as f:
                pickle.dump(model, f)
            return model
        if status == 'maxTimeLimit':
            result = check_constraints(model, model_name, iteration, config_params_path)
            if result:
                with open(model_output_name, 'wb') as f:
                    pickle.dump(model, f)
                return model

        # there is no optimal or feasible solution, so, drop task
        range_params = drop_task(range_params, model)


def read_simulation_config(path):
    params = dict()
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            key, value = line.strip('\n').split(',')
            params[key] = float(value)
    return params


if __name__ == '__main__':
    config_params_path = "../simulation_config.txt"
    config_params = read_simulation_config(config_params_path)

    model_path = f"./test_models"
    params_path = f"./test_params"

    comp_load_ratio = {'30': 100, '100': 300, '170': 500, '240': 700, '300': 1000}
    iterations = 2

    # Build directories
    if not os.path.exists(os.path.join(model_path)):
        os.makedirs(os.path.join(model_path))
    for load_ratio in comp_load_ratio.keys():
        if not os.path.exists(os.path.join(model_path, f'{load_ratio}')):
            os.makedirs(os.path.join(model_path, f'{load_ratio}'))
        if not os.path.exists(os.path.join(params_path, f'{load_ratio}')):
            os.makedirs(os.path.join(params_path, f'{load_ratio}'))

    for load_ratio,  load_cycles in comp_load_ratio.items():
        for iteration in range(iterations):
            print(f'iteration : {iteration}')
            constant_parameters, range_parameters = init_parameters(load_ratio, load_cycles, iteration, config_params, params_path)

            number_of_generated_tasks = len(range_parameters[0])
            model_output_path = os.path.join(model_path, f'{load_ratio}')

            bos_m = model_executor(constant_parameters, range_parameters, 'BOS', model_output_path, iteration)

            cc_m = model_executor(constant_parameters, range_parameters, 'ERAFL', model_output_path, iteration)

            rafs_m = model_executor(constant_parameters, range_parameters, 'RAFS', model_output_path, iteration)

            ccfs_m = eos(constant_parameters, range_parameters)[0]

