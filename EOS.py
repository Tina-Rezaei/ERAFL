import pyomo.environ as pyo
from pyomo.opt import TerminationCondition


def bandwidth_allocation(tasks_ids, total_bandwidth):
    bandwidth = dict()
    for id in tasks_ids:
        bandwidth[id] = total_bandwidth / len(tasks_ids)
    return bandwidth


def backhaul_bandwidth_allocation(tasks_ids, total_backhaul_bandwidth):
    backhaul_bandwidth = dict()
    for id in tasks_ids:
        backhaul_bandwidth[id] = total_backhaul_bandwidth / len(tasks_ids)
    return backhaul_bandwidth


def computation_allocation(tasks_ids, total_cpu_cycles):
    cpu_cycles = dict()
    for id in tasks_ids:
        cpu_cycles[id] = total_cpu_cycles / len(tasks_ids)
    return cpu_cycles


def completion_time(ids, data_size, epochs, comp_per_bit, model_size, backhaul_band, rsc):
    comp_time = dict()
    for id in ids:
        comp_time[id] = data_size[id] * epochs[id] * comp_per_bit[id] / rsc + model_size[id] / backhaul_band
    return comp_time


def eos(constant_params, range_params):
    bandwidth_budget, backhaul_bandwidth_budget, cpu_cycle_frequency = constant_params
    tasks_data_size, tasks_time_budget, tasks_computation_per_bit, tasks_epoch_number, tasks_model_size, \
        tasks_big_constant, tasks_privacy_score, tasks_offloading_state, tasks_ids = range_params

    model = pyo.ConcreteModel()

    model.i = pyo.Set(initialize=tasks_ids)

    # params
    model.D = pyo.Param(model.i, initialize=tasks_data_size)

    model.t = pyo.Param(model.i, initialize=tasks_time_budget)

    model.N_ic = pyo.Param(model.i, initialize=tasks_computation_per_bit)

    model.N_ie = pyo.Param(model.i, initialize=tasks_epoch_number)

    model.M = pyo.Param(model.i, initialize=tasks_model_size)

    model.big_constant = pyo.Param(model.i, initialize=tasks_big_constant)

    model.B = pyo.Param(model.i,
                        initialize=bandwidth_allocation(tasks_ids, bandwidth_budget))  # Bandwidth variables

    model.F = pyo.Param(model.i,
                        initialize=computation_allocation(tasks_ids, cpu_cycle_frequency))  # Frequency variables

    model.bB = pyo.Param(model.i, initialize=backhaul_bandwidth_allocation(tasks_ids,
                                                                           backhaul_bandwidth_budget))  # Backhaul Bandwidth variables

    model.alpha = pyo.Param(model.i, initialize=0)

    model.t_c = pyo.Param(model.i, initialize=completion_time(tasks_ids, tasks_data_size, tasks_epoch_number,
                                                              tasks_computation_per_bit, tasks_model_size,
                                                              backhaul_bandwidth_budget/len(tasks_ids),
                                                              cpu_cycle_frequency/len(tasks_ids)))

    model.D_o = pyo.Var(model.i, initialize=0)

    # create model instance
    instance = model.create_instance()

    return instance, 'ok'
