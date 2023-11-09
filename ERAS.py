import pyomo.environ as pyo
from pyomo.opt import TerminationCondition


def rule_converted(model, i):
    return model.b1[i] + model.b2[i] <= 1


def rule_offloaded_data(model, i):
    return model.D_o[i] == model.alpha[i] * model.D[i]
    # return model.D_o[i] == model.b1[i] * model.alpha[i] * model.D[i] + model.b2[i] * model.D[i]


def time_budget1(model, i):
    return model.D[i] / model.B[i] + model.e[i] * (model.D[i] - model.D_o[i]) * model.Z[i] / model.F[i] + \
        (model.M[i] / model.bB[i]) <= model.t[i]


def time_budget2(model, i):
    return model.D[i] / model.B[i] + model.D_o[i] / model.bB[i] <= model.t[i]


def obj_expression(model):
    # return sum(model.alpha[i] * model.D[i] for i in model.i)
    # return sum(model.D_o[i]*model.P[i] for i in model.i)
    return sum(model.D_o[i] for i in model.i)


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


def eras(constant_params, range_params):
    bandwidth_budget, backhaul_bandwidth_budget, cpu_cycle_frequency = constant_params
    tasks_data_size, tasks_time_budget, tasks_computation_per_bit, tasks_epoch_number, tasks_model_size, \
        tasks_privacy_score, tasks_offloading_state, tasks_ids = range_params

    model = pyo.ConcreteModel()

    model.i = pyo.Set(initialize=tasks_ids)

    # params
    model.D = pyo.Param(model.i, initialize=tasks_data_size)

    model.t = pyo.Param(model.i, initialize=tasks_time_budget)

    model.Z = pyo.Param(model.i, initialize=tasks_computation_per_bit)

    model.e = pyo.Param(model.i, initialize=tasks_epoch_number)

    model.M = pyo.Param(model.i, initialize=tasks_model_size)

    model.B = pyo.Param(model.i,
                        initialize=bandwidth_allocation(tasks_ids, bandwidth_budget))  # Bandwidth variables

    model.F = pyo.Param(model.i,
                        initialize=computation_allocation(tasks_ids, cpu_cycle_frequency))  # Frequency variables

    model.bB = pyo.Param(model.i, initialize=backhaul_bandwidth_allocation(tasks_ids,
                                                                           backhaul_bandwidth_budget))  # Backhaul Bandwidth variables

    model.f = pyo.Param(model.i,
                        initialize={i: 1 / model.F[i] for i in tasks_ids})  # auxilary variable

    model.P = pyo.Param(model.i, initialize=tasks_privacy_score)

    #  variables
    model.alpha = pyo.Var(model.i, initialize=0, bounds=(0, 1))

    model.b1 = pyo.Var(model.i, initialize=0, domain=pyo.Binary)  # auxilary variable

    model.b2 = pyo.Var(model.i, initialize=0, domain=pyo.Binary)  # auxilary variable

    model.D_o = pyo.Var(model.i, initialize=0)

    # constraints
    model.Constraint5 = pyo.Constraint(model.i, rule=time_budget1)
    model.Constraint6 = pyo.Constraint(model.i, rule=time_budget2)
    model.Constraint3 = pyo.Constraint(model.i, rule=rule_converted)
    model.Constraint4 = pyo.Constraint(model.i, rule=rule_offloaded_data)

    # fixed vars
    for task_id, alpha_value in tasks_offloading_state.items():
        if alpha_value == "no_offloading":
            model.alpha[task_id].fixed = True
            model.b1[task_id].fixed = True
            model.b2[task_id].fixed = True
            model.alpha[task_id].value = 0.3  # this value doesn't matter
            model.b1[task_id].value = 0
            model.b2[task_id].value = 0  # b1=0 and b2=0 indicates no offloading

    # objective function
    model.OBJ = pyo.Objective(expr=obj_expression(model), sense=pyo.minimize)

    # model instance
    instance = model.create_instance()

    # call solver
    opt = pyo.SolverFactory('gurobi')
    opt.options['timelimit'] = 60
    opt.options['NonConvex'] = 2
    opt.options['InfProofCuts'] = 0

    try:
        results = opt.solve(instance, tee=True)
        # instance.display()

        if results.solver.termination_condition == TerminationCondition.infeasible:
            print("-----==------- INFEASIBLE --------")
            return instance, results.solver.termination_condition

        elif results.solver.termination_condition != TerminationCondition.optimal:
            print("non optimal")

        print(results.solver.termination_condition)
        print(results.solver.status)
        print("==========================================")
        return instance, results.solver.termination_condition

    except Exception as e:
        return instance, e
