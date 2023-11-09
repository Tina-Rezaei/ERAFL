import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import TerminationCondition
import gurobipy as gp


def rule_bandwidth(model, bandwidth_budget):
    return pyo.summation(model.B) <= bandwidth_budget


def rule_bandwidth_auxilary(model, i):
    # return pyo.summation(model.B) >= n**2 / bandwidth_budget
    return model.b[i] * model.B[i] == 1


def rule_backhaul_bandwidth(model, backhaul_bandwidth_budget):
    return pyo.summation(model.bB) <= backhaul_bandwidth_budget


def rule_backhaul_bandwidth_auxilary(model, i):
    # return pyo.summation(model.bB) >= n**2 / backhaul_bandwidth_budget
    return model.bb[i] * model.bB[i] == 1


def rule_frequency(model, cpu_cycle_frequency):
    return pyo.summation(model.F) <= cpu_cycle_frequency


def rule_frequency_auxilary(model, i):
    # return pyo.summation(model.F) >= n**2 / cpu_cycle_frequency - 0.1
    return model.f[i] * model.F[i] == 1


def rule_converted1(model, i):
    return model.alpha[i] >= model.b1[i] * 0.3


def rule_converted2(model, i):
    return model.alpha[i] <= model.b1[i] * 0.7


def rule_converted3(model, i):
    return model.b1[i] + model.b2[i] <= 1


def rule_offloaded_data(model, i):
    return model.D_o[i] == model.b1[i] * model.alpha[i] * model.D[i] + model.b2[i] * model.D[i]
    # return model.D_o[i] == model.b1[i] * model.alpha[i] * model.D[i] + (1-model.b1[i]) * model.D[i]


def time_budget1(model, i):
    return model.D[i] * model.b[i] + model.e[i] * (model.D[i] - model.D_o[i]) * model.Z[i] * model.f[i] + \
        (model.M[i] * model.bb[i]) <= model.t[i]


def time_budget2(model, i):
    return model.D[i] * model.b[i] + (model.D_o[i]) * model.bb[i] <= model.t[i]


def time_budget3(model, i):
    return model.e[i] * (1.0 - model.b2[i] - model.alpha[i]) * model.D[i] * model.Z[i] * model.f[i] <= model.t[i]


def obj_expression(model):
    return sum(model.D_o[i]*model.P[i] for i in model.i)


def erafl(constant_params, range_params):
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

    model.P = pyo.Param(model.i, initialize=tasks_privacy_score)

    #  variables
    model.F = pyo.Var(model.i, bounds=(0.001, cpu_cycle_frequency))  # Frequency variables

    model.B = pyo.Var(model.i, bounds=(0.001, bandwidth_budget))  # Bandwidth variables

    model.bB = pyo.Var(model.i, bounds=(0.001, backhaul_bandwidth_budget))  # Backhaul Bandwidth variables

    model.alpha = pyo.Var(model.i, bounds=(0.3, 0.7))

    model.b1 = pyo.Var(model.i, domain=pyo.Binary)  # auxilary variable

    model.b2 = pyo.Var(model.i, domain=pyo.Binary)  # auxilary variable

    model.f = pyo.Var(model.i, bounds=(1 / cpu_cycle_frequency, 1 / 0.001))  # auxilary variable

    model.b = pyo.Var(model.i, bounds=(1 / bandwidth_budget, 1 / 0.001))  # auxilary variable for bandwidth

    model.bb = pyo.Var(model.i,
                       bounds=(1 / backhaul_bandwidth_budget, 1 / 0.001))  # auxilary variable for backhaul bandwidth

    model.D_o = pyo.Var(model.i, initialize=0)

    # fixed vars
    for task_id, alpha_value in tasks_offloading_state.items():
        if alpha_value == "no_offloading":
            model.alpha[task_id].fixed = True
            model.b1[task_id].fixed = True
            model.b2[task_id].fixed = True
            model.alpha[task_id].value = 0.3  # this value doesn't matter
            model.b1[task_id].value = 0
            model.b2[task_id].value = 0

    # constraints
    model.Constraint1 = pyo.Constraint(expr=rule_bandwidth(model, bandwidth_budget))
    model.Constraint2 = pyo.Constraint(expr=rule_backhaul_bandwidth(model, backhaul_bandwidth_budget))
    model.Constraint3 = pyo.Constraint(expr=rule_frequency(model, cpu_cycle_frequency))
    model.Constraint4 = pyo.Constraint(model.i, rule=rule_bandwidth_auxilary)
    model.Constraint5 = pyo.Constraint(model.i, rule=rule_backhaul_bandwidth_auxilary)
    model.Constraint6 = pyo.Constraint(model.i, rule=rule_frequency_auxilary)
    model.Constraint7 = pyo.Constraint(model.i, rule=rule_converted3)
    model.Constraint8 = pyo.Constraint(model.i, rule=time_budget1)
    model.Constraint9 = pyo.Constraint(model.i, rule=time_budget2)
    model.Constraint10 = pyo.Constraint(model.i, rule=rule_offloaded_data)

    # objective function
    model.OBJ = pyo.Objective(expr=obj_expression(model), sense=pyo.minimize)

    # model instance
    instance = model.create_instance()

    # call solver
    opt = pyo.SolverFactory('gurobi')
    opt.options['timelimit'] = 60
    opt.options['NonConvex'] = 2
    opt.options['Presolve'] = 0
    # opt.options['ScaleFlag'] = 2
    # opt.options['ObjScale'] = -1
    # opt.options['AggFill'] = 0
    # opt.options['Method'] = 3

    try:
        results = opt.solve(instance, tee=True)
        # instance.display()

        if results.solver.termination_condition == TerminationCondition.infeasible:
            print("------- INFEASIBLE --------")
            return instance, results.solver.termination_condition

        elif results.solver.termination_condition != TerminationCondition.optimal:
            print("non_optimal")

        print(results.solver.termination_condition)
        print(results.solver.status)
        print("==========================================")
        return instance, results.solver.termination_condition

    except Exception as e:
        return instance, e

