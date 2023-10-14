import pyomo.environ as pyo
from pyomo.opt import TerminationCondition


def rule_bandwidth(model, bandwidth_budget):
    return pyo.summation(model.B) <= bandwidth_budget


def rule_bandwidth_auxiliary(model, i):
    return model.b[i] * model.B[i] == 1


def rule_backhaul_bandwidth(model, backhaul_bandwidth_budget):
    return pyo.summation(model.bB) <= backhaul_bandwidth_budget


def rule_backhaul_bandwidth_auxiliary(model, i):
    return model.bb[i] * model.bB[i] == 1


def rule_frequency(model, cpu_cycle_frequency):
    return pyo.summation(model.F) <= cpu_cycle_frequency


def rule_frequency_auxiliary(model, i):
    return model.f[i] * model.F[i] == 1


def rule_offloaded_data(model, i):
    return model.D_o[i] == model.alpha[i] * model.D[i]


def time_budget1(model, i):
    return model.D[i] * model.b[i] + model.N_ie[i] * (model.D[i] - model.D_o[i]) * model.N_ic[i] * model.f[i] + \
        (model.M[i] * model.bb[i]) <= model.t[i]


def time_budget2(model, i):
    return model.D[i] * model.b[i] + (model.D_o[i]) * model.bb[i] <= model.t[i]


def time_budget3(model, i):
    return model.N_ie[i] * (model.D[i] - model.D_o[i]) * model.N_ic[i] * model.f[i] <= model.t[i]


def obj_expression(model):
    return sum(model.D_o[i]*model.P[i] for i in model.i)


def bos_model(constant_params, range_params):
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

    model.P = pyo.Param(model.i, initialize=tasks_privacy_score)

    #  variables
    model.F = pyo.Var(model.i, bounds=(1, cpu_cycle_frequency))  # Frequency variables

    model.B = pyo.Var(model.i, bounds=(1, bandwidth_budget))  # Bandwidth variables

    model.bB = pyo.Var(model.i, bounds=(1, backhaul_bandwidth_budget))  # Backhaul Bandwidth variables

    model.alpha = pyo.Var(model.i, initialize=1, domain=pyo.Binary)

    model.b = pyo.Var(model.i,
                      bounds=(1 / bandwidth_budget, 1))  # auxiliary variable for bandwidth

    model.bb = pyo.Var(model.i,
                       bounds=(1 / backhaul_bandwidth_budget, 1))  # auxiliary variable for backhaul bandwidth

    model.f = pyo.Var(model.i, bounds=(1 / cpu_cycle_frequency, 1))  # auxiliary variable

    model.y = pyo.Var(model.i, initialize=0, domain=pyo.Binary)  # auxiliary variable for max function

    model.t_c = pyo.Var(model.i, initialize=0)

    model.D_o = pyo.Var(model.i, initialize=0)

    # constraints
    model.Constraint1 = pyo.Constraint(expr=rule_bandwidth(model, bandwidth_budget))
    model.Constraint2 = pyo.Constraint(expr=rule_backhaul_bandwidth(model, backhaul_bandwidth_budget))
    model.Constraint3 = pyo.Constraint(expr=rule_frequency(model, cpu_cycle_frequency))
    model.Constraint4 = pyo.Constraint(model.i, rule=rule_bandwidth_auxiliary)
    model.Constraint5 = pyo.Constraint(model.i, rule=rule_backhaul_bandwidth_auxiliary)
    model.Constraint6 = pyo.Constraint(model.i, rule=rule_frequency_auxiliary)
    model.Constraint7 = pyo.Constraint(model.i, rule=rule_offloaded_data)
    model.Constraint8 = pyo.Constraint(model.i, rule=time_budget1)
    model.Constraint9 = pyo.Constraint(model.i, rule=time_budget2)
    model.Constraint10 = pyo.Constraint(model.i, rule=time_budget3)

    # fixed vars
    for task_id, alpha_value in tasks_offloading_state.items():
        if alpha_value == "no_offloading":
            model.alpha[task_id].fixed = True
            model.alpha[task_id].value = 0  # this value doesn't matter

    # objective function
    model.OBJ = pyo.Objective(expr=obj_expression(model), sense=pyo.minimize)

    # model instance
    instance = model.create_instance()

    # call solver
    opt = pyo.SolverFactory('gurobi')
    opt.options['timelimit'] = 60  # set timeout
    opt.options['NonConvex'] = 2
    opt.options['InfProofCuts'] = 0

    try:
        results = opt.solve(instance, tee=True)
        # instance.printQality()
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
