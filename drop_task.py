

def drop_task(range_params, instance):
    comp = list()

    # calculate the computational need of each task
    for i in (instance.i):
        comp.append(instance.D[i] * instance.Z[i] * instance.e[i])

    # index of task with the highest computational need
    index = comp.index(max(comp))

    # find id of the task with the highest computational need
    task_id = range_params[-1][index]
    print(f"drop task index : {task_id}")

    # remove all parameters values of the task with the highest computational need
    for param in range_params[0:-1]:
        param.pop(task_id)
        # param.pop(task_index)
        # new_params.append({i+1:v for i, v in enumerate(param.values())})

    # remove id of the task with the highest computational need
    range_params[-1].remove(task_id)

    return range_params
