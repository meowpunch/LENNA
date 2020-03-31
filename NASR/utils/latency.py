def get_time(target, prof, show_events=False):
    """
        motivated from `traces_to_display` function in torchprof module
    return: list of cpu total time (unit: us)
    """
    traces = prof.traces
    trace_events = prof.trace_profile_events
    paths = prof.paths

    # self.logger.debug(traces)

    cpu_times = []

    for trace in traces:
        [path, leaf, module] = trace

        if target in path:
            events = [te for t_events in trace_events[path] for te in t_events]
            for depth, name in enumerate(path, 1):
                if depth == len(path) and (
                        (paths is None and leaf) or (paths is not None and path in paths)
                ):
                    if show_events:
                        for event in events:
                            cpu_times.append(event.cpu_time_total)
                    else:
                        cpu_times.append(sum([e.self_cpu_time_total for e in events]))

                        for e in events:
                            print(e.cpu_children)
    return cpu_times


def get_df_times(prof, show_events=False):
    traces = prof.traces
    trace_events = prof.trace_profile_events
    paths = prof.paths

    # self.logger.debug(traces)

    cpu_times = []

    for trace in traces:
        [path, leaf, module] = trace

        events = [te for t_events in trace_events[path] for te in t_events]
        for depth, name in enumerate(path, 1):
            # print("name: ", name)
            if depth == len(path) and (
                    (paths is None and leaf) or (paths is not None and path in paths)
            ):
                if show_events:
                    for event in events:
                        cpu_times.append(event.cpu_time_total)
                else:
                    cpu_time = sum([e.cpu_time_total for e in events])
                    cpu_times.append(cpu_time)

                    # for e in events:
                    #     print(e.key)

    return cpu_times
