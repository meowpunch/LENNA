def get_time(prof, target, show_events=False):
    """
        motivated from `traces_to_display` function in torchprof module
    return: list of self cpu time total (unit: us) for target
    """
    traces = prof.traces
    trace_events = prof.trace_profile_events
    paths = prof.paths

    cpu_times = []

    for trace in traces:
        [path, leaf, module] = trace

        # search all latency in target
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

                        # for e in events:
                        #     print(e.cpu_children)
    return cpu_times

