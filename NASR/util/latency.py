
def get_time(prof, show_events=False):
    """
    return: list of cpu total time (unit: us)
    """
    traces = prof.traces
    trace_events = prof.trace_profile_events
    paths = prof.paths

    # self.logger.debug(traces)

    cpu_times = []

    for trace in traces:
        [path, leaf, module] = trace

        # self.logger.debug(trace)
        events = [te for t_events in trace_events[path] for te in t_events]
        for depth, name in enumerate(path, 1):
            if depth == len(path) and (
                    (paths is None and leaf) or (paths is not None and path in paths)
            ):
                if show_events:
                    for event in events:
                        cpu_times.append(event.cpu_time_total)
                else:
                    cpu_times.append(sum([e.cpu_time_total for e in events]))
    return cpu_times