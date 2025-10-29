import torch


class Timer:
    """
    Simple timer which takes forces cuda synchronization.
    """

    def __init__(self):
        self.timers_start = []

    def start(self):
        start_t = torch.cuda.Event(enable_timing=True)
        start_t.record()
        self.timers_start.append(start_t)

    def stop(self, tag=None):
        end_t = torch.cuda.Event(enable_timing=True)
        end_t.record()
        torch.cuda.synchronize()
        start_t = self.timers_start.pop()
        tag = f"{tag}: " if tag else ""
        elapsed_time_s = start_t.elapsed_time(end_t) / 1000
        print(f"{tag}Elapsed {elapsed_time_s}s")
        return elapsed_time_s


_global_timer = Timer()
tic = _global_timer.start
toc = _global_timer.stop
