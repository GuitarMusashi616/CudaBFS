import time

class Stopwatch:
    def __init__(self):
        self.start = time.perf_counter()

    def get_time(self) -> float:
        end = time.perf_counter()
        diff = end - self.start
        return diff
        