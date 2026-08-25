import time
from datetime import timedelta


class ExpensiveComputation:
    """
    Provides a context manager that prints (to stdout) at the start and end of the context, along with the time spent inside the context in seconds.
    """
    depth = 0
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        print("\t"*self.depth + f"Starting {self.name}...")
        type(self).depth += 1

    def __exit__(self, exc_type=None, exc_info=None, exc_tb=None) -> None:
        self.end = time.time()
        self.interval = self.end - self.start
        if exc_type is not None:
            print("\t"*self.depth + f"Errored {self.name}. Took {self.interval:.4f} seconds")
        else:
            print("\t"*self.depth + f"Finished {self.name}. Took {self.interval:.4f} seconds")
        type(self).depth -= 1

class MaybeExpensiveComputation:
    """
    Provides a context manager that calculates the time spent inside the context in seconds, and prints it to stdout if it exceeds a given threshold.
    """
    interval: float

    def __init__(self, name: str, threshold: timedelta = timedelta(seconds=1)):
        self.name, self.threshold = name, threshold

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.interval > self.threshold.total_seconds():
            print(f"{self.name} took {self.interval:.4f} seconds")