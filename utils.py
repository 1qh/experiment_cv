from contextlib import contextmanager
from time import perf_counter


@contextmanager
def howlong():
    s = perf_counter()
    try:
        yield
    finally:
        print(perf_counter() - s)
