import time
from functools import wraps


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_taken = end_time - start_time
        hours, remainder = divmod(time_taken, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            f"{func.__name__} completed in {int(hours)}h {int(minutes)}m {int(seconds)}s"
        )
        return result

    return wrapper
