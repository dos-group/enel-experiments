import asyncio
import functools
from asyncio import Task


def _to_task(future, loop):
    if isinstance(future, Task):
        return future
    return loop.create_task(future)


def force_sync_wrapper(fn):
    """
    turn an async function to sync function
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        return force_sync(res)
    return wrapper


def force_sync(func):
    if asyncio.iscoroutine(func):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        return loop.run_until_complete(_to_task(func, loop))
    else:
        return func


def async_return(result):
    f = asyncio.Future()
    f.set_result(result)
    return f
