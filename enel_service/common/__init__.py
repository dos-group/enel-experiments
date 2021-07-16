import os
import random
import string


def create_dirs(path: str):
    """Creates a directory, recursively if needed.

    Parameters
    ----------
    path : str
        A path that needs to be created
    """

    try:
        os.makedirs(path)
    except:
        pass


def id_generator(size=10, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
