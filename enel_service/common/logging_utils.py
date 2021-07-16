import logging
from importlib import reload


def init_logging(logging_level: str):
    """
    initialize logging setting
    :param logging_level:
    :return:
    """
    reload(logging)
    logging.basicConfig(
        level=getattr(logging, logging_level.upper()),
        format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s](%(name)s)__[L%(lineno)d] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)

    logging.info(f"Successfully initialized logging.")
