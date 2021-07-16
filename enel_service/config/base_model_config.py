class BaseModelConfig(object):
    scheduler: dict

    concurrency_limiter: dict

    early_stopping: dict

    score_function: dict

    reporter: dict

    tune_best_trial: dict

    stopper: dict

    tune_run: dict

    optuna_search: dict = {}

    # used for creating groups during train/val split in hp optimization
    essential_properties: list

    # default model setup
    model_setup: dict

    # transformer specs
    transformer_specs: list
