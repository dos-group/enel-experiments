from ray import tune

from .base_model_config import BaseModelConfig
from ..common.db_schemes import MetricsModel

job_context_emb_columns: list = [
                    "application_signature",
                    "attempt_id",
                    "job_id"
                ]


class OnlinePredictorConfig(BaseModelConfig):
    scheduler: dict = {
        "grace_period": 30,
        "reduction_factor": 4
    }

    concurrency_limiter: dict = {
        "max_concurrent": 6
    }

    early_stopping: dict = {
        "patience": 50
    }

    reporter: dict = {
        "metric_columns": ["validation_loss", "main_time", "autoencoder", "stage_time", "stage_metrics"]
    }

    score_function: dict = {
        "relation": "lt",
        "key": "ft_loss",
        "threshold": -5.
    }

    tune_best_trial: dict = {
        "scope": "all",
        "filter_nan_and_inf": True
    }

    stopper: dict = {
        "metric": "validation_loss",
        "std": 0.01,
        "num_results": 5,
        "grace_period": 15,
    }

    tune_run: dict = {
        # at least two, because functional Training API will ALWAYS save checkpoint for last iteration
        "keep_checkpoints_num": 2,
        "verbose": 1,
        "num_samples": 40,
        "config": {
            "auto_encoder_dropout": tune.choice([0.05, 0.1, 0.2, 0.5]),
            "metric_dropout": tune.choice([0.05, 0.1, 0.2, 0.5]),
            "lr": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.loguniform(1e-5, 1e-1),
        }
    }

    optuna_search: dict = {}

    # used for creating groups during train/val split in hp optimization
    essential_properties: list = ["context_emb", "stage_start_scale_out_vec"]

    # default model setup
    model_setup: dict = {
        "follow_batch": [],
        "batch_keys": [
            "edge_index",
            "stage_start_scale_out_vec",
            "stage_end_scale_out_vec",
            "stage_rescaling_time_ratio",
            "context_emb",
            "context_emb_batch",
            "context_opt",
            "context_opt_batch",
            "stage_context",
            "stage_context_batch",
            "stage_metrics",
            "prev_job_batch",
            "real_nodes_batch",
            "num_nodes",
            "batch"
        ],
        "device": "cpu",
        "training_loss_args": {
            "device": "cpu"
        },
        "fine_tuning_loss_args": {
            "device": "cpu"
        },
        "optimizer_args": {
            "lr": 0.001,
            "weight_decay": 0.001
        },
        "model_args": {
            "hidden_dim": 16,
            "auto_encoder_dropout": 0.0,
            "metric_dropout": 0.0,
            "encoding_dim": 33,
            "stage_metrics_dim": len(MetricsModel.__fields__)
        },
        "epochs": [
            300,  # for pre-training
            100  # for fine-tuning
        ],
        "batch_size": 1
    }

    # transformer specs
    transformer_specs: list = [
        {
            "transformer_class": "ScaleOutEnricher"
        },
        {
            "transformer_class": "MinMaxScaler",
            "transformer_args": {
                "target_fields": ["job_start_scale_out_vec",
                                  "job_end_scale_out_vec",
                                  "stage_start_scale_out_vec",
                                  "stage_end_scale_out_vec"]
            }
        },
        {
            "transformer_class": "BinaryTransformer",
            "transformer_args": {
                "target_fields": ["global_specs.data_size_MB",
                                  "job_id",
                                  "stages.$.stage_id",
                                  "stages.$.attempt_id",
                                  "stages.$.num_tasks",
                                  "stages.$.rdd_num_partitions"],
                "out_dim": 33
            }
        },
        {
            "transformer_class": "HashingVectorizer",
            "transformer_args": {
                "target_fields": [
                    "application_signature",
                    "attempt_id",
                    "stages.$.stage_name",
                    "stages.$.failure_reason"
                ],
                "out_dim": 33,
                "lowercase": True,  # default
                "ngram_range": [1, 3],
                "analyzer": "char_wb",
                "norm": "l2",  # default
                "alternate_sign": True  # default
            }
        },
        {
            "transformer_class": "ContextEmbedder",
            "transformer_args": {
                "context_emb_columns": job_context_emb_columns,
                "context_opt_columns": [
                    "global_specs.data_size_MB"
                ]
            }
        },
        {
            "transformer_class": "StageGraphCreator"
        },
        {
            "transformer_class": "NeighborScanner"
        },
        {
            "transformer_class": "OnlinePredictorFinalizer"
        }
    ]
