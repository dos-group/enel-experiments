import os
from typing import Optional, Tuple

import torch
import logging
import time
from functools import partial
import torch.nn as nn
from ray import tune
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter, ExperimentAnalysis
from ignite.engine import Events, Engine
from ignite.metrics import Loss, MeanAbsoluteError, MeanSquaredError
from torch_geometric.data import DataLoader
from datetime import datetime

from .datasets import ExecutionSubset
from .model_wrappers import update_flat_dicts
from .training_routines import create_supervised_trainer, create_supervised_evaluator

# load general settings
from enel_service.common.configuration import GeneralSettings
from .transforms import CustomCompose
from ..config.base_model_config import BaseModelConfig

general_settings: GeneralSettings = GeneralSettings.get_instance()


class HyperOptimizer(object):
    def __init__(self, *args, **kwargs):

        # define fields and their types
        self.epochs: Optional[tuple] = None
        self.output_transform_function: Optional[callable] = None
        self.pre_augmentation_function: Optional[callable] = None
        self.post_augmentation_function: Optional[callable] = None
        self.batch_keys: Optional[list] = None
        self.follow_batch: Optional[list] = None
        self.job_identifier: Optional[str] = None
        self.device: Optional[str] = None
        self.batch_size: Optional[int] = None
        self.training_loss_args: Optional[dict] = None
        self.training_loss_class: Optional[callable] = None
        self.optimizer_args: Optional[dict] = None
        self.optimizer_class: Optional[callable] = None
        self.model_args: Optional[dict] = None
        self.model_class: Optional[callable] = None

        # use kwargs to set fields (maybe not all)
        self.__dict__.update(kwargs)

        logging.info(
            f"Default-Model: {self.model_class}, Default-Args={self.model_args}")
        logging.info(
            f"Default-Optimizer: {self.optimizer_class}, Default-Args={self.optimizer_args}")
        logging.info(
            f"Default-Loss: {self.training_loss_class}, Default-Args={self.training_loss_args}")
        logging.info(
            f"Default-Config: Device={self.device}, BatchSize={self.batch_size}")

    @staticmethod
    def perform_optimization(hyperoptimizer_instance,
                             epochs: int,
                             subsets: Tuple[ExecutionSubset, ExecutionSubset],
                             config: BaseModelConfig) -> dict:
        """Perform hyperparameter optimization.

        Parameters
        ----------
        hyperoptimizer_instance : HyperOptimizer
            An instance to use for hyperparameter optimization
        epochs: int
            Max. numbers of epoch to train
        subsets: Tuple[ExecutionSubset, ExecutionSubset]
            A Tuple, containing a subset for training and validation
        config: BaseModelConfig
            configurations for tune and the respective model in general

        Returns
        ----------
        dict, CustomCompose
            A comprehensive dict with training results, and the fitted data transformer
        """

        # Extract optionally provided configurations #####
        scheduler_config: dict = config.scheduler
        optuna_search_config: dict = config.optuna_search
        concurrency_limiter_config: dict = config.concurrency_limiter
        tune_run_config: dict = config.tune_run
        stopper_config: dict = config.stopper
        tune_best_trial_config: dict = config.tune_best_trial
        reporter_config: dict = config.reporter

        ##################################################
        search_space_config: dict = tune_run_config.get("config", {})

        scheduler = ASHAScheduler(max_t=epochs, **scheduler_config)

        reporter = CLIReporter(parameter_columns=list(search_space_config.keys()),
                               **reporter_config)

        stopper = TrialPlateauStopper(**stopper_config)

        search_alg = OptunaSearch(**optuna_search_config)
        search_alg = ConcurrencyLimiter(
            search_alg, **concurrency_limiter_config)

        tune_run_name = f"{hyperoptimizer_instance.job_identifier}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        start = time.time()

        train_set, val_set = subsets

        logging.info(f"Number of training samples: {len(train_set.sub_ids)}")
        logging.info(f"Number of validation samples: {len(val_set.sub_ids)}")

        analysis: ExperimentAnalysis = tune.run(tune.with_parameters(partial(hyperoptimizer_instance,
                                                                             epochs=epochs),
                                                                     train_set=train_set,
                                                                     val_set=val_set),
                                                name=tune_run_name,
                                                scheduler=scheduler,
                                                progress_reporter=reporter,
                                                search_alg=search_alg,
                                                local_dir=os.path.join(general_settings.temp_dir, "ray_results"),
                                                stop=stopper,
                                                checkpoint_score_attr="min-validation_loss",
                                                mode="min",
                                                metric="validation_loss",
                                                **tune_run_config)

        time_taken = time.time() - start

        # get best trial
        best_trial = analysis.get_best_trial(**tune_best_trial_config)

        # get some information from best trial
        best_trial_val_loss = best_trial.metric_analysis["validation_loss"]["min"]

        logging.info(f"Time taken: {time_taken:.2f} seconds.")
        logging.info("Best trial config: {}".format(best_trial.config))
        logging.info("Best trial final validation loss: {}".format(
            best_trial_val_loss))

        # load best checkpoint of best trial
        best_checkpoint_dir = analysis.get_best_checkpoint(best_trial)

        model_state_dict, optimizer_state_dict, trainer_state_dict, evaluator_state_dict = torch.load(
            os.path.join(best_checkpoint_dir, "checkpoint"))

        return {
                   "model_state_dict": model_state_dict,
                   "optimizer_state_dict": optimizer_state_dict,
                   "trainer_state_dict": trainer_state_dict,
                   "evaluator_state_dict": evaluator_state_dict,
                   "best_trial_config": best_trial.config,
                   "best_trial_id": best_trial.trial_id,
                   "best_trial_val_loss": best_trial_val_loss,
                   "time_taken": time_taken,
                   "trial_dataframes": analysis.trial_dataframes
               }

    def __call__(self, config,
                 checkpoint_dir: str = None,
                 epochs: int = None,
                 train_set: ExecutionSubset = None,
                 val_set: ExecutionSubset = None):
        """Called by 'tune.run' during hyperparameter optimization.
        Check: https://docs.ray.io/en/releases-1.1.0/tune/api_docs/trainable.html

        Parameters
        ----------
        config : dict
            A new setup to test
        checkpoint_dir: str
            Path to temporary folder for checkpoints
        epochs: int
            Max. numbers of epoch to train
        train_set: list
            A list of Data-objects for training (default: None)
        val_set: list
            A list of Data-objects for validation (default: None)
        """

        # extract and override #####
        batch_size = self.batch_size
        model_args, optimizer_args, training_loss_args = update_flat_dicts(
            config, [self.model_args, self.optimizer_args, self.training_loss_args])
        ############################

        model = self.model_class(**model_args).double()
        optimizer = self.optimizer_class(
            filter(lambda p: p.requires_grad, model.parameters()), **optimizer_args)
        loss = self.training_loss_class(**training_loss_args)

        # create trainer #####
        trainer = create_supervised_trainer(model,
                                            optimizer,
                                            loss_fn=loss,
                                            device=self.device,
                                            non_blocking=True,
                                            pre_augmentation_function=self.pre_augmentation_function,
                                            post_augmentation_function=self.post_augmentation_function,
                                            batch_keys=self.batch_keys
                                            )

        ######################

        # create evaluator #####
        def extract_func(obj, x, keys):
            def map_key(key: str):
                return "y_pred" if "_pred" in key else "y"

            filtered_dict: dict = {k: v for k, v in obj.output_transform_function(x).items() if k in keys}
            renamed_dict: dict = {map_key(k): v for k, v in filtered_dict.items()}
            if "autoencoder_true" not in keys:
                indices = torch.max(renamed_dict["y"], dim=-1)[0].reshape(-1) > 0
                renamed_dict["y"] = renamed_dict["y"][indices, :]
                renamed_dict["y_pred"] = renamed_dict["y_pred"][indices, :]
            return renamed_dict

        val_metrics: dict = {
            "validation_loss": Loss(loss),
            "main_time": MeanAbsoluteError(output_transform=lambda x: extract_func(self, x, ["times_pred",
                                                                                             "times_true"])),
            "autoencoder": MeanSquaredError(output_transform=lambda x: extract_func(self, x, ["autoencoder_pred",
                                                                                              "autoencoder_true"])),
            "stage_time": MeanAbsoluteError(output_transform=lambda x: extract_func(self, x, ["stage_times_pred",
                                                                                              "stage_times_true"])),
            "stage_metrics": MeanSquaredError(output_transform=lambda x: extract_func(self, x, ["stage_metrics_pred",
                                                                                                "stage_metrics_true"])),
        }

        if self.__class__.__name__ == "OfflinePredictor":
            del val_metrics["stage_time"]
            del val_metrics["stage_metrics"]

        evaluator = create_supervised_evaluator(model,
                                                device=self.device,
                                                metrics=val_metrics,
                                                batch_keys=self.batch_keys)
        ########################

        # restore if possible #####
        if checkpoint_dir:
            checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            trainer.load_state_dict(checkpoint["trainer_state_dict"])
            evaluator.load_state_dict(checkpoint["evaluator_state_dict"])
        ###########################

        if self.device != "cpu" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)

        # create data loaders #####
        train_loader = DataLoader(train_set,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  follow_batch=self.follow_batch)

        val_loader = DataLoader(val_set,
                                shuffle=False,
                                batch_size=batch_size,
                                follow_batch=self.follow_batch)

        ###########################

        @trainer.on(Events.EPOCH_COMPLETED)
        def post_epoch_actions(trainer_instance: Engine):

            # evaluate model on validation set
            evaluator.run(val_loader)
            state_val_metrics = evaluator.state.metrics

            current_epoch: int = trainer_instance.state.epoch

            with tune.checkpoint_dir(current_epoch) as local_checkpoint_dir:
                model_state_dict = model.state_dict()
                if isinstance(model, nn.DataParallel):
                    model_state_dict = model.module.state_dict()

                # save model, optimizer and trainer checkpoints
                path = os.path.join(local_checkpoint_dir, "checkpoint")
                torch.save((model_state_dict, optimizer.state_dict(),
                            trainer_instance.state_dict(), evaluator.state_dict()), path)

            # report validation scores to ray-tune
            report_dict: dict = {
                **state_val_metrics,
                "done": current_epoch == epochs
            }

            tune.report(**report_dict)

        # start training
        trainer.run(train_loader, max_epochs=epochs)
