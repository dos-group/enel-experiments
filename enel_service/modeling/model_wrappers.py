import copy
import datetime
import math
from collections import OrderedDict
from typing import List, Mapping, Tuple, Any, Union, Iterator

import torch.utils.data
from ignite.handlers.checkpoint import BaseSaveHandler
from torch.jit import RecursiveScriptModule
from torch.optim import Adam
import numpy as np
from torch_geometric.data import DataLoader, Data
from ignite.handlers import Checkpoint, global_step_from_engine, EarlyStopping
from ignite.engine import Events
from ignite.metrics import Loss

from .datasets import ExecutionDataset, ExecutionSubset
from .training_routines import create_supervised_trainer
from .transforms import CustomData
from .losses import *
from .models import OnlinePredictor
from enel_service.config.base_model_config import BaseModelConfig


def timeit(f):
    def timed(self, *args, **kwargs):
        start = datetime.datetime.now()
        result = f(self, *args, **kwargs)
        duration = (datetime.datetime.now() - start).total_seconds()

        if f.__name__ == "fit":
            self.__fit_time__ = duration

        elif f.__name__ == "predict":
            self.__predict_time__ = duration

        return result

    return timed


class LocalSaveHandler(BaseSaveHandler):
    def __init__(self):
        self.checkpoints = {}
        self.__last_checkpoint__ = None

    @property
    def last_checkpoint(self):
        return self.__last_checkpoint__

    def __call__(self, checkpoint: Mapping, filename: str, **kwargs):
        self.checkpoints[filename] = checkpoint
        self.__last_checkpoint__ = checkpoint

    def remove(self, filename: str):
        try:
            del self.checkpoints[filename]
        except BaseException as exc:
            logging.debug("Could not remove checkpoint", exc_info=exc)
            pass


def update_flat_dicts(root: dict, targets: list):
    my_root = copy.deepcopy(root)
    my_targets = [copy.deepcopy(target) for target in targets]

    for k, v in my_root.items():
        for target in my_targets:
            if k in target:
                target[k] = v

    return my_targets


class BaseModel(object):
    def __init__(self):
        # define fields and their types
        self.model_class: Optional[OnlinePredictor] = None

        self.model_args: dict = {}
        self.optimizer_args: dict = {}
        self.training_loss_args: dict = {}
        self.fine_tuning_loss_args: dict = {}

        self.follow_batch: Optional[list] = None
        self.batch_keys: Optional[list] = None
        self.batch_size: Optional[int] = None
        self.device: Optional[str] = None

        self.__fit_time__: Optional[float] = None
        self.__predict_time__: Optional[float] = None

    @property
    def fit_time(self):
        return_value: float = 0.0
        if self.__fit_time__ is not None:
            return_value = self.__fit_time__
            self.__fit_time__ = None
        return return_value

    @property
    def predict_time(self):
        return_value: float = 0.0
        if self.__predict_time__ is not None:
            return_value = self.__predict_time__
            self.__predict_time__ = None
        return return_value

    @property
    def config_dict(self):
        """Configuration dictionary."""
        return {k: str(v) for k, v in self.__dict__.items() if isinstance(v, (str, float, int, list, tuple))}

    def get_shallow_model_instance(self) -> OnlinePredictor:
        return self.model_class(**self.model_args).to(self.device).to(torch.double)

    def incorporate_checkpoint(self, checkpoint: dict):
        config_dict = copy.deepcopy(self.config_dict)

        # load from checkpoint
        if len(checkpoint) and "best_trial_config" in checkpoint:
            best_trial_config = checkpoint['best_trial_config']
            config_dict = {**config_dict, **best_trial_config, "device": self.device}

        # extract and override
        self.model_args, self.optimizer_args = update_flat_dicts(config_dict, [self.model_args,
                                                                               self.optimizer_args])

    def check_device(self, model: RecursiveScriptModule, checkpoint: dict):
        if not torch.cuda.is_available():
            self.device = "cpu"
            # extract and override
            for att in ["model_args", "optimizer_args", "training_loss_args", "fine_tuning_loss_args"]:
                if hasattr(self, att) and isinstance(getattr(self, att), dict):
                    setattr(self, att, update_flat_dicts({"device": "cpu"}, [getattr(self, att)])[0])
            return self.get_torchscript_model_instance(checkpoint, disable_autoencoder=True)
        return model

    def get_torchscript_model_instance(self, checkpoint: dict = None,
                                       log_meta_information: bool = False,
                                       disable_autoencoder: bool = False):
        if checkpoint is None:
            checkpoint = {}
        shallow_model_instance: OnlinePredictor = self.get_shallow_model_instance()
        if len(checkpoint) and "model_state_dict" in checkpoint:
            shallow_model_instance.load_state_dict(checkpoint["model_state_dict"])
            shallow_model_instance = shallow_model_instance.to(torch.double)
        if disable_autoencoder:
            shallow_model_instance.disable_autoencoder()
        if log_meta_information:
            logging.info(f"Model: {shallow_model_instance}")
            logging.info(f"#Parameters: {shallow_model_instance.get_num_parameters()}")
            logging.info(f"Trainable #parameters: {shallow_model_instance.get_num_trainable_parameters()}")
        return torch.jit.script(shallow_model_instance)

    @timeit
    def fit(self, *args, **kwargs):
        return self._fit(*args, **kwargs)

    def _fit(self, *args, **kwargs):
        raise NotImplementedError

    @timeit
    def predict(self, *args, **kwargs):
        return self._predict(*args, **kwargs)

    def _predict(self, model: RecursiveScriptModule, data_list: List[Union[CustomData, Batch]]):
        model = model.to(self.device)
        model.eval()
        # predict
        result_list = []
        with torch.no_grad():
            iterable: Iterator[Batch] = iter(DataLoader(data_list,
                                                        batch_size=self.batch_size, follow_batch=self.follow_batch,
                                                        shuffle=False))
            for b in iterable:
                b: Batch = CustomData.prepare_batch_object(b).to(self.device).to(torch.double)

                call_args: List[Any] = [b[k] for k in self.batch_keys]
                res_dict: dict = model(*call_args)
                result_list += [res_dict]

        return result_list


# ONLINE PREDICTOR #
class OnlinePredictorModel(BaseModel):
    def __init__(self, config: BaseModelConfig):
        """
        Parameters
        ----------
        config: BaseModelConfig
            A configuration class with configurations for this model
        """
        super().__init__()

        # define fields and their types
        self.follow_batch: Optional[list] = None
        self.batch_keys: Optional[list] = None
        self.batch_size: Optional[int] = None
        self.epochs: Optional[tuple] = None
        self.device: Optional[str] = None
        self.config: BaseModelConfig = config

        # use kwargs to set fields (maybe not all)
        self.__dict__.update(self.config.model_setup)

        self.model_class = OnlinePredictor
        self.optimizer_class = Adam
        self.training_loss_class = OnlinePredictorTrainingLoss
        self.fine_tuning_loss_class = OnlinePredictorFineTuningLoss
        self.observation_loss = OnlinePredictorObservationLoss

        self.model_args = {**self.__dict__.get("model_args", {}),
                           "device": self.device}

        self.optimizer_args = self.__dict__.get("optimizer_args", {})
        self.training_loss_args = {**self.__dict__.get("training_loss_args", {}), "device": self.device}
        self.fine_tuning_loss_args = {**self.__dict__.get("fine_tuning_loss_args", {}), "device": self.device}

    def split_data(self, dataset: ExecutionDataset) -> Tuple[ExecutionSubset, ExecutionSubset]:

        unique_executions: List[Tuple[str, float]] = []
        for d in dataset:
            if not len(set(d["application_execution_id"])) == 1:
                raise ValueError("Jobs were not stacked correctly!")
            unique_executions.append((
                d["application_execution_id"][0],
                # use mean scale-out of first job as scale-out indicator
                torch.mean(torch.cat([
                    d["job_start_scale_out_vec"][:1, 0],
                    d["job_end_scale_out_vec"][:1, 0]
                ], dim=-1)).item()
            ))
        # sort by average scale out
        unique_executions = sorted(unique_executions, key=lambda key: key[1])
        unique_execution_ids: List[str] = [el[0] for el in unique_executions]

        list_val_indices = np.round(np.linspace(1, len(unique_execution_ids) - 2,
                                                1 + math.ceil(len(unique_execution_ids) * 0.2))).astype(int)
        logging.info(f"Validation indices from list: {list_val_indices}")

        val_ids = [unique_execution_ids[idx] for
                   idx in list_val_indices if idx < len(unique_execution_ids)]
        train_ids = [u_id for u_id in unique_execution_ids if u_id not in val_ids]

        train_set: ExecutionSubset = ExecutionSubset.from_parent(dataset, train_ids, suffix="training")
        val_set: ExecutionSubset = ExecutionSubset.from_parent(train_set, val_ids, suffix="inference")

        return train_set, val_set

    @classmethod
    def output_transform_function(cls, output: Tuple[dict, Batch], *args, **kwargs):
        pred_dict, batch = output

        response_dict: OrderedDict = OrderedDict()
        response_dict["stage_times_pred"] = pred_dict["stage_runtime_pred"]
        response_dict["stage_times_true"] = batch["stage_runtime"]

        indices = batch["real_nodes_batch"].to(torch.long)
        response_dict["stage_metrics_pred"] = pred_dict["stage_metrics_pred"][indices, :]
        response_dict["stage_metrics_true"] = batch["stage_metrics"][indices, :]

        response_dict["autoencoder_pred"] = torch.cat([pred_dict["context_emb_dec"],
                                                       pred_dict["context_opt_dec"],
                                                       pred_dict.get("stage_context_dec", torch.tensor([]))], dim=0)
        response_dict["autoencoder_true"] = torch.cat([batch["context_emb"],
                                                       batch["context_opt"],
                                                       getattr(batch, "stage_context")
                                                       if hasattr(batch, "stage_context") else torch.tensor([])], dim=0)

        response_dict["times_pred"] = pred_dict["job_runtime_pred"]
        response_dict["times_true"] = batch["job_runtime"]

        return response_dict

    @staticmethod
    def add_noise_to_tensor(original: torch.Tensor, noise_func_args: List[Any], calculate_absolute: bool = True):
        exclude_indices = torch.all(torch.eq(original, torch.zeros_like(original)), dim=1)

        noise = torch.normal(*noise_func_args, size=original.shape).to(original.device).to(original.dtype)
        original[~exclude_indices, :] += noise[~exclude_indices, :]
        if calculate_absolute:
            original[~exclude_indices, :] = torch.abs(original[~exclude_indices, :])

        # final check: did we by accident make a row zero-only due to our transformation?
        new_exclude_indices = torch.all(torch.eq(original, torch.zeros_like(original)), dim=1)
        adapt_indices = torch.logical_xor(exclude_indices, new_exclude_indices)
        # if so, add small number to make them non-zero again
        original[adapt_indices, :] += 0.001

        return original

    @classmethod
    def pre_augmentation_function(cls, batch: Batch):
        # add some noise to target values
        batch.job_runtime = OnlinePredictorModel.add_noise_to_tensor(batch.job_runtime, [0, 1.5])

        batch.stage_runtime = OnlinePredictorModel.add_noise_to_tensor(batch.stage_runtime, [0, 0.5])

        batch.job_rescaling_time_ratio = OnlinePredictorModel.add_noise_to_tensor(batch.job_rescaling_time_ratio,
                                                                                  [0, 0.025])

        batch.stage_rescaling_time_ratio = OnlinePredictorModel.add_noise_to_tensor(batch.stage_rescaling_time_ratio,
                                                                                    [0, 0.025])

        batch.stage_metrics = OnlinePredictorModel.add_noise_to_tensor(batch.stage_metrics, [0, 0.025])

        for key in ["context_emb", "context_opt", "stage_context"]:
            if hasattr(batch, key):
                batch[key] = OnlinePredictorModel.add_noise_to_tensor(batch[key], [0, 0.025], calculate_absolute=False)

        return batch

    @classmethod
    def post_augmentation_function(cls, pred_dict: dict, batch: Batch):
        return pred_dict, batch

    def __resume_from_checkpoint__(self, model: RecursiveScriptModule, checkpoint: dict):
        """Get a model, optimizer and loss, optionally with pretrained states."""

        config_dict = copy.deepcopy(self.config_dict)

        # load from checkpoint
        if len(checkpoint) and "best_trial_config" in checkpoint:
            best_trial_config = checkpoint['best_trial_config']
            config_dict = {**config_dict, **best_trial_config, "device": self.device}

        # pre-training completed, if it was necessary
        logging.info("Using config: {}".format(config_dict))

        # extract and override
        model_args, optimizer_args, fine_tuning_loss_args = update_flat_dicts(config_dict,
                                                                              [self.model_args,
                                                                               self.optimizer_args,
                                                                               self.fine_tuning_loss_args])

        model = model.to(torch.double).to(self.device)

        logging.info(f"#Parameters: {sum(p.numel() for p in model.parameters())}")
        logging.info(f"Trainable #parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        # init optimizer and loss
        optimizer = self.optimizer_class(model.parameters(), **optimizer_args)
        loss = self.fine_tuning_loss_class(**fine_tuning_loss_args)

        suffix = "PreTrained-"
        logging.info(f"{suffix}Model: {self.model_class}, {suffix}Args={model_args}")
        logging.info(f"{suffix}Optimizer: {self.optimizer_class}, {suffix}Args={optimizer_args}")
        logging.info(f"{suffix}Loss: {self.fine_tuning_loss_class}, {suffix}Args={fine_tuning_loss_args}")

        return model, optimizer, loss

    def _fit(self, model: RecursiveScriptModule, data_list: List[Data], checkpoint: dict):
        """Fit / Fine-tune a model."""

        # prepare model
        model, optimizer, loss = self.__resume_from_checkpoint__(model, checkpoint)

        if len(data_list):
            # setup ignite trainer
            trainer = create_supervised_trainer(
                model,
                optimizer,
                loss_fn=loss,
                device=self.device,
                non_blocking=True,
                batch_keys=self.batch_keys,
                pre_augmentation_function=self.pre_augmentation_function,
                post_augmentation_function=self.post_augmentation_function,
                output_transform=lambda x, y, y_pred, l: (y_pred, y)
            )

            to_save: dict = {
                "model_state_dict": model,
                "optimizer_state_dict": optimizer,
                "trainer_state_dict": trainer
            }

            # scorer function to determine training improvements
            score_function = Scorer(trainer, **self.config.score_function)
            Loss(self.observation_loss()).attach(trainer, "ft_loss")

            # setup ignite checkpoint handler
            save_handler = LocalSaveHandler()

            checkpoint_handler = Checkpoint(
                to_save,
                save_handler,
                score_function=score_function,
                global_step_transform=global_step_from_engine(trainer),
                n_saved=1)

            # configure early stopping to counter early stopping

            stopping_handler = EarlyStopping(score_function=score_function,
                                             trainer=trainer,
                                             patience=self.config.early_stopping.get("patience", 100))

            trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)
            trainer.add_event_handler(Events.EPOCH_COMPLETED, stopping_handler)

            epochs = self.epochs[1]

            # data loader for loading batches of training data
            train_loader = DataLoader(data_list,
                                      shuffle=True,
                                      batch_size=self.batch_size,
                                      follow_batch=self.follow_batch)

            # start training
            trainer.run(train_loader, max_epochs=epochs)

            # after fine-tuning, load best model state
            local_checkpoint = save_handler.last_checkpoint
            model.load_state_dict(local_checkpoint['model_state_dict'], strict=True)

        return model
