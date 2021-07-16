import logging
from typing import Optional

import torch
from ignite.engine import Engine
from torch import nn
from torch_geometric.data import Batch


class Scorer(object):
    def __init__(self, trainer: Engine, **kwargs):
        """
        Parameters
        ----------
        trainer : Engine
            The training-engine
        """

        # define fields and their types
        self.key: Optional[str] = None
        self.threshold: Optional[float] = None
        self.relation: Optional[str] = None

        # use kwargs to set fields
        kwargs.setdefault('relation', 'lt')
        kwargs.setdefault('key', 'ft_loss')
        kwargs.setdefault('threshold', 0.)
        self.__dict__.update(kwargs)

        self.trainer = trainer

        logging.info(f"Scorer-Configuration: {self.__dict__}")

    def __call__(self, engine: Engine):
        """Determines an improvement during training."""

        operator = min if self.relation == "lt" else max

        result = operator(self.threshold, -engine.state.metrics[self.key])

        if engine.state.epoch % 20 == 0:
            logging.info(f"[Epoch={str(engine.state.epoch).zfill(3)}] Fine-Tuning Loss: {result}")

        if result == self.threshold:
            logging.info(f"EarlyStopping: Stop training after {str(engine.state.epoch).zfill(3)} epochs!")
            self.trainer.terminate()

        return result


# #####################################################
class OnlinePredictorTrainingLoss(object):
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

        self.predictor_loss = nn.SmoothL1Loss()
        self.vector_loss = nn.MSELoss()

    def __call__(self, pred_dict: dict, batch: Batch):
        """Computes the loss given a batch-object and a result-dict from the model."""

        times_pred = torch.cat([pred_dict["stage_runtime_pred"],
                                pred_dict["job_runtime_pred"]], dim=0)
        times_true = torch.cat([batch["stage_runtime"],
                                batch["job_runtime"]], dim=0)
        times_indices = torch.max(times_true, dim=-1)[0].reshape(-1) > 0

        indices = batch["real_nodes_batch"].to(torch.long)
        stage_metric_pred = pred_dict["stage_metrics_pred"][indices, :]
        stage_metric_true = batch["stage_metrics"][indices, :]
        stage_metric_indices = torch.max(stage_metric_true, dim=-1)[0].reshape(-1) > 0

        autoencoder_pred = torch.cat([pred_dict["context_emb_dec"],
                                      pred_dict["context_opt_dec"],
                                      pred_dict.get("stage_context_dec", torch.tensor([]))], dim=0)
        autoencoder_true = torch.cat([batch["context_emb"],
                                      batch["context_opt"],
                                      getattr(batch, "stage_context")
                                      if hasattr(batch, "stage_context") else torch.tensor([])], dim=0)
        autoencoder_indices = torch.max(autoencoder_true, dim=-1)[0].reshape(-1) > 0

        return self.predictor_loss(times_pred[times_indices, :], times_true[times_indices, :]) + \
               self.vector_loss(stage_metric_pred[stage_metric_indices, :], stage_metric_true[stage_metric_indices, :]) + \
               self.vector_loss(autoencoder_pred[autoencoder_indices, :], autoencoder_true[autoencoder_indices, :])


class OnlinePredictorFineTuningLoss(object):
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

        self.predictor_loss = nn.SmoothL1Loss()

    def __call__(self, pred_dict: dict, batch: Batch):
        """Computes the loss given a batch-object and a result-dict from the model."""

        times_pred = torch.cat([pred_dict["stage_runtime_pred"],
                                pred_dict["job_runtime_pred"]], dim=0)
        times_true = torch.cat([batch["stage_runtime"],
                                batch["job_runtime"]], dim=0)
        times_indices = torch.max(times_true, dim=-1)[0].reshape(-1) > 0

        return self.predictor_loss(times_pred[times_indices, :], times_true[times_indices, :])


class OnlinePredictorObservationLoss(object):
    def __init__(self, *args, **kwargs):
        self.loss = nn.L1Loss()

    def __call__(self, pred_dict, batch):
        """Computes the loss given a batch-object and a result-dict from the model."""
        times_pred = torch.cat([pred_dict["stage_runtime_pred"],
                                pred_dict["job_runtime_pred"]], dim=0)
        times_true = torch.cat([batch["stage_runtime"],
                                batch["job_runtime"]], dim=0)
        times_indices = torch.max(times_true, dim=-1)[0].reshape(-1) > 0

        return self.loss(times_pred[times_indices, :], times_true[times_indices, :])
