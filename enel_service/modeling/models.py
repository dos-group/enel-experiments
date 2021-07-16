from typing import Optional, List, Dict

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool

from enel_service.modeling.model_ops import RuntimeConv, OverheadConv, MetricConv, init_weights, BasePredictor


# AUTO ENCODER #
class AutoEncoder(BasePredictor):
    def __init__(self, *args, **kwargs):
        super(AutoEncoder, self).__init__()

        self.__dict__.update(kwargs)

        self.downscale_hidden_dim: int = int(self.hidden_dim / 2)

        # encoder
        self.encoder = nn.Sequential(nn.Linear(self.encoding_dim, self.hidden_dim, bias=False),
                                     nn.SELU(),
                                     nn.AlphaDropout(p=self.auto_encoder_dropout),
                                     nn.Linear(self.hidden_dim, self.downscale_hidden_dim, bias=False),
                                     nn.SELU())
        self.encoder.apply(init_weights)

        # decoder
        self.decoder = nn.Sequential(nn.Linear(self.downscale_hidden_dim, self.hidden_dim, bias=False),
                                     nn.SELU(),
                                     nn.AlphaDropout(p=self.auto_encoder_dropout),
                                     nn.Linear(self.hidden_dim, self.encoding_dim, bias=False),
                                     nn.Tanh())
        self.decoder.apply(init_weights)

    def forward(self,
                context_emb: torch.Tensor,
                context_emb_batch: torch.Tensor,
                context_opt: torch.Tensor,
                context_opt_batch: torch.Tensor,
                stage_context: Optional[torch.Tensor],
                stage_context_batch: Optional[torch.Tensor]):

        context_emb_batch = context_emb_batch.to(torch.long)
        context_opt_batch = context_opt_batch.to(torch.long)
        if stage_context_batch is not None:
            stage_context_batch = stage_context_batch.to(torch.long)

        result_dict: Dict[str, torch.Tensor] = {}

        # ### #
        # compute required embeddings
        context_emb_codes_mean = torch.zeros(len(torch.unique(context_emb_batch)),
                                             self.downscale_hidden_dim,
                                             device=context_emb_batch.device,
                                             dtype=torch.double)
        if context_emb.numel() > 0:
            context_emb_enc = self.encoder(context_emb)
            result_dict["context_emb_codes"] = context_emb_enc.detach().clone()
            result_dict["context_emb_dec"] = self.decoder(context_emb_enc)
            context_emb_codes_mean = global_mean_pool(context_emb_enc, context_emb_batch)
        result_dict["context_emb_codes_mean"] = context_emb_codes_mean

        # ### #
        # compute optional embeddings
        context_opt_codes_mean = torch.zeros(len(torch.unique(context_opt_batch)),
                                             self.downscale_hidden_dim,
                                             device=context_opt_batch.device,
                                             dtype=torch.double)
        if context_opt.numel() > 0:
            context_opt_enc = self.encoder(context_opt)
            result_dict["context_opt_codes"] = context_opt_enc.detach().clone()
            result_dict["context_opt_dec"] = self.decoder(context_opt_enc)
            context_opt_codes_mean = global_mean_pool(context_opt_enc, context_opt_batch)
        result_dict["context_opt_codes_mean"] = context_opt_codes_mean

        # ### #
        # compute stage context embeddings
        stage_context_codes_mean: Optional[torch.Tensor] = None
        if stage_context_batch is not None:
            stage_context_codes_mean = torch.zeros(len(torch.unique(stage_context_batch)),
                                                   self.downscale_hidden_dim,
                                                   device=stage_context_batch.device,
                                                   dtype=torch.double)
        if stage_context is not None and stage_context.numel() > 0 and stage_context_batch is not None:
            stage_context_enc = self.encoder(stage_context)
            result_dict["stage_context_codes"] = stage_context_enc.detach().clone()
            result_dict["stage_context_dec"] = self.decoder(stage_context_enc)
            stage_context_codes_mean = global_mean_pool(stage_context_enc, stage_context_batch)
        if stage_context_codes_mean is not None:
            result_dict["stage_context_codes_mean"] = stage_context_codes_mean

        return result_dict


# ONLINE PREDICTOR #
class OnlinePredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super(OnlinePredictor, self).__init__()

        self.__dict__.update(kwargs)

        self.downscale_hidden_dim: int = int(self.hidden_dim / 2)
        self.upscale_hidden_dim: int = int(self.hidden_dim * 2)

        # auto-encoder
        self.auto_encoder = AutoEncoder(**kwargs)

        # dimensionality of concatened context vectors
        self.context_dim = int(3 * self.downscale_hidden_dim)

        # we use this conv layer to predict the overhead. we ignore the propagated predictions
        self.overhead_conv_in_dim: int = self.context_dim + self.stage_metrics_dim + (2 * 3) + 1
        self.overhead_conv: OverheadConv = OverheadConv(self.overhead_conv_in_dim, 1)
        # we use this conv layer to predict the runtime and propagate the predictions through the graph
        self.runtime_conv_in_dim: int = self.context_dim + self.stage_metrics_dim + 3 + 1
        self.runtime_conv: RuntimeConv = RuntimeConv(self.runtime_conv_in_dim, 1).jittable()

        # we use this conv layer to predict the metrics of next stage based on predecessor stages and their metrics
        self.metric_conv_in_dim: int = self.context_dim + (2 * 3)
        self.metric_conv: MetricConv = MetricConv(self.metric_conv_in_dim, self.stage_metrics_dim,
                                                  self.stage_metrics_dim, self.metric_dropout).jittable()

    @staticmethod
    def handle_metrics_update(true_metrics: torch.Tensor,
                              pred_metrics: torch.Tensor,
                              prev_job_batch: torch.Tensor,
                              real_nodes_batch: torch.Tensor,
                              batch: torch.Tensor):
        # non-zero predictions
        non_zero_indices: torch.Tensor = torch.max(pred_metrics[real_nodes_batch, :], dim=-1)[0]
        # recompute avg nodes
        avg_indices = torch.logical_and(non_zero_indices.reshape(-1).detach().clone() > 0,
                                        real_nodes_batch.reshape(-1).detach().clone())
        # compute metrics for summary nodes (for successor nodes)
        mean_pred_metrics: torch.Tensor = global_mean_pool(pred_metrics[real_nodes_batch[avg_indices], :],
                                                           batch[real_nodes_batch[avg_indices]])

        new_pred_tensor: torch.Tensor = torch.zeros_like(pred_metrics)
        # use original pred values
        new_pred_tensor[:] = pred_metrics[:]
        # use accumulated pred values
        if len(prev_job_batch) > 1:
            new_pred_tensor[prev_job_batch[1:], :] = mean_pred_metrics[:-1, :]

        new_true_tensor: torch.Tensor = torch.zeros_like(true_metrics)
        indices: torch.Tensor = torch.max(true_metrics, dim=-1)[0].reshape(-1) > 0
        # use original true values
        new_true_tensor[:] = true_metrics[:]
        # take predicted values
        new_true_tensor[~indices, :] = new_pred_tensor[~indices, :]

        return new_true_tensor, new_pred_tensor

    def forward(self,
                edge_index: torch.Tensor,
                stage_start_scale_out_vec: torch.Tensor,
                stage_end_scale_out_vec: torch.Tensor,
                stage_rescaling_time_ratio: torch.Tensor,
                context_emb: torch.Tensor,
                context_emb_batch: torch.Tensor,
                context_opt: torch.Tensor,
                context_opt_batch: torch.Tensor,
                stage_context: torch.Tensor,
                stage_context_batch: torch.Tensor,
                stage_metrics: torch.Tensor,
                prev_job_batch: torch.Tensor,
                real_nodes_batch: torch.Tensor,
                num_nodes: int,
                batch: torch.Tensor):

        edge_index = edge_index.to(torch.long)
        context_emb_batch = context_emb_batch.to(torch.long)
        context_opt_batch = context_opt_batch.to(torch.long)
        prev_job_batch = prev_job_batch.to(torch.long)
        real_nodes_batch = real_nodes_batch.to(torch.long)
        if stage_context_batch is not None:
            stage_context_batch = stage_context_batch.to(torch.long)
        batch = batch.to(torch.long)

        # during training, all metrics should be available. Thus, we dont need to do metric computation later
        # first two rows can be ignored as they are possibly invalid predecessor nodes of possibly root nodes
        early_skip: bool = self.training and (0 == torch.sum(torch.all(torch.eq(stage_metrics[2:, :],
                                                                                torch.zeros_like(stage_metrics[2:, :])),
                                                                       dim=1)).item())

        if self.training and not early_skip:
            print(stage_metrics)
            raise ValueError("This is unexpected and may never happen!")

        # compute embeddings
        response_dict: Dict[str, torch.Tensor] = self.auto_encoder(context_emb,
                                                                   context_emb_batch,
                                                                   context_opt,
                                                                   context_opt_batch,
                                                                   stage_context,
                                                                   stage_context_batch)
        # extract embeddings
        job_context_emb_codes: torch.Tensor = response_dict["context_emb_codes_mean"]
        job_context_opt_codes: torch.Tensor = response_dict["context_opt_codes_mean"]
        stage_context_codes: torch.Tensor = response_dict["stage_context_codes_mean"]

        # prepare context
        context: torch.Tensor = torch.cat([job_context_emb_codes,
                                           job_context_opt_codes,
                                           stage_context_codes], dim=-1)

        # init with dummy values
        stage_runtime_pred: torch.Tensor = torch.zeros(stage_metrics.size()[0], 1,
                                                       dtype=torch.double,
                                                       device=stage_metrics.device)
        stage_metrics_pred: torch.Tensor = torch.zeros_like(stage_metrics)

        stage_total_time_prop_cumsum: torch.Tensor = torch.zeros(stage_metrics.size()[0],
                                                                 1,
                                                                 dtype=stage_metrics.dtype,
                                                                 device=stage_metrics.device)

        # start propagating
        helper_list: List[None] = [None] * num_nodes
        while len(helper_list):
            helper_list.pop(0)
            # predict rescaling overhead for stage, if any
            stage_overhead_pred = self.overhead_conv(stage_start_scale_out_vec,
                                                     stage_end_scale_out_vec,
                                                     stage_rescaling_time_ratio,
                                                     context,
                                                     stage_metrics,
                                                     real_nodes_batch)
            # predict runtime for stage, and retrieve cumsum total-time value for stage
            stage_runtime_pred, stage_total_time_prop_cumsum = self.runtime_conv(edge_index,
                                                                                 stage_end_scale_out_vec,
                                                                                 context,
                                                                                 stage_metrics,
                                                                                 real_nodes_batch,
                                                                                 stage_overhead_pred,
                                                                                 stage_total_time_prop_cumsum)

            if not early_skip or torch.sum(stage_metrics_pred) == 0:
                # predict metrics based on metrics from precedessors
                stage_metrics_pred = self.metric_conv(edge_index,
                                                      stage_start_scale_out_vec,
                                                      stage_end_scale_out_vec,
                                                      context,
                                                      stage_metrics)

            if len(helper_list) and not early_skip:
                # compute 'prev-job-metrics' for successor job
                stage_metrics, stage_metrics_pred = OnlinePredictor.handle_metrics_update(stage_metrics,
                                                                                          stage_metrics_pred,
                                                                                          prev_job_batch,
                                                                                          real_nodes_batch, batch)
            # if each message was propagated to last stage of each job, we can stop.
            # However, we need to still add initial runtime (+ overhead) predictions to the cumsum values
            if not len(helper_list):
                stage_total_time_prop_cumsum += stage_runtime_pred

        # Now, we need to just select the maximum value for each job, which should intuitively be the job runtime,
        # i.e. the cumulative sum of overhead + runtime for all stages will be our job runtime.
        job_runtime_pred: Optional[torch.Tensor] = None
        if stage_total_time_prop_cumsum is not None:
            job_runtime_pred = global_max_pool(stage_total_time_prop_cumsum, batch)

        updates: Dict[str, torch.Tensor] = {
            "stage_runtime_pred": stage_runtime_pred,
            "stage_metrics_pred": stage_metrics_pred,
            "job_runtime_pred": job_runtime_pred
        }
        response_dict.update(updates)

        return response_dict
