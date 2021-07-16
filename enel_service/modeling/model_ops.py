import logging
import torch
from typing import Optional, Tuple
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter, Linear, Sequential, SELU
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Size, PairTensor
from torch_geometric.utils import softmax


class BasePredictor(nn.Module):

    def disable_autoencoder(self):
        logging.info("Disable auto-encoder...")
        # freeze all auto-encoder layers
        for name, param in self.named_parameters():
            if "auto_encoder" in name:
                logging.info(f"Disable gradient of '{name}' Parameter...")
                param.requires_grad = False
        # set dropout-value to zero
        for name, module in self.named_modules():
            if "auto_encoder" in name:
                if isinstance(module, (nn.Dropout, nn.AlphaDropout)):
                    logging.info(f"Unset dropout value of '{name}' ({module}) Module...")
                    module.p = 0.0

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config_dict(self):
        return {k: str(v) for k, v in self.__dict__.items() if isinstance(v, (str, float, int, list, tuple))}


def init_weights(m):
    """He Initialization."""
    if type(m) == nn.Linear:
        # weights are using lecun-normal initialization
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='selu')
        # biases zero
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class OverheadConv(BasePredictor):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        logging.info(f"Using custom class: {self.__class__.__name__}")

        self.transform = Sequential(
            Linear(in_channels, int(in_channels / 2)),
            SELU(),
            Linear(int(in_channels / 2), out_channels),
            SELU()
        )
        self.transform.apply(init_weights)

    def forward(self,
                stage_start_scale_out_vec: Tensor,
                stage_end_scale_out_vec: Tensor,
                stage_rescaling_time_ratio: Tensor,
                context: Tensor,
                stage_metrics: Tensor,
                real_nodes_batch: Tensor) -> Tensor:
        # learn overhead based on context and metrics
        pred_overhead: Tensor = self.transform(torch.cat([context,
                                                          stage_metrics,
                                                          stage_start_scale_out_vec,
                                                          stage_end_scale_out_vec,
                                                          stage_rescaling_time_ratio], dim=-1))
        # we only want predictions for actual nodes
        pred_overhead[~real_nodes_batch, :] = 0
        # we only want predictions for actual stages with actual rescalings
        pred_overhead[torch.all(torch.eq(stage_start_scale_out_vec, stage_end_scale_out_vec), dim=1), :] = 0
        return pred_overhead

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels})"


class RuntimeConv(BasePredictor, MessagePassing):
    propagate_type = {'x': Tensor}

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        kwargs.update({'aggr': 'max'})  # ensure we use a max-aggregation
        super(RuntimeConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        logging.info(f"Using custom class: {self.__class__.__name__}")

        self.transform = Sequential(
            Linear(in_channels, int(in_channels / 2)),
            SELU(),
            Linear(int(in_channels / 2), out_channels),
            SELU()
        )
        self.transform.apply(init_weights)

    def forward(self,
                edge_index: Tensor,
                stage_end_scale_out_vec: Tensor,
                context: Tensor,
                stage_metrics: Tensor,
                real_nodes_batch: Tensor,
                overhead: Tensor,
                total_time_cumsum: Tensor) -> Tuple[Tensor, Tensor]:
        # learn time based on static context and metrics
        pred_runtime: Tensor = self.transform(torch.cat([context,
                                                         stage_metrics,
                                                         stage_end_scale_out_vec,
                                                         overhead], dim=-1))
        # we only want predictions for actual nodes
        pred_runtime[~real_nodes_batch, :] = 0
        # Start propagating messages
        total_time_cumsum: Tensor = self.propagate(edge_index,
                                                   x=pred_runtime + total_time_cumsum,
                                                   size=None)
        return pred_runtime, total_time_cumsum

    def message(self, x_j: Tensor) -> Tensor:
        # x_j has shape [E, out_channels]
        return x_j

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels})"


class MetricConv(BasePredictor, MessagePassing):
    propagate_type = {'metrics': PairTensor, 'context': PairTensor}

    def __init__(self, context_channels: int, stage_metrics_channels: int, out_channels: int, metric_dropout: float,
                 **kwargs):
        kwargs.update({'aggr': 'add'})  # ensure we use an add-aggregation
        super(MetricConv, self).__init__(**kwargs)

        logging.info(f"Using custom class: {self.__class__.__name__}")

        self.context_channels = context_channels
        self.stage_metrics_channels = stage_metrics_channels
        self.out_channels = out_channels
        self.metric_dropout = metric_dropout

        self.lin_l = Linear(self.context_channels, self.context_channels)
        self.lin_r = Linear(self.context_channels, self.context_channels)

        final_transform_dim = self.context_channels + self.stage_metrics_channels
        self.final_transform = Sequential(
            Linear(final_transform_dim, int(final_transform_dim / 2)),
            SELU(),
            Linear(int(final_transform_dim / 2), self.out_channels),
            SELU()
        )
        self.final_transform.apply(init_weights)

        self.att = Parameter(torch.Tensor(1, self.context_channels))

        self.bias = Parameter(torch.Tensor(1, self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        init_weights(self.lin_l)
        init_weights(self.lin_r)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, edge_index: Tensor,
                stage_start_scale_out_vec: Tensor,
                stage_end_scale_out_vec: Tensor,
                context: Tensor,
                stage_metrics: Tensor,
                size: Size = None) -> Tensor:

        new_context: Tensor = torch.cat([stage_start_scale_out_vec, context, stage_end_scale_out_vec], dim=-1)

        new_context_l = self.lin_l(new_context)
        new_context_r = self.lin_r(new_context)

        # Start propagating messages
        out = self.propagate(edge_index,
                             context=(new_context_l, new_context_r),
                             metrics=(stage_metrics, stage_metrics),
                             size=size)

        # if out row is all-zero (parent nodes sent only zeros)
        overwrite_indices = torch.all(torch.eq(out, torch.zeros_like(out)), dim=1)
        out[overwrite_indices, :] = stage_metrics[overwrite_indices, :]

        if self.bias is not None:
            out[~overwrite_indices, :] += self.bias
        out[~overwrite_indices, :] = torch.sigmoid(out[~overwrite_indices, :])
        return out

    def message(self,
                context_i: Tensor,
                context_j: Tensor,
                metrics_j: Tensor,
                index: Tensor,
                size_i: Optional[int]) -> Tensor:

        context = context_i + context_j
        context = F.selu(context)
        # if metrics_j row is all-zero (nodes will send only zeros as metrics)
        context[torch.all(torch.eq(metrics_j, torch.zeros_like(metrics_j)), dim=1), :] = 0

        alpha = (context * self.att).sum(dim=-1)
        alpha = F.dropout(alpha, p=self.metric_dropout, training=self.training)

        zero_indices = torch.all(torch.eq(alpha.view(-1, 1), torch.zeros_like(alpha).view(-1, 1)), dim=1)
        alpha[~zero_indices] = softmax(alpha[~zero_indices], index[~zero_indices], num_nodes=size_i)
        # learn next metrics based on context and metrics
        final_x_j: Tensor = self.final_transform(torch.cat([context, metrics_j], dim=-1))
        final_x_j[zero_indices, :] = 0
        return final_x_j * alpha.view(-1, 1)

    def __repr__(self):
        return f"{self.__class__.__name__}(context_channels={self.context_channels}, " \
               f"stage_metrics_channels={self.stage_metrics_channels}, " \
               f"out_channels={self.out_channels}, metric_dropout={self.metric_dropout})"
