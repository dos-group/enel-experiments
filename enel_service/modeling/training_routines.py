import torch
from ignite.engine.engine import Engine

from enel_service.modeling.transforms import CustomData


def create_supervised_trainer(model, optimizer, loss_fn=None,
                              device=None, non_blocking=False,
                              batch_keys=None,
                              pre_augmentation_function=None,
                              post_augmentation_function=None,
                              output_transform=lambda x, y, y_pred, loss: loss.item()):
    """Adapted code from Ignite-Library in order to allow for handling of graphs."""

    if batch_keys is None:
        batch_keys = []

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()

        if device is not None:
            batch = batch.to(device, non_blocking=non_blocking)
        batch = CustomData.prepare_batch_object(batch).to(torch.double)

        if pre_augmentation_function is not None:
            batch = pre_augmentation_function(batch)

        res_dict = model(*[batch[k] for k in batch_keys])

        if post_augmentation_function is not None:
            res_dict, batch = post_augmentation_function(res_dict, batch)

        loss = loss_fn(res_dict, batch)

        loss.backward()

        optimizer.step()

        return output_transform(batch.x, batch, res_dict, loss)

    return Engine(_update)


def create_supervised_evaluator(model, metrics=None,
                                device=None, non_blocking=False,
                                batch_keys=None,
                                output_transform=lambda x, y, y_pred: (y_pred, y,)):
    """Adapted code from Ignite-Library in order to allow for handling of graphs."""

    if batch_keys is None:
        batch_keys = []
    metrics = metrics or {}

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            if device is not None:
                batch = batch.to(device, non_blocking=non_blocking)
            batch = CustomData.prepare_batch_object(batch).to(torch.double)

            res_dict = model(*[batch[k] for k in batch_keys])

            return output_transform(batch.x, batch, res_dict)

    engine_instance = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine_instance, name)

    return engine_instance
