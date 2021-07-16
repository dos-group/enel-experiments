from contextlib import redirect_stderr
from io import StringIO
from unittest import TestCase

import torch
from torch.jit import RecursiveScriptModule
from torch_geometric.data import Batch

from enel_service.config.onlinepredictor_config import OnlinePredictorConfig
from enel_service.modeling.model_wrappers import OnlinePredictorModel
from enel_service.modeling.models import OnlinePredictor
from enel_service.modeling.transforms import CustomData


class TestOnlinePredictorModel(TestCase):

    def setUp(self) -> None:
        self.config = OnlinePredictorConfig()
        self.config.model_setup["epochs"] = [25, 25]  # speed up fine-tuning
        self.config.model_setup["batch_size"] = 1
        self.encoding_dim = self.config.model_setup["model_args"]["encoding_dim"]
        self.wrapper = OnlinePredictorModel(self.config)

        self.basic_data = {
            "edge_index": torch.tensor([[0, 1, 1, 2, 3, 4],
                                        [1, 2, 3, 4, 4, 5]], dtype=torch.long),
            "stage_start_scale_out_vec": torch.rand(6, 3),
            "stage_end_scale_out_vec": torch.rand(6, 3),
            "stage_rescaling_time_ratio": torch.rand(6, 1),
            "context_emb": torch.rand(30, self.encoding_dim),
            "context_emb_batch": torch.tensor(sum([[idx] * 5 for idx in range(6)], []), dtype=torch.long),
            "context_opt": torch.rand(30, self.encoding_dim),
            "context_opt_batch": torch.tensor(sum([[idx] * 5 for idx in range(6)], []), dtype=torch.long),
            "stage_context": torch.rand(24, self.encoding_dim),
            "stage_context_batch": torch.tensor(sum([[idx] * 4 for idx in range(6)], []), dtype=torch.long),
            "stage_metrics": torch.rand(6, 5),
            "prev_job_batch": torch.tensor([-1], dtype=torch.long),
            "real_nodes_batch": torch.tensor(list(range(2, 6)), dtype=torch.long),
            "num_nodes": 6,
            "job_rescaling_time_ratio": torch.rand(1, 1),
        }

        self.basic_target = {
            "stage_runtime": torch.rand(6, 1),
            "stage_metrics": torch.rand(6, 5),
            "job_runtime": torch.rand(1, 1),
            "context_emb": torch.rand(30, self.encoding_dim),
            "context_opt": torch.rand(30, self.encoding_dim),
            "stage_context": torch.rand(24, self.encoding_dim),
            "real_nodes_batch": torch.tensor(list(range(2, 6)), dtype=torch.long),
            "num_nodes": 6
        }

    def test_output_transform_function(self):
        pred_dict = {"stage_runtime_pred": torch.rand(6, 1),
                     "stage_metrics_pred": torch.rand(6, 15),
                     "job_runtime_pred": torch.rand(1, 1),
                     "context_emb_dec": torch.rand(30, self.encoding_dim),
                     "context_opt_dec": torch.rand(30, self.encoding_dim),
                     "stage_context_dec": torch.rand(24, self.encoding_dim)}

        batch = Batch.from_data_list([CustomData(**self.basic_target)])

        result = self.wrapper.output_transform_function((pred_dict, batch))
        self.assertTrue(isinstance(result, dict))
        self.assertEqual(len(result), 8)  # 2x stage time, 2x stage metric, 2x job time, 2x autoencoder
        for element in result.values():
            self.assertTrue(isinstance(element, torch.Tensor))

    def test_get_shallow_model_instance(self):
        result = self.wrapper.get_shallow_model_instance()
        self.assertTrue(isinstance(result, OnlinePredictor))
        print(result)
        print(result.get_num_parameters())
        print(result.get_num_trainable_parameters())
        print(result.auto_encoder.get_num_parameters())
        print(result.overhead_conv.get_num_parameters())
        print(result.runtime_conv.get_num_parameters())
        print(result.metric_conv.get_num_parameters())

    def test_get_torchscript_model_instance(self):
        result = self.wrapper.get_torchscript_model_instance()
        self.assertTrue(isinstance(result, RecursiveScriptModule))

    def test_predict(self):
        data_object = CustomData(**{**self.basic_data, **self.basic_target})
        data_list = [data_object, data_object, data_object]
        model = self.wrapper.get_torchscript_model_instance()

        result = self.wrapper.predict(model, data_list)
        self.assertEqual(len(result), len(data_list))
        for element in result:
            self.assertTrue(isinstance(element, dict))
            self.assertEqual(len(element), 12)  # 9x from autoencoder, 3x from online-predictor
            for k, v in element.items():
                self.assertTrue(isinstance(v, torch.Tensor))

    def test_fit(self):
        data_object = CustomData(**{**self.basic_data, **self.basic_target})
        data_list = [data_object, data_object, data_object]

        model = self.wrapper.get_torchscript_model_instance()
        result = self.wrapper.fit(self.wrapper.get_torchscript_model_instance(), data_list, {})

        # test that both state dicts are different
        with redirect_stderr(StringIO()) as _:
            with self.assertRaises(ValueError) as _:
                for p1, p2 in zip(model.parameters(), result.parameters()):
                    if p1.data.ne(p2.data).sum() > 0:
                        raise ValueError()
