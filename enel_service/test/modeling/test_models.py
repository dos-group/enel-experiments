from unittest import TestCase

import torch

from enel_service.modeling.models import AutoEncoder, OnlinePredictor


class TestAutoEncoder(TestCase):

    def setUp(self) -> None:
        self.context_emb = torch.rand(5, 40)
        self.context_emb_batch = torch.ones(5)
        self.context_opt = torch.rand(5, 40)
        self.context_opt_batch = torch.ones(5)
        self.stage_context = torch.rand(5, 40)
        self.stage_context_batch = torch.ones(5)
        self.model_args = {"encoding_dim": 40, "hidden_dim": 8, "auto_encoder_dropout": 0.5, "device": "cpu"}
        self.model = AutoEncoder(**self.model_args)
        self.loss = torch.nn.MSELoss()

    def test_model_torch_script(self):
        try:
            model = torch.jit.script(self.model)
            print(model)
            print(self.model.get_num_parameters())
            print(self.model.get_num_trainable_parameters())
        except BaseException as exc:
            self.fail(f"Model can not be transformed to torch-script, error: '{exc}'")

    def test_model_forward(self):
        # first: with "normal model"
        result: dict = self.model(self.context_emb,
                                  self.context_emb_batch,
                                  self.context_opt,
                                  self.context_opt_batch,
                                  self.stage_context,
                                  self.stage_context_batch)
        for k, v in result.items():
            self.assertTrue(isinstance(v, torch.Tensor))
            if "codes" == k[-5:]:
                self.assertEqual(v.size(), (self.context_emb.size()[0], int(self.model_args["hidden_dim"] / 2)))
            elif "dec" == k[-3:]:
                self.assertEqual(v.size(), self.context_emb.size())
            self.assertEqual(v.isnan().sum(), 0)
            self.assertEqual(v.isinf().sum(), 0)

        # now: with torchscript model
        model = torch.jit.script(self.model)
        result: dict = model(self.context_emb,
                             self.context_emb_batch,
                             self.context_opt,
                             self.context_opt_batch,
                             self.stage_context,
                             self.stage_context_batch)
        for k, v in result.items():
            self.assertTrue(isinstance(v, torch.Tensor))
            if "codes" == k[-5:]:
                self.assertEqual(v.size(), (self.context_emb.size()[0], int(self.model_args["hidden_dim"] / 2)))
            elif "dec" == k[-3:]:
                self.assertEqual(v.size(), self.context_emb.size())
            self.assertEqual(v.isnan().sum(), 0)
            self.assertEqual(v.isinf().sum(), 0)

    def test_model_backward(self):
        try:
            result: dict = self.model(self.context_emb,
                                      self.context_emb_batch,
                                      self.context_opt,
                                      self.context_opt_batch,
                                      self.stage_context,
                                      self.stage_context_batch)
            for k, v in result.items():
                self.assertTrue(isinstance(v, torch.Tensor))
                self.assertEqual(v.isnan().sum(), 0)
                self.assertEqual(v.isinf().sum(), 0)
                if "codes" == k[-5:]:
                    self.assertEqual(v.size(), (self.context_emb.size()[0], int(self.model_args["hidden_dim"] / 2)))
                elif "dec" == k[-3:]:
                    self.assertEqual(v.size(), self.context_emb.size())
                    lookup_key = "_".join(k.split("_")[:-1])
                    loss = self.loss(v, getattr(self, lookup_key))
                    loss.backward(retain_graph=True)
            # now: with torchscript model
            model = torch.jit.script(self.model)
            result: dict = model(self.context_emb,
                                 self.context_emb_batch,
                                 self.context_opt,
                                 self.context_opt_batch,
                                 self.stage_context,
                                 self.stage_context_batch)
            for k, v in result.items():
                self.assertTrue(isinstance(v, torch.Tensor))
                self.assertEqual(v.isnan().sum(), 0)
                self.assertEqual(v.isinf().sum(), 0)
                if "codes" == k[-5:]:
                    self.assertEqual(v.size(), (self.context_emb.size()[0], int(self.model_args["hidden_dim"] / 2)))
                elif "dec" == k[-3:]:
                    self.assertEqual(v.size(), self.context_emb.size())
                    lookup_key = "_".join(k.split("_")[:-1])
                    loss = self.loss(v, getattr(self, lookup_key))
                    loss.backward(retain_graph=True)
        except BaseException as exc:
            self.fail(f"Model output produces error during backpropagation, error: '{exc}'")


class TestOnlinePredictor(TestCase):

    def setUp(self) -> None:
        self.edge_index = torch.tensor([[0, 1, 2, 3, 4, 4, 5, 6, 8, 9, 10, 11, 12, 12, 13, 14],
                                        [2, 2, 3, 4, 5, 6, 7, 7, 10, 10, 11, 12, 13, 14, 15, 15]], dtype=torch.long)
        self.stage_start_scale_out_vec = torch.cat([torch.zeros(2, 3), torch.rand(14, 3)], dim=0)
        self.stage_end_scale_out_vec = torch.cat([torch.zeros(2, 3), torch.rand(14, 3)], dim=0)
        self.stage_rescaling_time_ratio = torch.cat([torch.zeros(2, 1), torch.rand(14, 1)], dim=0)
        self.job_context_emb = torch.rand(80, 40)
        self.job_context_emb_batch = torch.tensor(sum([[idx] * 5 for idx in range(16)], []), dtype=torch.long)
        self.job_context_opt = torch.rand(80, 40)
        self.job_context_opt_batch = torch.tensor(sum([[idx] * 5 for idx in range(16)], []), dtype=torch.long)
        self.stage_context = torch.cat([torch.zeros(10, 40), torch.rand(70, 40)], dim=0)
        self.stage_context_batch = torch.tensor(sum([[idx] * 5 for idx in range(16)], []), dtype=torch.long)
        self.stage_metrics = torch.cat([torch.zeros(2, 10), torch.rand(14, 10)], dim=0)
        self.prev_job_batch = torch.tensor([0, 8], dtype=torch.long)
        self.real_nodes_batch = torch.tensor(list(range(2, 8)) + list(range(10, 16)), dtype=torch.long)
        self.batch = torch.cat([torch.zeros(8), torch.ones(8)], dim=-1)
        self.num_nodes = 16

        self.model_args = {"hidden_dim": 8,
                           "stage_metrics_dim": 10,
                           "auto_encoder_dropout": 0.05,
                           "metric_dropout": 0.05,
                           "encoding_dim": 40,
                           "device": "cpu"}

        self.model = OnlinePredictor(**self.model_args)
        self.loss = torch.nn.MSELoss()

        self.stage_runtime_pred = torch.rand(16, 1)
        self.stage_metrics_pred = torch.rand(16, 10)
        self.job_runtime_pred = torch.rand(2, 1)

    def test_model_torch_script(self):
        try:
            model = torch.jit.script(self.model)
            print(model)
            print(self.model.get_num_parameters())
            print(self.model.get_num_trainable_parameters())
        except BaseException as exc:
            self.fail(f"Model can not be transformed to torch-script, error: '{exc}'")

    def test_model_forward(self):
        # first: with "normal model"
        result: dict = self.model(self.edge_index, self.stage_start_scale_out_vec, self.stage_end_scale_out_vec,
                                  self.stage_rescaling_time_ratio,
                                  self.job_context_emb, self.job_context_emb_batch,
                                  self.job_context_opt, self.job_context_opt_batch,
                                  self.stage_context, self.stage_context_batch,
                                  self.stage_metrics, self.prev_job_batch, self.real_nodes_batch,
                                  self.num_nodes, self.batch)
        for k, v in result.items():
            self.assertTrue(isinstance(v, torch.Tensor))
            if hasattr(self, k):
                self.assertEqual(v.size(), getattr(self, k).size())
            self.assertEqual(v.isnan().sum(), 0)
            self.assertEqual(v.isinf().sum(), 0)

        # now: with torchscript model
        model = torch.jit.script(self.model)
        result: dict = model(self.edge_index, self.stage_start_scale_out_vec, self.stage_end_scale_out_vec,
                             self.stage_rescaling_time_ratio,
                             self.job_context_emb, self.job_context_emb_batch,
                             self.job_context_opt, self.job_context_opt_batch,
                             self.stage_context, self.stage_context_batch,
                             self.stage_metrics, self.prev_job_batch, self.real_nodes_batch,
                             self.num_nodes, self.batch)
        for k, v in result.items():
            self.assertTrue(isinstance(v, torch.Tensor))
            if hasattr(self, k):
                self.assertEqual(v.size(), getattr(self, k).size())
            self.assertEqual(v.isnan().sum(), 0)
            self.assertEqual(v.isinf().sum(), 0)

    def test_model_backward(self):
        try:
            # first: with "normal model"
            result: dict = self.model(self.edge_index, self.stage_start_scale_out_vec, self.stage_end_scale_out_vec,
                                      self.stage_rescaling_time_ratio,
                                      self.job_context_emb, self.job_context_emb_batch,
                                      self.job_context_opt, self.job_context_opt_batch,
                                      self.stage_context, self.stage_context_batch,
                                      self.stage_metrics, self.prev_job_batch, self.real_nodes_batch,
                                      self.num_nodes, self.batch)
            for k, v in result.items():
                self.assertTrue(isinstance(v, torch.Tensor))
                if hasattr(self, k):
                    self.assertEqual(v.size(), getattr(self, k).size())
                self.assertEqual(v.isnan().sum(), 0)
                self.assertEqual(v.isinf().sum(), 0)
                if hasattr(self, k):
                    loss = self.loss(v, getattr(self, k))
                    loss.backward(retain_graph=True)

            # now: with torchscript model
            model = torch.jit.script(self.model)
            result: dict = model(self.edge_index, self.stage_start_scale_out_vec, self.stage_end_scale_out_vec,
                                 self.stage_rescaling_time_ratio,
                                 self.job_context_emb, self.job_context_emb_batch,
                                 self.job_context_opt, self.job_context_opt_batch,
                                 self.stage_context, self.stage_context_batch,
                                 self.stage_metrics, self.prev_job_batch, self.real_nodes_batch,
                                 self.num_nodes, self.batch)
            for k, v in result.items():
                self.assertTrue(isinstance(v, torch.Tensor))
                if hasattr(self, k):
                    self.assertEqual(v.size(), getattr(self, k).size())
                self.assertEqual(v.isnan().sum(), 0)
                self.assertEqual(v.isinf().sum(), 0)
                if hasattr(self, k):
                    loss = self.loss(v, getattr(self, k))
                    loss.backward(retain_graph=True)
        except BaseException as exc:
            self.fail(f"Model output produces error during backpropagation, error: '{exc}'")
