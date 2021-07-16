from unittest import TestCase

import torch

from enel_service.modeling.model_ops import RuntimeConv, MetricConv, OverheadConv


class TestOverheadConv(TestCase):

    def setUp(self) -> None:
        self.context_tensor = torch.rand(5, 40)
        self.x_start_scale_out_vec = torch.rand(5, 3)
        self.x_end_scale_out_vec = torch.rand(5, 3)
        self.x_rescaling_time_ratio = torch.rand(5, 1)
        self.metric_tensor = torch.rand(5, 10)
        self.true_time = torch.abs(torch.rand(5, 1))
        self.edge_index = torch.ones(5, 5).nonzero(as_tuple=False).t()
        self.real_nodes_batch = torch.tensor(list(range(2, 5)), dtype=torch.long)
        self.conv = OverheadConv(57, 1)
        self.loss = torch.nn.MSELoss()

    def test_module_scriptable(self):
        try:
            conv = self.conv
            torch.jit.script(conv)
            print(conv)
            print(conv.get_num_parameters())
            print(conv.get_num_trainable_parameters())
        except BaseException as exc:
            self.fail(f"Module is not scriptable, error: '{exc}'")

    def test_module_forward(self):
        # first: with "normal module"
        pred_overhead = self.conv(self.x_start_scale_out_vec, self.x_end_scale_out_vec, self.x_rescaling_time_ratio,
                                  self.context_tensor, self.metric_tensor, self.real_nodes_batch)
        self.assertTrue(isinstance(pred_overhead, torch.Tensor))
        self.assertEqual(pred_overhead.size(), (self.context_tensor.size()[0], 1))
        self.assertEqual(pred_overhead.isnan().sum(), 0)
        self.assertEqual(pred_overhead.isinf().sum(), 0)

        # now: with torchscript module
        script_conv = torch.jit.script(self.conv)
        pred_overhead = script_conv(self.x_start_scale_out_vec, self.x_end_scale_out_vec, self.x_rescaling_time_ratio,
                                    self.context_tensor, self.metric_tensor, self.real_nodes_batch)
        self.assertTrue(isinstance(pred_overhead, torch.Tensor))
        self.assertEqual(pred_overhead.size(), (self.context_tensor.size()[0], 1))
        self.assertEqual(pred_overhead.isnan().sum(), 0)
        self.assertEqual(pred_overhead.isinf().sum(), 0)

    def test_module_backward(self):
        try:
            # first: with "normal module"
            pred_overhead = self.conv(self.x_start_scale_out_vec, self.x_end_scale_out_vec, self.x_rescaling_time_ratio,
                                      self.context_tensor, self.metric_tensor, self.real_nodes_batch)
            loss = self.loss(pred_overhead, self.true_time)
            loss.backward(retain_graph=True)

            # now: with torchscript module
            script_conv = torch.jit.script(self.conv)
            pred_overhead = script_conv(self.x_start_scale_out_vec, self.x_end_scale_out_vec,
                                        self.x_rescaling_time_ratio, self.context_tensor,
                                        self.metric_tensor, self.real_nodes_batch)
            loss = self.loss(pred_overhead, self.true_time)
            loss.backward()
        except BaseException as exc:
            self.fail(f"Module output produces error during backpropagation, error: '{exc}'")


class TestRuntimeConv(TestCase):

    def setUp(self) -> None:
        self.context_tensor = torch.rand(5, 40)
        self.x_scale_out_vec = torch.rand(5, 3)
        self.metric_tensor = torch.rand(5, 10)
        self.overhead_tensor = torch.rand(5, 1)
        self.time_cumsum_tensor = torch.rand(5, 1)
        self.true_time = torch.abs(torch.rand(5, 1))
        self.edge_index = torch.ones(5, 5).nonzero(as_tuple=False).t()
        self.real_nodes_batch = torch.tensor(list(range(2, 5)), dtype=torch.long)
        self.conv = RuntimeConv(54, 1)
        self.loss = torch.nn.MSELoss()

    def test_module_jittable(self):
        try:
            conv = self.conv.jittable()
            torch.jit.script(conv)
            print(conv)
            print(conv.get_num_parameters())
            print(conv.get_num_trainable_parameters())
        except BaseException as exc:
            self.fail(f"Module is not jittable, error: '{exc}'")

    def test_module_forward(self):
        # first: with "normal module"
        pred_time, prop_time = self.conv(self.edge_index, self.x_scale_out_vec,
                                         self.context_tensor, self.metric_tensor, self.real_nodes_batch,
                                         self.overhead_tensor, self.time_cumsum_tensor)
        self.assertTrue(isinstance(pred_time, torch.Tensor))
        self.assertEqual(pred_time.size(), (self.context_tensor.size()[0], 1))
        self.assertEqual(pred_time.isnan().sum(), 0)
        self.assertEqual(pred_time.isinf().sum(), 0)

        self.assertTrue(isinstance(prop_time, torch.Tensor))
        self.assertEqual(prop_time.size(), (self.context_tensor.size()[0], 1))
        self.assertEqual(prop_time.isnan().sum(), 0)
        self.assertEqual(prop_time.isinf().sum(), 0)

        # now: with torchscript module
        script_conv = torch.jit.script(self.conv.jittable())
        pred_time, prop_time = script_conv(self.edge_index, self.x_scale_out_vec,
                                           self.context_tensor, self.metric_tensor, self.real_nodes_batch,
                                           self.overhead_tensor, self.time_cumsum_tensor)
        self.assertTrue(isinstance(pred_time, torch.Tensor))
        self.assertEqual(pred_time.size(), (self.context_tensor.size()[0], 1))
        self.assertEqual(pred_time.isnan().sum(), 0)
        self.assertEqual(pred_time.isinf().sum(), 0)

        self.assertTrue(isinstance(prop_time, torch.Tensor))
        self.assertEqual(prop_time.size(), (self.context_tensor.size()[0], 1))
        self.assertEqual(prop_time.isnan().sum(), 0)
        self.assertEqual(prop_time.isinf().sum(), 0)

    def test_module_backward(self):
        try:
            # first: with "normal module"
            pred_time, prop_time = self.conv(self.edge_index, self.x_scale_out_vec,
                                             self.context_tensor, self.metric_tensor, self.real_nodes_batch,
                                             self.overhead_tensor, self.time_cumsum_tensor)
            loss = self.loss(pred_time, self.true_time)
            loss.backward(retain_graph=True)
            loss = self.loss(prop_time, self.true_time)
            loss.backward(retain_graph=True)

            # now: with torchscript module
            script_conv = torch.jit.script(self.conv.jittable())
            pred_time, prop_time = script_conv(self.edge_index, self.x_scale_out_vec,
                                               self.context_tensor, self.metric_tensor, self.real_nodes_batch,
                                               self.overhead_tensor, self.time_cumsum_tensor)
            loss = self.loss(pred_time, self.true_time)
            loss.backward(retain_graph=True)
            loss = self.loss(prop_time, self.true_time)
            loss.backward()
        except BaseException as exc:
            self.fail(f"Module output produces error during backpropagation, error: '{exc}'")


class TestMetricConv(TestCase):

    def setUp(self) -> None:
        self.x_start_scale_out_vec = torch.rand(5, 3)
        self.x_end_scale_out_vec = torch.rand(5, 3)
        self.context_tensor = torch.rand(5, 40)
        self.metric_tensor = torch.rand(5, 10)
        self.true_metric = torch.abs(torch.rand(5, 10))
        self.edge_index = torch.ones(5, 5).nonzero(as_tuple=False).t()
        self.conv = MetricConv(46, 10, 10, 0.5)
        self.loss = torch.nn.MSELoss()

    def test_module_jittable(self):
        try:
            conv = self.conv.jittable()
            torch.jit.script(conv)
            print(conv)
            print(conv.get_num_parameters())
            print(conv.get_num_trainable_parameters())
        except BaseException as exc:
            self.fail(f"Module is not jittable, error: '{exc}'")

    def test_module_forward(self):
        # first: with "normal module"
        prop_metric = self.conv(self.edge_index, self.x_start_scale_out_vec, self.x_end_scale_out_vec,
                                self.context_tensor, self.metric_tensor)
        self.assertTrue(isinstance(prop_metric, torch.Tensor))
        self.assertEqual(prop_metric.size(), self.metric_tensor.size())
        self.assertEqual(prop_metric.isnan().sum(), 0)
        self.assertEqual(prop_metric.isinf().sum(), 0)

        # now: with torchscript module
        script_conv = torch.jit.script(self.conv.jittable())
        prop_metric = script_conv(self.edge_index, self.x_start_scale_out_vec, self.x_end_scale_out_vec,
                                  self.context_tensor, self.metric_tensor)
        self.assertTrue(isinstance(prop_metric, torch.Tensor))
        self.assertEqual(prop_metric.size(), self.metric_tensor.size())
        self.assertEqual(prop_metric.isnan().sum(), 0)
        self.assertEqual(prop_metric.isinf().sum(), 0)

    def test_module_backward(self):
        try:
            # first: with "normal module"
            prop_metric = self.conv(self.edge_index, self.x_start_scale_out_vec, self.x_end_scale_out_vec,
                                    self.context_tensor, self.metric_tensor)
            loss = self.loss(prop_metric, self.true_metric)
            loss.backward(retain_graph=True)

            # now: with torchscript module
            script_conv = torch.jit.script(self.conv.jittable())
            prop_metric = script_conv(self.edge_index, self.x_start_scale_out_vec, self.x_end_scale_out_vec,
                                      self.context_tensor, self.metric_tensor)
            loss = self.loss(prop_metric, self.true_metric)
            loss.backward()
        except BaseException as exc:
            self.fail(f"Module output produces error during backpropagation, error: '{exc}'")
