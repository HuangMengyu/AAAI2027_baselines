import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import Model
from models.TC import TC
from models.loss import loss_ntxent
from models.smdclr import graph_project, instance_graph, mid_loss


class SmdclrLossTests(unittest.TestCase):
    def test_graph_excludes_self_and_has_finite_gradients(self):
        torch.manual_seed(13)
        features = torch.randn(5, 8, requires_grad=True)
        graph = instance_graph(features)
        torch.testing.assert_close(graph.diagonal(), torch.zeros(5))
        torch.testing.assert_close(graph.sum(dim=1), torch.ones(5))
        self.assertTrue((graph[~torch.eye(5, dtype=torch.bool)] > 0).all())
        (graph * torch.arange(25).reshape(5, 5)).sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)

    def test_projection_reuses_head_weights_and_matches_graph_equation(self):
        torch.manual_seed(14)
        features = torch.randn(4, 8, requires_grad=True)
        head = nn.Sequential(nn.Linear(8, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Linear(6, 3)).eval()
        adjacency = instance_graph(features)
        expected = head[3](adjacency @ head[2](head[1](head[0](adjacency @ features))))
        actual = graph_project(features, head)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in head.parameters()))

    def test_mid_is_online_entropy_with_nonzero_gradients(self):
        torch.manual_seed(15)
        first = torch.randn(5, 8, requires_grad=True)
        second = torch.randn(5, 8, requires_grad=True)
        logits = F.normalize(first, dim=1) @ F.normalize(second, dim=1).t() / 0.2
        probabilities = logits.softmax(dim=1)
        loss = mid_loss(first, second)
        torch.testing.assert_close(loss, torch.distributions.Categorical(probs=probabilities).entropy().mean())
        loss.backward()
        for features in (first, second):
            self.assertTrue(torch.isfinite(features.grad).all())
            self.assertGreater(features.grad.abs().sum().item(), 0)
        # A matched diagonal remains a candidate: uniform rows have entropy log B.
        torch.testing.assert_close(mid_loss(torch.ones(4, 8), torch.ones(4, 8)), torch.tensor(4.).log())

    def test_singleton_and_small_temperature_are_finite(self):
        head = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 8))
        features = torch.randn(1, 8, requires_grad=True)
        torch.testing.assert_close(instance_graph(features), torch.zeros(1, 1))
        torch.testing.assert_close(graph_project(features, head), head(features))
        torch.testing.assert_close(mid_loss(features, features), torch.tensor(0.))
        features = torch.randn(4, 8, requires_grad=True)
        loss = mid_loss(features, features, temperature=1e-4)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_invalid_settings_are_rejected(self):
        for temperature in (0., -1., float('nan'), float('inf')):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                TC('cpu', smdclr_temperature=temperature)
        for weight in (-1., float('nan'), float('inf')):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                TC('cpu', mid_weight=weight)

    def test_temporal_loss_matches_id_plus_mid_at_each_horizon(self):
        torch.manual_seed(16)
        temporal = TC('cpu', final_out_channels=8, hidden_dim=8, timesteps=2,
                      depth=1, smdclr_temporal=True, mid_weight=0.7).eval()
        graph_inputs, scores = [], []

        def record_projection(features, head, temperature):
            graph_inputs.append(features)
            return graph_project(features, head, temperature)

        handle = temporal.lsoftmax.register_forward_pre_hook(
            lambda module, args: scores.append(args[0]))
        try:
            with patch('models.TC.graph_project', side_effect=record_projection):
                loss = temporal(torch.randn(4, 8, 7), torch.randn(4, 8, 7))[0]
        finally:
            handle.remove()
        self.assertEqual(len(graph_inputs), 4)
        expected = torch.stack([
            F.cross_entropy(score, torch.arange(4))
            + 0.7 * mid_loss(graph_inputs[2 * k], graph_inputs[2 * k + 1])
            for k, score in enumerate(scores)
        ]).mean()
        torch.testing.assert_close(loss, expected)

    def test_future_targets_do_not_enter_prediction_graph(self):
        torch.manual_seed(17)
        temporal = TC('cpu', final_out_channels=8, hidden_dim=8, timesteps=2,
                      depth=1, smdclr_temporal=True).eval()
        prefix_view = torch.randn(4, 8, 7)
        future_view = torch.randn(4, 8, 7)

        def capture(targets):
            inputs, outputs = [], []

            def record_projection(features, head, temperature):
                inputs.append(features.detach().clone())
                output = graph_project(features, head, temperature)
                outputs.append(output.detach().clone())
                return output

            torch.manual_seed(123)  # use the same sampled prefix in both calls
            with patch('models.TC.graph_project', side_effect=record_projection):
                temporal(prefix_view, targets)
            return inputs, outputs

        original, original_outputs = capture(future_view)
        changed, changed_outputs = capture(future_view + torch.randn_like(future_view) * 5)
        for k in range(2):
            target, prediction = 2 * k, 2 * k + 1
            self.assertFalse(torch.equal(original[target], changed[target]))
            torch.testing.assert_close(original[prediction], changed[prediction], rtol=0, atol=0)
            torch.testing.assert_close(original_outputs[prediction], changed_outputs[prediction], rtol=0, atol=0)
            torch.testing.assert_close(instance_graph(original[prediction]),
                                       instance_graph(changed[prediction]), rtol=0, atol=0)


class SmdclrModelTests(unittest.TestCase):
    def make_model(self, **kwargs):
        settings = dict(mod1_dims=3, mod2_dims=3, device='cpu', final_out_channels=32,
                        hidden_dim=16, timesteps=3, depth=1, num_epochs=1)
        settings.update(kwargs)
        return Model(**settings)

    def test_flags_control_only_the_requested_losses(self):
        torch.manual_seed(18)
        inputs = tuple(torch.randn(4, 3, 128) for _ in range(4))
        # Context projection adds no parameters or random initialization.
        torch.manual_seed(19)
        vanilla = self.make_model().eval()
        torch.manual_seed(19)
        contextual = self.make_model(smdclr_context=True).eval()
        self.assertEqual(vanilla.state_dict().keys(), contextual.state_dict().keys())
        for key, value in vanilla.state_dict().items():
            torch.testing.assert_close(value, contextual.state_dict()[key], rtol=0, atol=0)
        torch.manual_seed(20)
        baseline = vanilla.TFCC.loss_compute(inputs)
        torch.manual_seed(20)
        context = contextual.TFCC.loss_compute(inputs)
        torch.testing.assert_close(baseline[0], context[0], rtol=0, atol=0)
        self.assertTrue(all(not torch.equal(baseline[i], context[i]) for i in (1, 2, 3)))

        temporal = self.make_model(smdclr_temporal=True).eval()
        temporal.load_state_dict(vanilla.state_dict(), strict=False)
        torch.manual_seed(20)
        temporal_only = temporal.TFCC.loss_compute(inputs)
        for i in (1, 2, 3):
            torch.testing.assert_close(baseline[i], temporal_only[i], rtol=0, atol=0)
        self.assertFalse(torch.equal(baseline[0], temporal_only[0]))

    def test_all_context_losses_include_mid(self):
        model = self.make_model(smdclr_context=True, mid_weight=0.7).eval()
        # Precomputed contexts isolate wiring and avoid random prefix sampling.
        torch.manual_seed(21)
        contexts = [torch.randn(4, 16) for _ in range(4)]
        projections = [torch.randn(4, 8) for _ in range(4)]
        eeg = [(torch.tensor(1.), contexts[i], projections[i]) for i in (0, 1)]
        eog = [(torch.tensor(1.), contexts[i], projections[i]) for i in (2, 3)]
        with patch.object(model.TFCC.EEG_contrasting, 'forward', side_effect=eeg), \
             patch.object(model.TFCC.EOG_contrasting, 'forward', side_effect=eog):
            result = model.TFCC.loss_compute(tuple(torch.randn(4, 3, 128) for _ in range(4)))
        for slot, first, second in ((1, 0, 1), (2, 2, 3)):
            expected = loss_ntxent([projections[first], projections[second]], 'cpu')[0]
            expected += 0.7 * mid_loss(contexts[first], contexts[second])
            torch.testing.assert_close(result[slot], expected)
        first = graph_project(contexts[0], model.TFCC.shared_projection_head_1)
        second = graph_project(contexts[2], model.TFCC.shared_projection_head_2)
        expected = (loss_ntxent([first, second], 'cpu')[0]
                    + loss_ntxent([second, first], 'cpu')[0]
                    + 0.7 * (mid_loss(contexts[0], contexts[2])
                             + mid_loss(contexts[2], contexts[0]))) / 2
        torch.testing.assert_close(result[3], expected)

    def test_training_optimizer_checkpoint_and_inference_for_dataset_shapes(self):
        cases = (('UCI-HAR', 3, 3, 128, 3, 6),
                 ('PAMAP2', 9, 9, 256, 5, 9),
                 ('SleepEDFx', 2, 1, 3000, 50, 5))
        for dataset, first_channels, second_channels, length, timesteps, classes in cases:
            with self.subTest(dataset=dataset):
                torch.manual_seed(22)
                model = self.make_model(mod1_dims=first_channels, mod2_dims=second_channels,
                                        timesteps=timesteps, num_class=classes,
                                        smdclr_context=True, smdclr_temporal=True)
                # Three samples ensure a nontrivial graph (B=2 has one neighbor).
                inputs = tuple(torch.randn(3, channels, length) for channels in
                               (first_channels, second_channels, first_channels, second_channels))
                result = model.TFCC.loss_compute(inputs)
                self.assertEqual(len(result), 16)
                self.assertEqual(result[10:], (0,) * 6)
                self.assertEqual([value.numel() for value in result[7:10]], [6, 6, 12])
                loss = 0.1 * result[0] + 0.6 * (result[1] + result[2]) + 0.4 * result[3]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                optimized = {id(p) for group in model.TFCC.optimizer.param_groups for p in group['params']}
                for temporal in (model.TFCC.EEG_contrasting, model.TFCC.EOG_contrasting):
                    for p in temporal.temporal_projection_head.parameters():
                        self.assertIn(id(p), optimized)
                        self.assertIsNotNone(p.grad)
                        self.assertTrue(torch.isfinite(p.grad).all())
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                before = model.TFCC.EEG_contrasting.temporal_projection_head[0].weight.detach().clone()
                loader = DataLoader(TensorDataset(*inputs, torch.arange(3)), batch_size=3)
                model(loader, ssl=True)
                self.assertFalse(torch.equal(before, model.TFCC.EEG_contrasting.temporal_projection_head[0].weight))
                model.eval()
                # Graph heads are pretraining-only and must never enter inference.
                with patch('models.TC.graph_project', side_effect=AssertionError('graph used in inference')), \
                     patch('model.graph_project', side_effect=AssertionError('graph used in inference')), torch.no_grad():
                    prediction = model(inputs[:2])
                    one = model((inputs[0][:1], inputs[1][:1]))
                self.assertEqual(prediction.shape, (3, classes))
                torch.testing.assert_close(one[0], prediction[0], rtol=1e-4, atol=1e-5)
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = self.make_model(mod1_dims=first_channels, mod2_dims=second_channels,
                                           timesteps=timesteps, num_class=classes,
                                           smdclr_context=True, smdclr_temporal=True).eval()
                restored.load_state_dict(torch.load(buffer, map_location='cpu'), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(restored(inputs[:2]), prediction)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
