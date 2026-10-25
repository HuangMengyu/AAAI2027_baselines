import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model as model_module
from models.TC import TC
from models.loss import loss_ntxent, nsf_keep_mask, validate_nsf_threshold


class NSFLossTests(unittest.TestCase):
    def test_plain_loss_matches_cross_entropy(self):
        torch.manual_seed(0)
        first, second = torch.randn(4, 8), torch.randn(4, 8)
        similarity = F.normalize(first, dim=1) @ F.normalize(second, dim=1).t()
        result = loss_ntxent([first, second], 'cpu')
        torch.testing.assert_close(result[0], F.cross_entropy(similarity / 0.2, torch.arange(4)))
        torch.testing.assert_close(result[1], similarity.diagonal())
        torch.testing.assert_close(result[2], similarity[~torch.eye(4, dtype=torch.bool)])
        self.assertEqual(result[3:], (0, 0))
        self.assertEqual(len(result), 5)

        # Two orthogonal pairs have one negative each, with no extra exp(0).
        actual = loss_ntxent([torch.eye(2), torch.eye(2)], 'cpu')[0]
        torch.testing.assert_close(actual, torch.log1p(torch.exp(torch.tensor(-5.))))

    def test_threshold_boundary_and_positive_retention(self):
        similarity = torch.tensor([[0.5, 0.25, -0.1],
                                   [0.25, 1.0, 0.6],
                                   [0.1, 0.125, 0.25]], dtype=torch.float64)
        expected = torch.tensor([[True, False, True],
                                 [True, True, False],
                                 [True, False, True]])
        self.assertTrue(torch.equal(nsf_keep_mask(similarity, 0.5), expected))
        self.assertTrue(nsf_keep_mask(similarity, -1).all())

    def test_nonpositive_and_near_zero_rows_stay_unfiltered(self):
        similarity = torch.full((4, 4), 1.0, dtype=torch.float64)
        similarity.diagonal().copy_(torch.tensor([-0.5, 0., 1e-9, 1e-8], dtype=torch.float64))
        self.assertTrue(nsf_keep_mask(similarity, 0.5).all())

    def test_filtered_logits_have_zero_gradient(self):
        similarity = torch.tensor([[1., 0.9, 0.1], [0.1, 1., 0.9],
                                   [0.9, 0.1, 1.]], requires_grad=True)
        keep = nsf_keep_mask(similarity, 0.8)
        loss = -F.log_softmax((similarity / 0.2).masked_fill(~keep, -torch.inf), dim=1).diagonal().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(similarity.grad).all())
        self.assertTrue((similarity.grad[~keep] == 0).all())
        self.assertTrue((similarity.grad[keep & ~torch.eye(3, dtype=torch.bool)] > 0).all())

    def test_all_negatives_filtered_out(self):
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                features = torch.ones(batch_size, 4, requires_grad=True)
                loss, _, negatives, _, _ = loss_ntxent([features, features], 'cpu', 0.9)
                self.assertEqual(loss.item(), 0.)
                self.assertEqual(negatives.numel(), 0)
                loss.backward()
                self.assertTrue(torch.isfinite(features.grad).all())

    def test_threshold_validation(self):
        for value in (-1, 0.9, 2.):
            self.assertEqual(validate_nsf_threshold(value), value)
        for value in (0, -0.5, -2, float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_nsf_threshold(value)

    def test_temporal_plain_loss_matches_cross_entropy(self):
        import models.TC as temporal_module

        temporal = TC('cpu', final_out_channels=8, hidden_dim=8, timesteps=2, depth=1).eval()
        features = (torch.randn(3, 8, 6), torch.randn(3, 8, 6))
        with patch.object(temporal_module, 'nsf_keep_mask', wraps=nsf_keep_mask) as masks:
            loss = temporal(*features)[0]
        expected = torch.stack([
            F.cross_entropy(call.args[0], torch.arange(3)) for call in masks.call_args_list
        ]).mean()
        torch.testing.assert_close(loss, expected)


class NSFModelTests(unittest.TestCase):
    def test_shared_threshold_forward_backward_and_checkpoint(self):
        for threshold in (-1, 0.9):
            with self.subTest(threshold=threshold):
                torch.manual_seed(0)
                model = model_module.Model(mod1_dims=3, mod2_dims=3, device='cpu',
                                           final_out_channels=32, hidden_dim=16,
                                           timesteps=3, depth=1, num_epochs=1, warm_epochs=0,
                                           nsf_threshold=threshold)
                inputs = tuple(torch.randn(2, 3, 128) for _ in range(4))
                original_forward = TC.forward
                with patch.object(TC, 'forward', autospec=True, side_effect=original_forward) as temporal, \
                        patch.object(model_module, 'loss_ntxent', wraps=loss_ntxent) as contextual:
                    result = model.TFCC.loss_compute(inputs)
                self.assertEqual(len(result), 16)
                self.assertEqual(result[10:], (0,) * 6)
                for calls in (temporal.call_args_list, contextual.call_args_list):
                    self.assertEqual(len(calls), 4)
                    self.assertTrue(all(call.kwargs['nsf_threshold'] == threshold for call in calls))
                loss = 0.1 * result[0] + 0.6 * (result[1] + result[2]) + 0.4 * result[3]
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

                # Exercise the existing training loop and classifier too.
                loader = DataLoader(TensorDataset(*inputs, torch.arange(2)), batch_size=2)
                before = model.EEG_encoder.conv_block1[0].weight.detach().clone()
                model(loader, ssl=True)
                self.assertFalse(torch.equal(before, model.EEG_encoder.conv_block1[0].weight))
                model.eval()
                with torch.no_grad():
                    prediction = model(inputs[:2])
                self.assertEqual(prediction.shape, (2, 5))
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                model.load_state_dict(torch.load(buffer, map_location='cpu'), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(model(inputs[:2]), prediction)

    def test_filtering_starts_after_warmup(self):
        cases = ((2, 3, 0.9, [-1, -1, 0.9]),
                 (0, 1, 0.9, [0.9]),
                 (3, 2, 0.9, [-1, -1]),
                 (1, 2, -1, [-1, -1]))
        for warm_epochs, num_epochs, threshold, expected in cases:
            with self.subTest(warm_epochs=warm_epochs, num_epochs=num_epochs, threshold=threshold):
                model = model_module.Model(mod1_dims=3, mod2_dims=3, device='cpu',
                                           final_out_channels=32, hidden_dim=16,
                                           timesteps=3, depth=1, num_epochs=num_epochs,
                                           warm_epochs=warm_epochs, nsf_threshold=threshold)
                inputs = tuple(torch.randn(4, 3, 128) for _ in range(4))
                loader = DataLoader(TensorDataset(*inputs, torch.arange(4)), batch_size=2)
                original_forward = TC.forward
                with patch.object(TC, 'forward', autospec=True, side_effect=original_forward) as temporal, \
                        patch.object(model_module, 'loss_ntxent', wraps=loss_ntxent) as contextual:
                    model(loader, ssl=True)
                expected_calls = [value for value in expected for _ in range(4 * len(loader))]
                for calls in (temporal.call_args_list, contextual.call_args_list):
                    self.assertEqual([call.kwargs['nsf_threshold'] for call in calls], expected_calls)
                self.assertEqual(model.TFCC.nsf_threshold, threshold)

    def test_negative_warmup_rejected(self):
        with self.assertRaises(ValueError):
            model_module.Model(device='cpu', warm_epochs=-1)

    def test_iteration_training_rejected(self):
        with self.assertRaises(NotImplementedError):
            model_module.Model(device='cpu', use_iteration=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
