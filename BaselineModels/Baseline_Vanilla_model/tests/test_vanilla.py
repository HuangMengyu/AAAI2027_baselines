import io
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import Model
from models.TC import TC
from models.loss import loss_ntxent


class VanillaLossTests(unittest.TestCase):
    def test_contextual_loss_matches_cross_entropy(self):
        torch.manual_seed(0)
        first, second = torch.randn(4, 8), torch.randn(4, 8)
        similarity = F.normalize(first, dim=1) @ F.normalize(second, dim=1).t()
        result = loss_ntxent([first, second], 'cpu')
        torch.testing.assert_close(result[0], F.cross_entropy(similarity / 0.2, torch.arange(4)))
        torch.testing.assert_close(result[1], similarity.diagonal())
        torch.testing.assert_close(result[2], similarity[~torch.eye(4, dtype=torch.bool)])
        self.assertEqual(len(result), 5)
        self.assertEqual(result[3:], (0, 0))
        actual = loss_ntxent([torch.eye(2), torch.eye(2)], 'cpu')[0]
        torch.testing.assert_close(actual, torch.log1p(torch.exp(torch.tensor(-5.))))

    def test_identical_negatives_are_retained(self):
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                features = torch.ones(batch_size, 4, requires_grad=True)
                loss, _, negatives, _, _ = loss_ntxent([features, features], 'cpu')
                torch.testing.assert_close(loss, torch.tensor(float(batch_size)).log())
                self.assertEqual(negatives.numel(), batch_size * (batch_size - 1))
                loss.backward()
                self.assertTrue(torch.isfinite(features.grad).all())

    def test_temporal_loss_matches_cross_entropy(self):
        temporal = TC('cpu', final_out_channels=8, hidden_dim=8, timesteps=2, depth=1).eval()
        features = (torch.randn(3, 8, 6), torch.randn(3, 8, 6))
        scores = []
        handle = temporal.lsoftmax.register_forward_pre_hook(
            lambda module, args: scores.append(args[0].detach()))
        try:
            loss = temporal(*features)[0]
        finally:
            handle.remove()
        self.assertEqual(len(scores), 2)
        expected = torch.stack([F.cross_entropy(score, torch.arange(3)) for score in scores]).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in temporal.parameters() if p.grad is not None))

    def test_filter_flag_does_not_change_losses(self):
        features = (torch.randn(3, 8), torch.randn(3, 8))
        for enabled in (False, True):
            result = loss_ntxent(features, 'cpu', filter=enabled)
            reference = loss_ntxent(features, 'cpu')
            for actual, expected in zip(result[:3], reference[:3]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        temporal = TC('cpu', final_out_channels=8, hidden_dim=8, timesteps=2, depth=1).eval()
        features = (torch.randn(3, 8, 6), torch.randn(3, 8, 6))
        torch.manual_seed(7)
        reference = temporal(*features, filter=False)
        torch.manual_seed(7)
        actual = temporal(*features, filter=True)
        for actual_value, expected in zip(actual, reference):
            torch.testing.assert_close(actual_value, expected, rtol=0, atol=0)


class VanillaModelTests(unittest.TestCase):
    def test_training_and_checkpoint_for_dataset_shapes(self):
        cases = (('UCI-HAR', 3, 3, 128, 3, 6),
                 ('PAMAP2', 9, 9, 256, 5, 9),
                 ('SleepEDFx', 2, 1, 3000, 50, 5))
        for dataset, first_channels, second_channels, length, timesteps, classes in cases:
            with self.subTest(dataset=dataset):
                model = Model(mod1_dims=first_channels, mod2_dims=second_channels,
                              device='cpu', final_out_channels=32, hidden_dim=16,
                              timesteps=timesteps, num_class=classes, depth=1, num_epochs=1)
                inputs = tuple(torch.randn(2, channels, length) for channels in
                               (first_channels, second_channels, first_channels, second_channels))
                result = model.TFCC.loss_compute(inputs)
                self.assertEqual(len(result), 16)
                self.assertEqual(result[10:], (0,) * 6)
                self.assertTrue(all(torch.isfinite(value) for value in result[:4]))
                self.assertEqual(result[7].numel(), 2)
                self.assertEqual(result[8].numel(), 2)
                self.assertEqual(result[9].numel(), 4)
                loss = 0.1 * result[0] + 0.6 * (result[1] + result[2]) + 0.4 * result[3]
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

                loader = DataLoader(TensorDataset(*inputs, torch.arange(2)), batch_size=2)
                before = model.EEG_encoder.conv_block1[0].weight.detach().clone()
                model(loader, ssl=True)
                self.assertFalse(torch.equal(before, model.EEG_encoder.conv_block1[0].weight))
                model.eval()
                with torch.no_grad():
                    prediction = model(inputs[:2])
                self.assertEqual(prediction.shape, (2, classes))
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                model.load_state_dict(torch.load(buffer, map_location='cpu'), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(model(inputs[:2]), prediction)

    def test_warmup_and_filter_settings_do_not_change_training(self):
        inputs = tuple(torch.randn(2, 3, 128) for _ in range(4))
        reference = None
        for warm_epochs, enabled in ((0, False), (1, True)):
            torch.manual_seed(7)
            model = Model(mod1_dims=3, mod2_dims=3, device='cpu', final_out_channels=32,
                          hidden_dim=16, timesteps=3, depth=1, num_epochs=2,
                          warm_epochs=warm_epochs, filter=enabled)
            self.assertEqual(model.TFCC.warm_epochs, warm_epochs)
            self.assertEqual(model.TFCC.filter, enabled)
            loader = DataLoader(TensorDataset(*inputs, torch.arange(2)), batch_size=2)
            torch.manual_seed(8)
            model(loader, ssl=True)
            state = model.state_dict()
            if reference is None:
                reference = {key: value.clone() for key, value in state.items()}
            else:
                self.assertEqual(state.keys(), reference.keys())
                for key, value in state.items():
                    torch.testing.assert_close(value, reference[key], rtol=0, atol=0)

    def test_iteration_training_rejected(self):
        with self.assertRaises(NotImplementedError):
            Model(device='cpu', use_iteration=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
