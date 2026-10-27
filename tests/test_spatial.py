"""CPU-only tests. Run: python -m unittest discover -s tests -v"""
import importlib.util
import pathlib
import unittest
import torch
from torch import nn

SPEC = importlib.util.spec_from_file_location("spatial", pathlib.Path(__file__).resolve().parents[1] / "spatial.py")
spatial = importlib.util.module_from_spec(SPEC)
import sys
sys.modules["spatial"] = spatial
SPEC.loader.exec_module(spatial)


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(2, 6, bias=False)
        self.out = nn.Linear(2, 2, bias=False)


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = ToyAttention()


class ToyNet(nn.Module):
    patch_size = 2
    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(1, 2)
        self.layers = nn.ModuleList([ToyBlock()])
    def forward(self, x):
        return self.layers[0].attention.qkv(x)


class SpatialTests(unittest.TestCase):
    def test_bbox_gate_is_zero_in_text_and_outside_box(self):
        mask = torch.tensor([[[1., 1., 0., 0.], [1., 1., 0., 0.]]])
        gate = spatial.build_gate_from_latent(mask, 1, 4, 2, 4, 2, "cpu", torch.float32)
        self.assertEqual(gate.flatten().tolist(), [0., 0., 1., 0.])

    def test_image_padding_is_after_image_tokens(self):
        mask = torch.tensor([[[1., 1., 0., 0.], [1., 1., 0., 0.]]])
        # Image has 2 tokens, padded to 4. Two prompt tokens precede the image.
        gate = spatial.build_gate_from_latent(mask, 1, 6, 2, 4, 2, "cpu", torch.float32, 4)
        self.assertEqual(gate.flatten().tolist(), [0., 0., 1., 0., 0., 0.])

    def test_native_and_split_adapters_load(self):
        state = {
            "diffusion_model.layers.0.attention.qkv.lora_down.weight": torch.ones(1, 2),
            "diffusion_model.layers.0.attention.qkv.lora_up.weight": torch.ones(6, 1),
            "diffusion_model.layers.0.attention.qkv.alpha": torch.tensor(1.),
        }
        adapters, _ = spatial.extract_adapters(state)
        self.assertEqual(len(adapters), 1)
        self.assertEqual(adapters[0].target, "layers.0.attention.qkv")
        self.assertEqual(adapters[0].scale, 1.0)

    def test_hook_does_not_change_text_or_unmasked_image_delta(self):
        net = ToyNet()
        with torch.no_grad():
            net.layers[0].attention.qkv.weight.zero_()
        a = spatial.Adapter("layers.0.attention.qkv", torch.ones(1, 2), torch.ones(6, 1), 1.0)
        x = torch.ones(1, 4, 2)
        mask = torch.tensor([[[1., 1., 0., 0.], [1., 1., 0., 0.]]])
        before = net(x)
        with spatial.temporary_lora_hooks(net, [a], mask, torch.zeros(1, 1, 2, 4), 1.0):
            after = net(x)
        self.assertTrue(torch.allclose(after[0, 0], before[0, 0]))
        self.assertTrue(torch.allclose(after[0, 1], before[0, 1]))
        self.assertTrue(torch.allclose(after[0, 3], before[0, 3]))
        self.assertTrue(torch.allclose(after[0, 2], torch.full((6,), 2.)))
        self.assertTrue(torch.allclose(net(x), before))  # hooks removed

    def test_split_qkv_only_changes_selected_projection(self):
        net = ToyNet()
        with torch.no_grad():
            net.layers[0].attention.qkv.weight.zero_()
        a = spatial.Adapter("layers.0.attention.qkv", torch.ones(1, 2), torch.ones(2, 1), 1.0, slice_index=1)
        x = torch.ones(1, 4, 2)
        mask = torch.tensor([[[1., 1., 0., 0.], [1., 1., 0., 0.]]])
        with spatial.temporary_lora_hooks(net, [a], mask, torch.zeros(1, 1, 2, 4), 1.0):
            after = net(x)
        self.assertEqual(after[0, 2].tolist(), [0., 0., 2., 2., 0., 0.])

    def test_unmatched_lora_key_raises_instead_of_silent_skip(self):
        state = {"transformer.layers.0.attention.qkv.lora_A.mystery.weight": torch.ones(1, 2)}
        with self.assertRaisesRegex(ValueError, "nao reconhecido"):
            spatial.extract_adapters(state)

    def test_alt_lora_alpha_key_is_parsed(self):
        state = {
            "transformer.layers.0.attention.qkv.lora_A.weight": torch.ones(2, 2),
            "transformer.layers.0.attention.qkv.lora_B.weight": torch.ones(6, 2),
            "transformer.layers.0.attention.qkv.lora_alpha": torch.tensor(1.),
        }
        adapters, ignored = spatial.extract_adapters(state)
        self.assertEqual(len(adapters), 1)
        self.assertEqual(ignored, [])
        self.assertEqual(adapters[0].scale, 0.5)

    def test_invalid_qkv_slice_is_rejected(self):
        net = ToyNet()
        bad = spatial.Adapter("layers.0.attention.qkv", torch.ones(1, 2), torch.ones(2, 1), 1., slice_index=3)
        with self.assertRaisesRegex(ValueError, "QKV invalido"):
            spatial.validate_targets(net, [bad])

    def test_too_short_token_sequence_fails_closed(self):
        mask = torch.ones(1, 4, 4)
        with self.assertRaisesRegex(ValueError, "menor"):
            spatial.build_gate_from_latent(mask, 1, 3, 4, 4, 2, "cpu", torch.float32)

    def test_hooks_are_removed_after_exception(self):
        net = ToyNet()
        adapter = spatial.Adapter("layers.0.attention.qkv", torch.ones(1, 2), torch.ones(6, 1), 1.)
        mask = torch.ones(1, 2, 4)
        with self.assertRaisesRegex(RuntimeError, "simulated failure"):
            with spatial.temporary_lora_hooks(net, [adapter], mask, torch.zeros(1, 1, 2, 4), 1.):
                raise RuntimeError("simulated failure")
        self.assertEqual(len(net.layers[0].attention.qkv._forward_hooks), 0)

    def test_fail_for_wrong_model(self):
        class NotZImage(nn.Module):
            pass
        with self.assertRaises(TypeError):
            spatial.validate_targets(NotZImage(), [spatial.Adapter("layers.0.attention.qkv", torch.ones(1, 2), torch.ones(6, 1), 1.)])


if __name__ == "__main__":
    unittest.main()
