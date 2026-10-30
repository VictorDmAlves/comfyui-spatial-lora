"""CPU-only synthetic checks for Anima spatial LoRA, not GPU inference proof."""
import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch import nn

from test_multi import ToyPatcher

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = 'spatial_anima_test_plugin'
if PKG not in sys.modules:
    spec = importlib.util.spec_from_file_location(PKG, ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
    obj = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = obj
    spec.loader.exec_module(obj)
from spatial_anima_test_plugin.backends.anima import (extract_anima_adapters, AnimaBackend,
                                                       anima_mask_gate, temporary_anima_hooks)
from spatial_anima_test_plugin.spatial import SpatialRegion
from spatial_anima_test_plugin.backends import resolve_backend
from spatial_anima_test_plugin.multi_lora import SpatialMultiLoRA


class SimpleAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 2, bias=False)
        self.k_proj = nn.Linear(2, 2, bias=False)
        self.v_proj = nn.Linear(2, 2, bias=False)
        self.o_proj = nn.Linear(2, 2, bias=False)
        for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.zeros_(m.weight)


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 2, bias=False)
        self.layer2 = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(self.layer1.weight)
        nn.init.zeros_(self.layer2.weight)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = SimpleAttn()
        self.cross_attn = SimpleAttn()
        self.mlp = SimpleMLP()


class AnimaToy(nn.Module):
    patch_spatial = 2
    patch_temporal = 1

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block()])
        self.llm_adapter = nn.Linear(2, 2)
        self.x_embedder = nn.Linear(2, 2)

    def forward(self, x):
        # [B,1,2,2,2], attention sees [B,4,2], MLP sees 5D tokens.
        h = x.reshape(x.shape[0], 4, 2)
        att = self.blocks[0].self_attn.q_proj(h)
        cross_q = self.blocks[0].cross_attn.q_proj(h)
        mlp = self.blocks[0].mlp.layer1(x)
        return att + cross_q + mlp.reshape(x.shape[0], 4, 2)


class AnimaTests(unittest.TestCase):
    def setUp(self):
        self.net = AnimaToy()
        self.patcher = ToyPatcher(self.net)
        self.latent = torch.zeros(1, 16, 1, 4, 4)
        self.x = torch.ones(1, 1, 2, 2, 2)
        self.left_mask = torch.tensor([[[1., 1., 0., 0.]] * 4])
        self.right_mask = torch.tensor([[[0., 0., 1., 1.]] * 4])

    def state(self, multiplier=1):
        return {
            'diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight': torch.ones(1, 2),
            'diffusion_model.blocks.0.self_attn.q_proj.lora_up.weight': torch.full((2, 1), float(multiplier)),
            'net.blocks.0.cross_attn.q_proj.lora_A.weight': torch.ones(1, 2),
            'net.blocks.0.cross_attn.q_proj.lora_B.weight': torch.full((2, 1), float(multiplier)),
            'blocks.0.mlp.layer1.lora_down.weight': torch.ones(1, 2),
            'blocks.0.mlp.layer1.lora_up.weight': torch.full((2, 1), float(multiplier)),
            # Never patch contextual keys/values, which have text tokens.
            'blocks.0.cross_attn.k_proj.lora_down.weight': torch.ones(1, 2),
            'blocks.0.cross_attn.k_proj.lora_up.weight': torch.ones(2, 1),
        }

    def test_auto_detect_and_no_cosmos_false_positive(self):
        self.assertEqual(resolve_backend(self.patcher).backend_id, 'anima_cosmos2')
        generic = ToyPatcher(AnimaToy())
        del generic.model.diffusion_model.llm_adapter
        with self.assertRaisesRegex(TypeError, 'no compatible backend'):
            resolve_backend(generic)

    def test_extract_dotted_and_flattened_formats(self):
        adapters, ignored = extract_anima_adapters(self.state())
        self.assertEqual(len(adapters), 3)
        self.assertEqual(len(ignored), 1)
        self.assertEqual({a.target for a in adapters}, {'blocks.0.self_attn.q_proj',
                        'blocks.0.cross_attn.q_proj', 'blocks.0.mlp.layer1'})
        s = {'lora_unet_blocks_0_self_attn_q_proj.lora_down.weight': torch.ones(1, 2),
             'lora_unet_blocks_0_self_attn_q_proj.lora_up.weight': torch.ones(2, 1)}
        v, _ = extract_anima_adapters(s)
        self.assertEqual(v[0].target, 'blocks.0.self_attn.q_proj')

    def test_single_frame_gate_preserves_grid_in_attention_and_mlp(self):
        flat = anima_mask_gate(self.left_mask, self.latent, torch.ones(1, 4, 2), 2)
        self.assertEqual(flat.flatten().tolist(), [1., 0., 1., 0.])
        spatial = anima_mask_gate(self.left_mask, self.latent, self.x, 2)
        self.assertEqual(tuple(spatial.shape), (1, 1, 2, 2, 1))
        self.assertEqual(spatial.flatten().tolist(), [1., 0., 1., 0.])

    def test_two_loras_are_spatial_and_remove_hooks(self):
        a, _ = extract_anima_adapters(self.state(1))
        b, _ = extract_anima_adapters(self.state(2))
        r1 = SpatialRegion(tuple(a), self.left_mask, 1, 'left')
        r2 = SpatialRegion(tuple(b), self.right_mask, 1, 'right')
        before = self.net(self.x)
        with temporary_anima_hooks(self.net, (r1, r2), self.latent):
            during = self.net(self.x)
        self.assertTrue(torch.allclose(before, torch.zeros_like(before)))
        self.assertTrue(torch.allclose(during[0, 0], torch.full((2,), 6.)))
        self.assertTrue(torch.allclose(during[0, 1], torch.full((2,), 12.)))
        self.assertTrue(torch.allclose(during[0, 2], torch.full((2,), 6.)))
        self.assertTrue(torch.allclose(during[0, 3], torch.full((2,), 12.)))
        self.assertTrue(torch.allclose(self.net(self.x), before))
        self.assertEqual(len(self.net.blocks[0].self_attn.q_proj._forward_hooks), 0)

    def test_video_frame_rejected(self):
        a, _ = extract_anima_adapters(self.state())
        with self.assertRaisesRegex(ValueError, 'image latent only'):
            with temporary_anima_hooks(self.net, (SpatialRegion(tuple(a), self.left_mask, 1),),
                                       torch.zeros(1, 16, 2, 4, 4)):
                pass

    def test_exception_cleans_hooks(self):
        a, _ = extract_anima_adapters(self.state())
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with temporary_anima_hooks(self.net, (SpatialRegion(tuple(a), self.left_mask, 1),), self.latent):
                self.net(self.x)
                raise RuntimeError('intentional')
        self.assertEqual(len(self.net.blocks[0].self_attn.q_proj._forward_hooks), 0)

    def test_unified_node_accepts_anima_without_modifying_zimage_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_file = pathlib.Path(tmp) / 'anima.safetensors'
            fake_file.write_bytes(b'test')
            folder_paths = types.ModuleType('folder_paths')
            folder_paths.get_full_path_or_raise = lambda cat, name: str(fake_file)
            folder_paths.get_filename_list = lambda cat: ['anima.safetensors']
            comfy = types.ModuleType('comfy'); comfy.__path__ = []
            utils = types.ModuleType('comfy.utils')
            utils.load_torch_file = lambda path, safe_load=True: self.state()
            comfy.utils = utils
            with mock.patch.dict(sys.modules, {'folder_paths': folder_paths, 'comfy': comfy, 'comfy.utils': utils}):
                layout = json.dumps({'canvas': {'width': 4, 'height': 4}, 'regions': [
                    {'id': 1, 'name': 'left', 'enabled': True, 'x': 0, 'y': 0, 'w': 2, 'h': 4, 'feather': 0}]})
                # parse_layout enforces minimum editor canvas 64, so use scaled coordinates.
                layout = json.dumps({'canvas': {'width': 64, 'height': 64}, 'regions': [
                    {'id': 1, 'name': 'left', 'enabled': True, 'x': 0, 'y': 0, 'w': 32, 'h': 64, 'feather': 0}]})
                node = SpatialMultiLoRA()
                m, preview, mask, _, _ = node.apply(self.patcher, 64, 64, 'auto', layout,
                    'anima.safetensors', 1., 'None', 1., 'None', 1.)
                self.assertEqual(m.model_options['model_function_wrapper'].__class__.__name__, 'AnimaModelWrapper')
                self.assertEqual(tuple(mask.shape), (1, 64, 64))
                self.assertEqual(tuple(preview.shape), (1, 64, 64, 3))

if __name__ == '__main__':
    unittest.main()
