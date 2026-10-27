"""CPU integration tests for 2-3 chained MODEL nodes. No ComfyUI installation required."""
import copy
import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Load plugin as a real Python package to test its node implementations.
PACKAGE_NAME = "spatial_test_plugin_v03"
if PACKAGE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)

from spatial_test_plugin_v03 import nodes, spatial


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(2, 6, bias=False)
        with torch.no_grad():
            self.qkv.weight.zero_()


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = ToyAttention()


class ToyModel(nn.Module):
    patch_size = 2

    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(1, 2)
        self.layers = nn.ModuleList([ToyBlock()])

    def forward(self, x):
        return self.layers[0].attention.qkv(x)


class ToyPatcher:
    def __init__(self, model):
        self.model = types.SimpleNamespace(diffusion_model=model)
        self.model_options = {}

    def clone(self):
        new = ToyPatcher(self.model.diffusion_model)
        # Mimic ComfyUI ModelPatcher.clone: options are copied, model object shared.
        new.model_options = self.model_options.copy()
        return new

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn

    def run(self, x, latent=None):
        if latent is None:
            latent = torch.zeros(1, 1, 2, 4)
        args = {"input": latent, "timestep": torch.tensor([1]), "c": {}}
        def apply_model(_latent, _timestep, **kwargs):
            return self.model.diffusion_model(x)
        wrapper = self.model_options.get("model_function_wrapper")
        return wrapper(apply_model, args) if wrapper else apply_model(latent, args["timestep"])


class MultiNodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = [pathlib.Path(self.tmp.name) / f"char{i}.safetensors" for i in range(4)]
        for path in self.paths:
            path.write_bytes(b"placeholder for mocked loader")
        self.folder_paths = types.ModuleType("folder_paths")
        self.folder_paths.get_full_path_or_raise = lambda cat, name: str(pathlib.Path(self.tmp.name) / name)
        self.comfy = types.ModuleType("comfy")
        self.comfy.__path__ = []
        self.utils = types.ModuleType("comfy.utils")

        def load_torch_file(filename, safe_load=True):
            num = int(pathlib.Path(filename).stem[-1]) + 1
            return {
                "diffusion_model.layers.0.attention.qkv.lora_down.weight": torch.ones(1, 2),
                "diffusion_model.layers.0.attention.qkv.lora_up.weight": torch.full((6, 1), float(num)),
            }
        self.utils.load_torch_file = load_torch_file
        self.comfy.utils = self.utils
        self.patch = mock.patch.dict(sys.modules, {
            "folder_paths": self.folder_paths,
            "comfy": self.comfy,
            "comfy.utils": self.utils,
        })
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.net = ToyModel()
        self.base = ToyPatcher(self.net)
        self.x = torch.ones(1, 4, 2)
        # 2 text tokens + 2 image tokens. Left and right regions respectively.
        self.left = torch.tensor([[[1., 1., 0., 0.], [1., 1., 0., 0.]]])
        self.right = torch.tensor([[[0., 0., 1., 1.], [0., 0., 1., 1.]]])

    def add(self, previous, i, mask, strength=1.0):
        return nodes.ZImageSpatialLora().apply(previous, self.paths[i].name, mask, strength)[0]

    def test_two_regions_each_modify_only_its_own_image_token(self):
        original = self.base.run(self.x).clone()
        one = self.add(self.base, 0, self.left)
        two = self.add(one, 1, self.right)
        self.assertEqual(len(two.model_options["model_function_wrapper"].regions), 2)
        self.assertIsInstance(two.model_options["model_function_wrapper"], nodes.SpatialModelWrapper)
        # 2*1 on left, 2*2 on right (rank-1 adapter with [1,1] input).
        out = two.run(self.x)
        self.assertTrue(torch.allclose(out[0, 0:2], original[0, 0:2]))
        self.assertTrue(torch.allclose(out[0, 2], torch.full((6,), 2.)))
        self.assertTrue(torch.allclose(out[0, 3], torch.full((6,), 4.)))
        self.assertEqual(len(self.net.layers[0].attention.qkv._forward_hooks), 0)
        # Earlier branch unchanged; model doesn't retain patches after sampling.
        self.assertEqual(len(one.model_options["model_function_wrapper"].regions), 1)
        self.assertTrue(torch.allclose(self.base.run(self.x), original))

    def test_overlap_adds_adapter_contributions_not_overwriting_previous_one(self):
        one = self.add(self.base, 0, self.left)
        two = self.add(one, 1, self.left)
        out = two.run(self.x)
        self.assertTrue(torch.allclose(out[0, 2], torch.full((6,), 6.)))
        self.assertTrue(torch.allclose(out[0, 3], torch.zeros(6)))

    def test_three_regions_supported_fourth_rejected(self):
        model = self.base
        for i, mask in enumerate((self.left, self.right, self.left)):
            model = self.add(model, i, mask)
        self.assertEqual(len(model.model_options["model_function_wrapper"].regions), 3)
        self.assertTrue(torch.allclose(model.run(self.x)[0, 2], torch.full((6,), 8.)))
        with self.assertRaisesRegex(ValueError, "Limite experimental"):
            self.add(model, 3, self.right)

    def test_other_model_wrapper_called_once_with_all_regions(self):
        calls = []
        def earlier(apply_model, args):
            calls.append(1)
            return apply_model(args["input"], args["timestep"], **args["c"])
        self.base.set_model_unet_function_wrapper(earlier)
        model = self.add(self.add(self.base, 0, self.left), 1, self.right)
        result = model.run(self.x)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result[0, 2].tolist(), [2.] * 6)

    def test_hooks_removed_when_downstream_wrapper_raises(self):
        def failing(apply_model, args):
            apply_model(args["input"], args["timestep"], **args["c"])
            raise RuntimeError("downstream failed")
        self.base.set_model_unet_function_wrapper(failing)
        model = self.add(self.add(self.base, 0, self.left), 1, self.right)
        with self.assertRaisesRegex(RuntimeError, "downstream failed"):
            model.run(self.x)
        self.assertEqual(len(self.net.layers[0].attention.qkv._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
