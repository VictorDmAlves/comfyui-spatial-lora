"""v0.6 CPU end-to-end: unified node parity with proven legacy chaining."""
import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

from test_multi import ToyModel, ToyPatcher

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAME = "spatial_unified_test_plugin"
if NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(NAME, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = package
    spec.loader.exec_module(package)
from spatial_unified_test_plugin import SpatialMultiLoRA, nodes
from spatial_unified_test_plugin.multi_lora import parse_layout
from spatial_unified_test_plugin.backends import resolve_backend


def layout(width=128, height=64, regions=None):
    if regions is None:
        regions = [
            {"id": 1, "name": "Left", "enabled": True, "x": 0, "y": 0, "w": 64, "h": 64, "feather": 0},
            {"id": 2, "name": "Right", "enabled": True, "x": 64, "y": 0, "w": 64, "h": 64, "feather": 0},
        ]
    return json.dumps({"canvas": {"width": width, "height": height}, "regions": regions})


class UnifiedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for i in range(3):
            (pathlib.Path(self.tmp.name) / f"char{i}.safetensors").write_bytes(b"file")
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_filename_list = lambda category: [f"char{i}.safetensors" for i in range(3)]
        folder_paths.get_full_path_or_raise = lambda cat, name: str(pathlib.Path(self.tmp.name) / name)
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        utils = types.ModuleType("comfy.utils")
        self.reads = []
        def load_torch_file(filename, safe_load=True):
            self.reads.append(filename)
            num = int(pathlib.Path(filename).stem[-1]) + 1
            return {
                "diffusion_model.layers.0.attention.qkv.lora_down.weight": torch.ones(1, 2),
                "diffusion_model.layers.0.attention.qkv.lora_up.weight": torch.full((6, 1), float(num)),
            }
        utils.load_torch_file = load_torch_file
        comfy.utils = utils
        self.patch = mock.patch.dict(sys.modules, {"folder_paths": folder_paths, "comfy": comfy, "comfy.utils": utils})
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.base = ToyPatcher(ToyModel())
        self.x = torch.ones(1, 4, 2)
        self.main = SpatialMultiLoRA()

    def run_main(self, geometry=None, width=128, height=64, loras=None, strengths=(1, 1, 1), backend="auto"):
        if loras is None:
            loras = ("char0.safetensors", "char1.safetensors", "None")
        return self.main.apply(self.base, width, height, backend,
                               geometry if geometry is not None else layout(),
                               loras[0], strengths[0], loras[1], strengths[1], loras[2], strengths[2])

    def test_unified_matches_legacy_two_lora_and_keeps_text_unmodified(self):
        model, image, first, second, third = self.run_main()
        self.assertEqual(image.shape, (1, 64, 128, 3))
        self.assertEqual((first[0,0,0].item(), first[0,0,127].item()), (1., 0.))
        self.assertEqual((second[0,0,0].item(), second[0,0,127].item()), (0., 1.))
        self.assertEqual(third.sum().item(), 0)
        out = model.run(self.x)
        self.assertTrue(torch.allclose(out[0, :2], torch.zeros(2, 6)))
        self.assertTrue(torch.allclose(out[0, 2], torch.ones(6)*2))
        self.assertTrue(torch.allclose(out[0, 3], torch.ones(6)*4))
        legacy = nodes.ZImageSpatialLora().apply(self.base, "char0.safetensors", first, 1)[0]
        legacy = nodes.ZImageSpatialLora().apply(legacy, "char1.safetensors", second, 1)[0]
        self.assertTrue(torch.allclose(legacy.run(self.x), out))
        self.assertEqual(len(self.base.model.diffusion_model.layers[0].attention.qkv._forward_hooks), 0)

    def test_three_regions_all_work_and_loader_caches(self):
        config = layout(regions=[
            {"id":1,"enabled":True,"x":0,"y":0,"w":64,"h":64,"feather":0},
            {"id":2,"enabled":True,"x":64,"y":0,"w":64,"h":64,"feather":0},
            {"id":3,"enabled":True,"x":0,"y":0,"w":64,"h":64,"feather":0},
        ])
        model, _, *_ = self.run_main(config, loras=("char0.safetensors", "char1.safetensors", "char2.safetensors"))
        out = model.run(self.x)
        self.assertTrue(torch.allclose(out[0, 2], torch.ones(6)*8))
        self.assertTrue(torch.allclose(out[0, 3], torch.ones(6)*4))
        self.run_main(config, loras=("char0.safetensors", "char1.safetensors", "char2.safetensors"))
        self.assertEqual(len(self.reads), 3, "Each adapter should be loaded once")

    def test_disabled_region_may_have_no_lora(self):
        model, _, *_ = self.run_main()
        self.assertEqual(len(model.model_options["model_function_wrapper"].regions), 2)

    def test_missing_enabled_lora_rejected(self):
        with self.assertRaisesRegex(ValueError, "Region 1.*no LoRA"):
            self.run_main(loras=("None", "char1.safetensors", "None"))

    def test_backend_detection_fails_closed(self):
        model = ToyPatcher(types.SimpleNamespace())
        with self.assertRaisesRegex(TypeError, "no compatible backend"):
            resolve_backend(model)
        with self.assertRaisesRegex(TypeError, "does not support"):
            resolve_backend(model, "zimage_nextdit")

    def test_runtime_scaling_and_shared_border_are_seamless(self):
        regions = parse_layout(layout(width=1024, height=1024, regions=[
            {"id":1,"enabled":True,"x":0,"y":100,"w":512,"h":924,"feather":0},
            {"id":2,"enabled":True,"x":512,"y":100,"w":512,"h":924,"feather":0},
        ]), 1216, 832)
        self.assertEqual(regions[0]["x"] + regions[0]["w"], regions[1]["x"])
        self.assertEqual(regions[0]["w"], 608)
        self.assertEqual(regions[1]["x"], 608)
        self.assertEqual(regions[0]["y"], 81)
        self.assertEqual(regions[0]["h"], 751)

    def test_unified_equals_v05_mask_at_same_resolution_with_feather(self):
        from spatial_unified_test_plugin.multi_bbox import make_mask
        configs = [{"id":1,"enabled":True,"x":100,"y":110,"w":300,"h":400,"feather":16}]
        geometry = layout(1024, 768, configs)
        actual = parse_layout(geometry, 1024, 768)
        old = make_mask(1024,768,configs[0])
        self.assertTrue(torch.allclose(make_mask(1024,768,actual[0]), old, atol=1e-6))

    def test_invalid_layout_cannot_silently_fall_back(self):
        for invalid in ("", "{}", '{"canvas":{"width":0,"height":1024},"regions":[]}', "{bad"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_layout(invalid, 1024, 768)

    def test_diagnostics_report_backend_masks_and_strengths(self):
        with self.assertLogs("ComfyUI-Spatial-LoRA", level="INFO") as captured:
            self.run_main(strengths=(0.85, 1.0, 1.0))
        all_text = "\n".join(captured.output)
        self.assertIn("Backend: Z-Image Native NextDiT", all_text)
        self.assertIn("Resolution: 128 x 64", all_text)
        self.assertIn("Active regions: 2", all_text)
        self.assertIn("strength: 0.85", all_text)
        self.assertIn("mask coverage: 50.00%", all_text)
        self.assertNotIn("overlap (weighted", all_text)
        self.assertNotIn(self.tmp.name, all_text, "Do not leak absolute paths into diagnostics")

    def test_diagnostics_warn_on_active_mask_overlap(self):
        config = layout(regions=[
            {"id": 1, "name": "First", "enabled": True, "x": 0, "y": 0, "w": 70, "h": 64, "feather": 0},
            {"id": 2, "name": "Second", "enabled": True, "x": 60, "y": 0, "w": 68, "h": 64, "feather": 0},
        ])
        with self.assertLogs("ComfyUI-Spatial-LoRA", level="WARNING") as captured:
            self.run_main(config)
        self.assertIn("Regions 1/2 overlap", "\n".join(captured.output))

    def test_hook_lifecycle_logs_only_once_and_cleans_after_each_forward(self):
        model, *_ = self.run_main()
        with self.assertLogs("ComfyUI-Spatial-LoRA", level="INFO") as captured:
            out1 = model.run(self.x)
            out2 = model.run(self.x)
        self.assertTrue(torch.allclose(out1, out2))
        logs = "\n".join(captured.output)
        self.assertEqual(logs.count("Sampling hooks installed"), 1)
        self.assertEqual(logs.count("Sampling hooks removed cleanly"), 1)
        self.assertEqual(len(self.base.model.diffusion_model.layers[0].attention.qkv._forward_hooks), 0)

    def test_error_from_previous_wrapper_cleans_hooks_and_warns(self):
        def fail_after_forward(apply_model, args):
            apply_model(args["input"], args["timestep"], **args["c"])
            raise RuntimeError("simulated sampling error")
        self.base.set_model_unet_function_wrapper(fail_after_forward)
        model, *_ = self.run_main()
        with self.assertLogs("ComfyUI-Spatial-LoRA", level="WARNING") as captured:
            with self.assertRaisesRegex(RuntimeError, "simulated sampling error"):
                model.run(self.x)
        self.assertIn("temporary hooks cleaned", "\n".join(captured.output))
        self.assertEqual(len(self.base.model.diffusion_model.layers[0].attention.qkv._forward_hooks), 0)

    def test_disabled_lora_never_reported_as_active(self):
        config = layout(regions=[
            {"id": 1, "name": "Only", "enabled": True, "x": 0, "y": 0, "w": 64, "h": 64, "feather": 0},
            {"id": 2, "name": "Disabled", "enabled": False, "x": 64, "y": 0, "w": 64, "h": 64, "feather": 0},
        ])
        with self.assertLogs("ComfyUI-Spatial-LoRA", level="INFO") as captured:
            self.run_main(config)
        logs = "\n".join(captured.output)
        self.assertIn("Active regions: 1", logs)
        self.assertNotIn("Region 2 |", logs)

    def test_only_unified_node_registered(self):
        from spatial_unified_test_plugin import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        self.assertEqual(set(NODE_CLASS_MAPPINGS), {"SpatialMultiLoRA"})
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["SpatialMultiLoRA"], "Spatial LoRA | Multi LoRA")

if __name__ == "__main__":
    unittest.main()
