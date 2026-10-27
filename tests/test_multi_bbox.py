"""Standalone CPU tests of v0.5 mask generator. No running ComfyUI needed."""
import importlib.util
import json
import pathlib
import sys
import unittest

import torch
import torch.nn.functional as F

root = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("multi_bbox", root / "multi_bbox.py")
module = importlib.util.module_from_spec(spec)
sys.modules["multi_bbox"] = module
spec.loader.exec_module(module)


class MultiBBoxTests(unittest.TestCase):
    def test_regions_unique_slots(self):
        r = module.parse_regions(json.dumps([
            {"id": 3, "enabled": True, "x": 8, "y": 10, "w": 4, "h": 8, "feather": 3},
        ]), 64, 64)
        self.assertEqual([i["id"] for i in r], [1, 2, 3])
        self.assertFalse(r[0]["enabled"])
        self.assertTrue(r[2]["enabled"])
        self.assertEqual(r[2]["feather"], 3)

    def test_reject_duplicate_and_bad_json(self):
        for raw in ("broken", "{}", '[{"id":1},{"id":1}]', '[{"id":5}]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                module.parse_regions(raw, 64, 64)

    def test_mask_edges_match_padded_average_pool(self):
        for w, h, box, feather in [
            (16, 14, (0, 0, 6, 7), 0),
            (16, 14, (0, 0, 6, 7), 3),
            (16, 14, (5, 4, 11, 10), 4),
            (16, 14, (0, 0, 16, 14), 6),
            (13, 12, (11, 10, 2, 2), 4),
            (16, 14, (3, 2, 4, 2), 12),
        ]:
            with self.subTest(box=box, feather=feather):
                x, y, bw, bh = box
                expected = torch.zeros((1, h, w))
                expected[:, y:y+bh, x:x+bw] = 1
                if feather:
                    padded = F.pad(expected.unsqueeze(1), (feather,) * 4, mode="replicate")
                    expected = F.avg_pool2d(padded, 2*feather + 1, stride=1).squeeze(1)
                actual = module.make_mask(w, h, {"enabled": True, "x": x, "y": y,
                    "w": bw, "h": bh, "feather": feather})
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_outputs_and_preview_colors(self):
        first = {"id": 1, "enabled": True, "x": 0, "y": 0, "w": 16, "h": 32, "feather": 0}
        second = {"id": 2, "enabled": True, "x": 16, "y": 0, "w": 16, "h": 32, "feather": 0}
        masks = module.ZImageSpatialMultiBBox().create(64, 64, json.dumps([first, second]))
        self.assertEqual(len(masks), 4)
        self.assertEqual([tuple(v.shape) for v in masks[:3]], [(1, 64, 64)] * 3)
        self.assertEqual(tuple(masks[3].shape), (1, 64, 64, 3))
        self.assertEqual(masks[0][0, 4, 4].item(), 1.0)
        self.assertEqual(masks[1][0, 4, 4].item(), 0.0)
        self.assertEqual(masks[1][0, 4, 20].item(), 1.0)
        self.assertEqual(masks[2].sum().item(), 0.0)
        self.assertTrue(torch.allclose(masks[3][0, 4, 4], torch.tensor(module.REGION_COLORS[0])))

    def test_overlap_is_not_clipped(self):
        regions = [{"id": 1, "enabled": True, "x": 0, "y": 0, "w": 32, "h": 32, "feather": 0},
                   {"id": 2, "enabled": True, "x": 16, "y": 16, "w": 32, "h": 32, "feather": 0}]
        a, b, _, preview = module.ZImageSpatialMultiBBox().create(64, 64, json.dumps(regions))
        self.assertEqual((a[0,20,20].item(), b[0,20,20].item()), (1., 1.))
        self.assertTrue((preview <= 1).all().item())


if __name__ == "__main__": unittest.main()
