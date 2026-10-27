"""Shared multi-box mask implementation for Z-Image Spatial LoRA v0.5.

This node DOES NOT change model patching: it only produces up to three MASKs
and a diagnostic RGB IMAGE, retaining the proven v0.3 multi-LoRA sampler path.
"""
import json
import math

import torch

MAX_REGIONS = 3
REGION_COLORS = ((0.32, 0.66, 1.0), (0.37, 0.91, 0.57), (1.0, 0.68, 0.28))
DEFAULT_REGIONS = [
    {"id": 1, "name": "Region 1", "enabled": True, "x": 90, "y": 90,
     "w": 380, "h": 720, "feather": 8},
    {"id": 2, "name": "Region 2", "enabled": False, "x": 550, "y": 90,
     "w": 380, "h": 720, "feather": 8},
    {"id": 3, "name": "Region 3", "enabled": False, "x": 350, "y": 90,
     "w": 300, "h": 720, "feather": 8},
]
DEFAULT_REGIONS_JSON = json.dumps(DEFAULT_REGIONS, ensure_ascii=False, separators=(",", ":"))


def _int_field(value, fallback):
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return fallback
    return round(value) if math.isfinite(value) else fallback


def parse_regions(regions_json, width, height):
    """Validate and normalize regions; each id maps to a fixed output slot."""
    if not isinstance(regions_json, str) or len(regions_json) > 16384:
        raise ValueError("regions_json must be a JSON string of at most 16384 characters")
    try:
        raw = json.loads(regions_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid regions_json: {exc}") from exc
    if not isinstance(raw, list) or len(raw) > MAX_REGIONS:
        raise ValueError("regions_json must contain at most 3 region objects")
    result = {i: dict(DEFAULT_REGIONS[i - 1], enabled=False) for i in range(1, 4)}
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each region must be a JSON object")
        identity = _int_field(item.get("id"), -1)
        if identity not in (1, 2, 3) or identity in seen:
            raise ValueError("Region id must be unique and within 1..3")
        seen.add(identity)
        x = max(0, min(_int_field(item.get("x"), 0), width - 1))
        y = max(0, min(_int_field(item.get("y"), 0), height - 1))
        w = max(1, min(_int_field(item.get("w"), 1), width - x))
        h = max(1, min(_int_field(item.get("h"), 1), height - y))
        result[identity] = {
            "id": identity,
            "name": str(item.get("name", f"Region {identity}"))[:64],
            "enabled": item.get("enabled", True) is True,
            "x": x, "y": y, "w": w, "h": h,
            "feather": max(0, min(_int_field(item.get("feather"), 8), 128)),
        }
    return [result[i] for i in range(1, 4)]


def _axis_weights(size, start, extent, feather):
    """Analytic 1-D replicate-pad box average; O(size), not O(kernel area)."""
    pixels = torch.arange(size, dtype=torch.float32)
    if feather == 0:
        return ((pixels >= start) & (pixels < start + extent)).float()
    low = pixels - feather
    high = pixels + feather
    inside = (torch.minimum(high, torch.tensor(float(start + extent - 1)))
              - torch.maximum(torch.maximum(low, torch.zeros_like(low)), torch.tensor(float(start))) + 1).clamp_min_(0)
    if start == 0:
        inside += (-low).clamp_min(0)
    if start + extent >= size:
        inside += (high - (size - 1)).clamp_min(0)
    return inside / (2 * feather + 1)


def make_mask(width, height, region):
    """Exact replicate-padded rectangular feather, computed as two 1-D axes."""
    if not region["enabled"]:
        return torch.zeros((1, height, width), dtype=torch.float32)
    xs = _axis_weights(width, region["x"], region["w"], region["feather"])
    ys = _axis_weights(height, region["y"], region["h"], region["feather"])
    return (ys[:, None] * xs[None, :]).unsqueeze(0)


def preview_image(masks):
    """RGB diagnostic overlay: color intensity represents mask strength.

    Colors are fixed to slot 1=blue, 2=green, 3=amber. This is NOT an
    image-generation preview and does not represent attention leakage.
    """
    height, width = masks[0].shape[-2:]
    preview = torch.zeros((height, width, 3), dtype=torch.float32)
    for mask, rgb in zip(masks, REGION_COLORS):
        preview += mask[0, :, :, None] * torch.tensor(rgb, dtype=torch.float32)
    return preview.clamp_(0, 1).unsqueeze(0)


class ZImageSpatialMultiBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
            "regions_json": ("STRING", {"default": DEFAULT_REGIONS_JSON, "multiline": False}),
        }}

    RETURN_TYPES = ("MASK", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("mask_1", "mask_2", "mask_3", "preview")
    FUNCTION = "create"
    CATEGORY = "Spatial LoRA"

    def create(self, width, height, regions_json):
        regions = parse_regions(regions_json, width, height)
        masks = tuple(make_mask(width, height, region) for region in regions)
        return (*masks, preview_image(masks))
