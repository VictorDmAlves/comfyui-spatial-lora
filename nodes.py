"""Backwards-compatible single-BBox and chained single-LoRA nodes."""
import torch
import torch.nn.functional as F

from .spatial import SpatialRegion
from .adapter_loader import AdapterCache
from .backends import resolve_backend, SpatialModelWrapper


class ZImageSpatialBBoxMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
            "x": ("INT", {"default": 100, "min": 0, "max": 8192}),
            "y": ("INT", {"default": 100, "min": 0, "max": 8192}),
            "box_width": ("INT", {"default": 360, "min": 1, "max": 8192}),
            "box_height": ("INT", {"default": 700, "min": 1, "max": 8192}),
            "feather": ("INT", {"default": 8, "min": 0, "max": 128}),
        }}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create"
    CATEGORY = "Spatial LoRA"

    def create(self, width, height, x, y, box_width, box_height, feather):
        mask = torch.zeros((1, height, width), dtype=torch.float32)
        x0, y0 = max(0, min(x, width)), max(0, min(y, height))
        x1, y1 = max(0, min(x + box_width, width)), max(0, min(y + box_height, height))
        mask[:, y0:y1, x0:x1] = 1.0
        if feather > 0:
            k = 2 * feather + 1
            padded = F.pad(mask.unsqueeze(1), (feather,) * 4, mode="replicate")
            mask = F.avg_pool2d(padded, kernel_size=k, stride=1).squeeze(1)
        return (mask,)


class ZImageSpatialLora:
    def __init__(self):
        self.adapter_cache = AdapterCache()

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "mask": ("MASK",),
            "strength": ("FLOAT", {"default": 0.85, "min": -3.0, "max": 3.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Spatial LoRA"

    def apply(self, model, lora_name, mask, strength):
        if strength == 0 or not bool(torch.any(mask > 0)):
            return (model,)
        backend = resolve_backend(model)
        adapters = self.adapter_cache.get(lora_name)
        region = SpatialRegion(adapters, mask.detach().float().clamp(0, 1).cpu(), float(strength), lora_name)
        result = backend.attach(model, (region,))
        print(f"[Spatial LoRA] Added region: {lora_name}")
        return (result,)
