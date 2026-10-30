"""Single-node multi-region multi-LoRA compositor (Z-Image NextDiT backend)."""
import json
import math

from .adapter_loader import AdapterCache
from .diagnostics import log, report_regions
from .backends import resolve_backend
from .multi_bbox import DEFAULT_REGIONS, make_mask, parse_regions, preview_image
from .spatial import SpatialRegion
from .reference_cache import cache_reference

# Save editor geometry to make coordinates resolution-independent across linked
# runtime width/height changes. Legacy MultiBBox retains absolute coordinates.
DEFAULT_LAYOUT_JSON = json.dumps({
    "canvas": {"width": 1024, "height": 1024},
    "regions": DEFAULT_REGIONS,
}, ensure_ascii=False, separators=(",", ":"))


def _scale(value, source, destination):
    return int(round(value * destination / source))


def parse_layout(layout_json, width, height):
    """Return backend masks expressed in actual runtime pixel coordinates.

    v0.6 layouts store the editor coordinate system (canvas). The runtime may
    receive different dimensions from a linked latent size picker; scale axes
    independently before clamping to the output grid. Old array-format v0.5
    configs are accepted as direct pixel coordinates for import/migration.
    """
    if not isinstance(layout_json, str) or len(layout_json) > 16384:
        raise ValueError("Spatial LoRA layout must be a JSON string <= 16384 chars")
    try:
        data = json.loads(layout_json)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid Spatial LoRA layout JSON: {exc}") from exc
    if isinstance(data, list):
        return parse_regions(layout_json, width, height)
    if not isinstance(data, dict) or not isinstance(data.get("regions"), list):
        raise ValueError("Spatial LoRA layout requires an array of regions")
    canvas = data.get("canvas") or {}
    try:
        source_width = int(canvas["width"])
        source_height = int(canvas["height"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Spatial LoRA layout missing valid canvas width/height") from exc
    if not (64 <= source_width <= 8192 and 64 <= source_height <= 8192):
        raise ValueError("Spatial LoRA editor canvas dimensions must be 64..8192")
    scaled = []
    for item in data["regions"]:
        if not isinstance(item, dict):
            raise ValueError("Spatial LoRA region must be an object")
        r = dict(item)
        # Round edges, not width independently, to avoid off-by-one gaps when
        # boxes share a boundary (e.g., A right edge == B left edge).
        for axis, extent, src, dst in (("x", "w", source_width, width),
                                       ("y", "h", source_height, height)):
            left = r.get(axis, 0)
            length = r.get(extent, 1)
            if not isinstance(left, (int, float)) or not isinstance(length, (int, float)) or not math.isfinite(left) or not math.isfinite(length):
                raise ValueError("Spatial LoRA region coordinates must be finite numbers")
            start = _scale(left, src, dst)
            end = _scale(left + length, src, dst)
            r[axis] = start
            r[extent] = max(1, end - start)
        # The feather is a width in image pixels, not a relative strength.
        # Scale it proportionally using the geometric mean of the image scales.
        feather = float(r.get("feather", 0))
        if not math.isfinite(feather):
            raise ValueError("Spatial LoRA feather must be finite")
        r["feather"] = round(feather * math.sqrt((width / source_width) * (height / source_height)))
        scaled.append(r)
    return parse_regions(json.dumps(scaled), width, height)


class SpatialMultiLoRA:
    """A single MODEL->MODEL node with 1..3 editable region/LoRA pairs."""

    def __init__(self):
        self.adapter_cache = AdapterCache()

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        names = ["None"] + [n for n in folder_paths.get_filename_list("loras") if n != "None"]
        required = {
            "model": ("MODEL",),
            "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
            "backend": (["auto", "zimage_nextdit", "anima_cosmos2"], {"default": "auto"}),
            "regions_json": ("STRING", {"default": DEFAULT_LAYOUT_JSON, "multiline": False}),
        }
        for i in (1, 2, 3):
            required[f"lora_{i}"] = (names, {"default": "None"})
            required[f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -3.0, "max": 3.0, "step": 0.05})
        return {"required": required,
                "optional": {"reference_image": ("IMAGE",)},
                "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("MODEL", "IMAGE", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("model", "preview", "mask_1", "mask_2", "mask_3")
    FUNCTION = "apply"
    CATEGORY = "Spatial LoRA"

    def apply(self, model, width, height, backend, regions_json,
              lora_1, strength_1, lora_2, strength_2, lora_3, strength_3,
              reference_image=None, unique_id=None):
        # The optional IMAGE is editor-only: it never changes conditioning or LoRA weights.
        # Cache only a bounded thumbnail for reference after execution; direct Load Image
        # preview is independently available in the editor without any execution.
        cache_reference(unique_id, reference_image)
        # Resolve before loading LoRA files, so unsupported models fail early.
        resolved = resolve_backend(model, backend)
        configs = parse_layout(regions_json, width, height)
        masks = tuple(make_mask(width, height, r) for r in configs)
        image = preview_image(masks)
        names = (lora_1, lora_2, lora_3)
        strengths = (strength_1, strength_2, strength_3)
        active = []
        active_slots = []
        for idx, (config, mask, name, strength) in enumerate(zip(configs, masks, names, strengths), 1):
            if not config["enabled"]:
                continue
            if not isinstance(name, str) or not name.strip() or name == "None":
                raise ValueError(f"Spatial LoRA: Region {idx} ({config['name']}) enabled but no LoRA selected")
            if strength == 0:
                continue
            adapters = self.adapter_cache.get(name, resolved.backend_id)
            active.append(SpatialRegion(adapters, mask, float(strength), f"{config['name']} : {name}"))
            active_slots.append(idx)
        result = resolved.attach(model, active)
        try:
            layout_info = json.loads(regions_json)
            canvas = layout_info.get("canvas") if isinstance(layout_info, dict) else None
            canvas_size = (int(canvas["width"]), int(canvas["height"])) if isinstance(canvas, dict) else None
        except (ValueError, TypeError, KeyError):
            canvas_size = None
        report_regions(backend=resolved.display_name, width=width, height=height,
                       canvas_size=canvas_size, configs=configs, masks=masks,
                       names=names, strengths=strengths, active_slots=active_slots)
        log.debug("MODEL wrapper ready: %d active regions", len(active))
        return (result, image, *masks)
