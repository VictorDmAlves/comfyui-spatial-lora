"""Transient reference-image previews for the editor (never stored in workflows).

Images supplied through IMAGE inputs are available to the frontend after the
node has executed. Direct Load Image input previews are resolved in the editor
before execution using ComfyUI's normal /view endpoint.
"""
from collections import OrderedDict
from io import BytesIO
from threading import Lock
import re

import torch
import torch.nn.functional as F
from PIL import Image

_CACHE = OrderedDict()
_LOCK = Lock()
_MAX_ENTRIES = 12
_MAX_PREVIEW_SIDE = 1024
_NODE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_ROUTE_REGISTERED = False


def _key(node_id):
    text = str(node_id) if node_id is not None else ""
    return text if _NODE_ID_RE.fullmatch(text) else None


def make_reference_png(image):
    """First frame of IMAGE [B,H,W,C], safely downsized; no changes to input."""
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1:
        raise ValueError("Spatial LoRA reference must be a nonempty ComfyUI IMAGE tensor")
    if image.shape[-1] not in (1, 3, 4) or min(image.shape[1:3]) <= 0:
        raise ValueError("Spatial LoRA reference IMAGE requires 1, 3 or 4 channels")
    frame = image[0].detach()
    height, width = frame.shape[:2]
    max_side = max(height, width)
    if max_side > _MAX_PREVIEW_SIDE:
        scale = _MAX_PREVIEW_SIDE / max_side
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        frame = F.interpolate(frame.permute(2, 0, 1)[None].float(), size=size,
                              mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    pixels = (frame.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    if pixels.shape[-1] == 1:
        pil = Image.fromarray(pixels[..., 0], mode="L")
    elif pixels.shape[-1] == 4:
        pil = Image.fromarray(pixels, mode="RGBA")
    else:
        pil = Image.fromarray(pixels, mode="RGB")
    out = BytesIO()
    pil.save(out, format="PNG")
    return out.getvalue()


def cache_reference(node_id, image):
    """Cache a thumbnail for a node from its latest execution; bounded in RAM."""
    key = _key(node_id)
    if key is None:
        return
    data = make_reference_png(image) if image is not None else None
    with _LOCK:
        _CACHE.pop(key, None)
        if data is not None:
            _CACHE[key] = data
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)


def get_reference(node_id):
    key = _key(node_id)
    if key is None:
        return None
    with _LOCK:
        value = _CACHE.get(key)
        if value is not None:
            _CACHE.move_to_end(key)
        return value


def register_reference_route():
    """Register endpoint only in actual ComfyUI; keep unit tests importable."""
    global _ROUTE_REGISTERED
    if _ROUTE_REGISTERED:
        return True
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        return False
    if getattr(PromptServer, "instance", None) is None:
        return False

    @PromptServer.instance.routes.get("/spatial_lora/reference/{node_id}")
    async def _get_reference(request):
        data = get_reference(request.match_info.get("node_id"))
        if data is None:
            return web.Response(status=404, text="No reference image from this node execution")
        return web.Response(body=data, content_type="image/png", headers={"Cache-Control": "no-store"})

    _ROUTE_REGISTERED = True
    return True
