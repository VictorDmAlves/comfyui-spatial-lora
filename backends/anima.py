"""Experimental Anima 2B/Cosmos-Predict2 spatial LoRA backend (image frames only).

Anima does not share NextDiT's text-prefix token layout. Its image-only DiT
attention projections see [B,T*H*W,D]; MLP sees [B,T,H,W,D]. Cross-attention
K/V and llm_adapter operate on text tokens and deliberately are not patched.

Grounded in ComfyUI comfy/ldm/anima/model.py and cosmos/predict2.py.
This backend needs validation against real Anima LoRA weights and GPU sampling.
"""
from __future__ import annotations

from contextlib import contextmanager
import re

import torch
import torch.nn.functional as F

from ..diagnostics import log
from ..spatial import Adapter

_PAIR = re.compile(r"^(?P<name>.+?)\.(?P<part>lora_down|lora_up|lora_A|lora_B)(?:\.default)?\.weight$")
_TARGET = re.compile(
    r"^blocks\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"cross_attn\.(?:q_proj|o_proj)|mlp\.layer[12])$"
)


def normalize_target(raw):
    name = raw
    for prefix in (
        "base_model.model.", "model.diffusion_model.", "diffusion_model.",
        "model.", "transformer.", "net.",
    ):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name.startswith("net."):
        name = name[4:]
    # Common kohya-style flattened module names, not arbitrary underscore keys.
    match = re.fullmatch(
        r"lora_unet_(?:net_)?blocks_(\d+)_(self_attn|cross_attn)_(q_proj|k_proj|v_proj|o_proj)",
        name,
    )
    if match:
        i, block, projection = match.groups()
        name = f"blocks.{i}.{block}.{projection}"
    match = re.fullmatch(r"lora_unet_(?:net_)?blocks_(\d+)_mlp_(layer1|layer2)", name)
    if match:
        name = f"blocks.{match[1]}.mlp.{match[2]}"
    return name


def extract_anima_adapters(state):
    """Extract spatially meaningful image-token LoRAs; report unsupported keys.

    Refuse models with insufficient compatible spatial coverage, rather than
    silently applying a weak/incomplete LoRA. Global text modules cannot be
    spatially masked and are excluded by design.
    """
    groups = {}
    ignored = []
    for key, tensor in state.items():
        match = _PAIR.match(key)
        if match:
            name = match["name"]
            group = groups.setdefault(name, {})
            part = "down" if match["part"] in ("lora_down", "lora_A") else "up"
            if part in group:
                raise ValueError(f"Anima LoRA duplicate {part} for {name}")
            group[part] = tensor
        elif key.endswith((".alpha", ".lora_alpha")):
            name = key[:-len(".lora_alpha")] if key.endswith(".lora_alpha") else key[:-len(".alpha")]
            groups.setdefault(name, {})["alpha"] = tensor
        elif any(k in key for k in ("lora_magnitude", "dora_scale", "lokr_", "loha_", ".lora_ups.", ".lora_downs.")):
            raise ValueError(f"Unsupported Anima adapter variant (DoRA/LyCORIS/MoE): {key}")
        elif re.search(r"\.(?:lora_A|lora_B|lora_up|lora_down)(?:\.|$)", key):
            raise ValueError(f"Unsupported Anima LoRA key: {key}")
    selected = []
    for raw, parts in groups.items():
        if "up" not in parts or "down" not in parts:
            ignored.append(f"{raw} (incomplete LoRA pair)")
            continue
        target = normalize_target(raw)
        if not _TARGET.fullmatch(target):
            ignored.append(f"{raw} (non-spatial or unknown Anima projection)")
            continue
        down, up = parts["down"], parts["up"]
        if not (torch.is_tensor(down) and torch.is_tensor(up)
                and down.ndim == 2 and up.ndim == 2
                and down.shape[0] == up.shape[1]):
            ignored.append(f"{raw} (invalid linear ranks)")
            continue
        alpha = parts.get("alpha")
        scale = float(alpha.item()) / down.shape[0] if alpha is not None else 1.0
        selected.append(Adapter(target, down.detach().cpu().contiguous(),
                                up.detach().cpu().contiguous(), scale))
    if not selected:
        raise ValueError("No spatial Anima LoRA projections found (expected blocks.N.self_attn / cross_attn / mlp)")
    # Count all LoRA pairs (including unsupported text modules) to warn users
    # about partial coverage. Many standard Anima LoRAs train cross-attn K/V;
    # those projections are deliberately excluded, so 60% is a cautious floor.
    if len(selected) < max(1, len(groups) * 0.6):
        raise ValueError(f"Anima LoRA has insufficient spatial coverage: {len(selected)}/{len(groups)}; "
                         f"examples ignored: {ignored[:3]}")
    return selected, ignored


def anima_mask_gate(mask, latent, x, patch_spatial):
    """Image-token gate for [B,S,D] attention or [B,T,H,W,D] MLP.

    Supported input is one image frame: [B,C,H,W] or [B,C,1,H,W].
    The first pass pads latent dimensions to patches, so ceil division is used.
    """
    if latent.ndim == 4:
        height, width = latent.shape[-2:]
        frames = 1
    elif latent.ndim == 5:
        height, width = latent.shape[-2:]
        frames = latent.shape[-3]
    else:
        raise ValueError("Anima Spatial LoRA expects [B,C,H,W] or [B,C,1,H,W] latent")
    if frames != 1:
        raise ValueError("Anima Spatial LoRA currently supports image only (T=1), not video")
    grid_h = (height + patch_spatial - 1) // patch_spatial
    grid_w = (width + patch_spatial - 1) // patch_spatial
    if mask.ndim != 3:
        raise ValueError("Spatial mask must be [B,H,W]")
    if x.ndim == 3:
        if x.shape[1] != grid_h * grid_w:
            raise ValueError(f"Anima attention sequence mismatch: {x.shape[1]} != {grid_h}*{grid_w}")
        shape = (x.shape[0], grid_h * grid_w, 1)
    elif x.ndim == 5:
        if x.shape[1:4] != (1, grid_h, grid_w):
            raise ValueError(f"Anima MLP token grid mismatch: {tuple(x.shape[1:4])} != (1,{grid_h},{grid_w})")
        shape = (x.shape[0], 1, grid_h, grid_w, 1)
    else:
        raise ValueError(f"Unexpected Anima spatial projection shape: {tuple(x.shape)}")
    flat = F.interpolate(mask.float().unsqueeze(1), size=(grid_h, grid_w),
                         mode="bilinear", align_corners=False).squeeze(1)
    if flat.shape[0] != x.shape[0]:
        if flat.shape[0] == 1:
            flat = flat.expand(x.shape[0], -1, -1)
        elif x.shape[0] % flat.shape[0] == 0:
            flat = flat.repeat(x.shape[0] // flat.shape[0], 1, 1)
        else:
            raise ValueError("Anima mask batch does not match latent batch")
    return flat.reshape(shape).to(x.device, dtype=x.dtype)


def validate_anima_targets(diffusion, adapters):
    named = dict(diffusion.named_modules())
    grouped = {}
    for adapter in adapters:
        if adapter.slice_index is not None or not _TARGET.fullmatch(adapter.target):
            raise ValueError(f"Anima LoRA projection not allowed: {adapter.target}")
        target = named.get(adapter.target)
        weight = getattr(target, "weight", None)
        if weight is None or weight.ndim != 2:
            raise ValueError(f"Anima module missing / not linear: {adapter.target}")
        if tuple(weight.shape) != (adapter.up.shape[0], adapter.down.shape[1]):
            raise ValueError(f"Anima LoRA dimensions don't match module: {adapter.target}")
        grouped.setdefault(adapter.target, []).append(adapter)
    return grouped


@contextmanager
def temporary_anima_hooks(diffusion, regions, latent):
    if latent.ndim not in (4, 5) or (latent.ndim == 5 and latent.shape[-3] != 1):
        raise ValueError("Spatial LoRA Anima experimental: image latent only (T=1)")
    grouped = {}
    active = []
    for region in regions:
        if not torch.any(region.mask > 0) or region.strength == 0:
            continue
        idx = len(active)
        active.append(region)
        for target, adapters in validate_anima_targets(diffusion, region.adapters).items():
            for adapter in adapters:
                grouped.setdefault(target, []).append((idx, adapter, region.strength))
    named = dict(diffusion.named_modules())
    handles = []
    cache = {}
    try:
        for target, assignments in grouped.items():
            def hook(module, inputs, output, *, here=assignments):
                if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                    raise RuntimeError("Anima LoRA hook expected tensor input/output")
                x = inputs[0]
                if x.ndim not in (3, 5) or x.shape[:-1] != output.shape[:-1]:
                    raise RuntimeError(f"Anima LoRA unexpected projection layout {tuple(x.shape)}")
                result = output
                for idx, adapter, strength in here:
                    gate_key = (idx, x.shape, x.device, x.dtype)
                    if gate_key not in cache:
                        cache[gate_key] = anima_mask_gate(active[idx].mask, latent, x, int(diffusion.patch_spatial))
                    weight_key = (id(adapter), x.device, x.dtype)
                    if weight_key not in cache:
                        cache[weight_key] = (adapter.down.to(x.device, dtype=x.dtype),
                                             adapter.up.to(x.device, dtype=x.dtype))
                    down, up = cache[weight_key]
                    delta = F.linear(F.linear(x, down), up) * (adapter.scale * strength)
                    result = result + (delta * cache[gate_key]).to(result.dtype)
                return result
            handles.append(named[target].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
        cache.clear()


class AnimaModelWrapper:
    MAX_REGIONS = 3

    def __init__(self, diffusion, regions, previous_wrapper=None):
        self.diffusion = diffusion
        self.regions = tuple(regions)
        self.previous_wrapper = previous_wrapper
        self._reported = False

    def __call__(self, apply_model, args):
        latent = args["input"]
        first = not self._reported
        try:
            with temporary_anima_hooks(self.diffusion, self.regions, latent):
                if first:
                    log.info("Sampling hooks installed (temporary; Anima experimental)")
                if self.previous_wrapper is not None:
                    result = self.previous_wrapper(apply_model, args)
                else:
                    result = apply_model(latent, args["timestep"], **args["c"])
            if first:
                log.info("Sampling hooks removed cleanly after forward (Anima)")
            return result
        except Exception:
            if first:
                log.warning("Anima forward failed; temporary hooks cleaned during exception handling")
            raise
        finally:
            self._reported = True


class AnimaBackend:
    backend_id = "anima_cosmos2"
    display_name = "Anima Cosmos-Predict2 (EXPERIMENTAL)"

    def supports(self, model):
        diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
        return bool(diffusion is not None and hasattr(diffusion, "llm_adapter")
                    and hasattr(diffusion, "blocks") and hasattr(diffusion, "patch_spatial")
                    and hasattr(diffusion, "x_embedder")
                    and int(diffusion.patch_spatial) == 2
                    and int(getattr(diffusion, "patch_temporal", -1)) == 1)

    def attach(self, model, regions):
        if not self.supports(model):
            raise TypeError("Anima backend requires native Anima Cosmos-Predict2 (not GGUF)")
        if not regions:
            return model
        out = model.clone()
        diffusion = out.model.diffusion_model
        previous = out.model_options.get("model_function_wrapper")
        earlier = ()
        if isinstance(previous, AnimaModelWrapper):
            if previous.diffusion is not diffusion:
                raise ValueError("Anima spatial regions require the same diffusion model")
            earlier = previous.regions
            previous = previous.previous_wrapper
        if len(earlier) + len(regions) > AnimaModelWrapper.MAX_REGIONS:
            raise ValueError("Anima Spatial LoRA supports up to 3 regions")
        out.set_model_unet_function_wrapper(AnimaModelWrapper(diffusion, earlier + tuple(regions), previous))
        return out
