"""Per-image-token LoRA injection, for ComfyUI's native Lumina2/NextDiT model.

The hooks are temporary: no LoRA tensor is merged into a base weight.
Only the main joint transformer layers are supported (``layers.N``).
Other target locations are rejected rather than silently blended.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Adapter:
    target: str
    down: torch.Tensor
    up: torch.Tensor
    scale: float
    slice_index: int | None = None


# Model's native joint layers use attention.qkv, attention.out and
# feed_forward.w1/w2/w3. LoRA files can instead call Q/K/V projections by name.
LORA_KEY_PATTERN = re.compile(
    r"^(?P<name>.+?)\.(?P<part>lora_down|lora_up|lora_A|lora_B)(?:\.default)?\.weight$"
)


def canonical_target(raw: str) -> tuple[str, int | None]:
    name = raw
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = name.replace(".attention.to_qkv", ".attention.qkv")
    name = name.replace(".attention.to_out.0", ".attention.out")
    name = name.replace(".attention.to_out", ".attention.out")
    # OneTrainer and diffusers commonly save split Q/K/V projections.
    for suffix, idx in ((".attention.to_q", 0), (".attention.to_k", 1), (".attention.to_v", 2)):
        if name.endswith(suffix):
            return name[:-len(suffix)] + ".attention.qkv", idx
    return name, None


def extract_adapters(state: dict) -> tuple[list[Adapter], list[str]]:
    groups: Dict[str, dict] = {}
    unsupported: list[str] = []
    for key, tensor in state.items():
        match = LORA_KEY_PATTERN.match(key)
        if match:
            base, kind = match.group("name"), match.group("part")
            groups.setdefault(base, {})["down" if kind in ("lora_down", "lora_A") else "up"] = tensor
        elif key.endswith((".alpha", ".lora_alpha")):
            base = key[:-len(".lora_alpha")] if key.endswith(".lora_alpha") else key[:-len(".alpha")]
            groups.setdefault(base, {})["alpha"] = tensor
        elif any(t in key for t in ("lora_mid", "dora_scale", "lokr_", "loha_", "lora_magnitude")):
            unsupported.append(key)
        elif re.search(r"\.(?:lora_down|lora_up|lora_A|lora_B)(?:\.|$)", key):
            # Catch export variants that would otherwise be silently skipped.
            raise ValueError(f"Formato de chave LoRA nao reconhecido: {key}")
    if unsupported:
        raise ValueError("Formato LoRA com componentes nao suportados (DoRA/LyCORIS): " + ", ".join(unsupported[:3]))

    adapters = []
    for raw, parts in groups.items():
        if "down" not in parts or "up" not in parts:
            unsupported.append(raw + " (par LoRA incompleto)")
            continue
        target, segment = canonical_target(raw)
        # Context, timestep and norm layers have no image-token correspondence.
        if not re.fullmatch(r"layers\.\d+\.(?:attention\.(?:qkv|out)|feed_forward\.w[123])", target):
            unsupported.append(raw + " (camada nao espacial)")
            continue
        a, b = parts["down"], parts["up"]
        if not (torch.is_tensor(a) and torch.is_tensor(b) and a.ndim == 2 and b.ndim == 2 and a.shape[0] == b.shape[1]):
            unsupported.append(raw + " (dimensoes incompatíveis)")
            continue
        alpha = parts.get("alpha")
        if alpha is None:
            scale = 1.0  # Comfy convention when no per-target alpha is saved
        else:
            scale = float(alpha.detach().cpu().item()) / int(a.shape[0])
        adapters.append(Adapter(target, a.detach().cpu().contiguous(), b.detach().cpu().contiguous(), scale, segment))

    if not adapters:
        raise ValueError("Nenhuma camada LoRA conjunta compatível encontrada. Verifique o formato OneTrainer/Z-Image.")
    # Prevent a misleading partial-strength result with unknown formats.
    n_total = len(groups)
    if len(adapters) < max(1, n_total * 0.75):
        raise ValueError(
            f"Cobertura LoRA insuficiente: {len(adapters)}/{n_total} camadas. "
            f"Exemplos ignorados: {unsupported[:4]}. "
            "Nao aplicamos uma LoRA parcial silenciosamente."
        )
    return adapters, unsupported


def validate_targets(diffusion_model, adapters: list[Adapter]) -> dict[str, list[Adapter]]:
    if not (hasattr(diffusion_model, "layers") and hasattr(diffusion_model, "patch_size") and hasattr(diffusion_model, "x_embedder")):
        raise TypeError("Este node exige o Z-Image nativo (NextDiT / Lumina2), nao GGUF ou Nunchaku.")
    names = dict(diffusion_model.named_modules())
    grouped: dict[str, list[Adapter]] = {}
    for a in adapters:
        if a.slice_index is not None and (a.slice_index not in (0, 1, 2) or not a.target.endswith(".attention.qkv")):
            raise ValueError(f"Segmento QKV invalido para {a.target}: {a.slice_index}")
        module = names.get(a.target)
        if module is None or not hasattr(module, "weight"):
            raise ValueError(f"Camada {a.target} nao existe no diffusion model. Confira a conversao da LoRA.")
        weight_shape = tuple(module.weight.shape)
        if len(weight_shape) != 2:
            raise ValueError(f"Camada {a.target} nao e uma projecao linear")
        out_dim, in_dim = weight_shape
        if a.down.shape[1] != in_dim:
            raise ValueError(f"LoRA {a.target}: entrada {a.down.shape[1]} != {in_dim}")
        if a.slice_index is None:
            if a.up.shape[0] != out_dim:
                raise ValueError(f"LoRA {a.target}: saida {a.up.shape[0]} != {out_dim}")
        else:
            if out_dim % 3 or a.up.shape[0] != out_dim // 3:
                raise ValueError(f"LoRA {a.target}: QKV fuso de dimensoes nao suportadas")
        grouped.setdefault(a.target, []).append(a)
    return grouped


def build_gate_from_latent(mask, b: int, seq_len: int, latent_height: int, latent_width: int,
                           patch_size: int, device, dtype, pad_tokens_multiple=None):
    """Map spatial mask to [text prefix | image tokens | optional image padding].

    ComfyUI's Z-Image native NextDiT pads image tokens to a multiple of 32
    when x_pad_token is enabled. The padding goes AFTER the image tokens.
    """
    if mask.ndim != 3 or mask.shape[0] < 1 or mask.shape[1] < 1 or mask.shape[2] < 1:
        raise ValueError("Mascara deve ter formato [B,H,W], com dimensoes positivas")
    if b < 1 or patch_size < 1 or latent_height < 1 or latent_width < 1:
        raise ValueError("Batch, patch_size e dimensoes latentes devem ser positivos")
    grid_h = (latent_height + patch_size - 1) // patch_size
    grid_w = (latent_width + patch_size - 1) // patch_size
    n_image = grid_h * grid_w
    multiple = int(pad_tokens_multiple) if pad_tokens_multiple else 0
    pad_after = (-n_image) % multiple if multiple else 0
    if seq_len < n_image + pad_after:
        raise ValueError(f"Sequencia {seq_len} menor que os {n_image}+{pad_after} tokens da imagem")
    flat = F.interpolate(mask.float().unsqueeze(1), size=(grid_h, grid_w), mode="bilinear", align_corners=False).flatten(1)
    if flat.shape[0] != b:
        if flat.shape[0] == 1:
            flat = flat.expand(b, -1)
        elif b % flat.shape[0] == 0:
            flat = flat.repeat(b // flat.shape[0], 1)
        else:
            raise ValueError(f"Batch da mascara {flat.shape[0]} incompativel com batch {b}")
    text_prefix = seq_len - (n_image + pad_after)
    # True image tokens start after text. Never inject into padded image tokens.
    flat = F.pad(flat, (text_prefix, pad_after), value=0.0)
    return flat.to(device=device, dtype=dtype).unsqueeze(-1)


@dataclass(frozen=True)
class SpatialRegion:
    """An independent adapter, weight and MASK, using image-pixel coordinates."""
    adapters: tuple[Adapter, ...]
    mask: torch.Tensor
    strength: float
    label: str = ""


@contextmanager
def temporary_multi_lora_hooks(diffusion_model, regions, latent):
    """Apply all regional contributions in *one hook per target module*.

    Calling this on an existing MODEL with more regions preserves previous
    regions without nesting PyTorch hooks. The base module weights are not
    modified. The model is shared across ModelPatcher clones; this context
    therefore must not be used concurrently on the same module object.
    """
    if latent.ndim != 4:
        raise ValueError("Versao experimental aceita apenas latents de imagem [B,C,H,W], sem video/omni")
    grouped = {}  # target -> [(region_idx, adapter, strength)]
    active = []
    for region in regions:
        if not torch.is_tensor(region.mask) or region.mask.ndim != 3:
            raise ValueError("Mascara deve ter formato [B,H,W]")
        if not bool(torch.any(region.mask > 0)) or region.strength == 0:
            continue
        region_idx = len(active)
        active.append(region)
        for target, adapters in validate_targets(diffusion_model, list(region.adapters)).items():
            for adapter in adapters:
                grouped.setdefault(target, []).append((region_idx, adapter, region.strength))

    if not grouped:
        yield
        return
    named = dict(diffusion_model.named_modules())
    handles = []
    cache = {}
    patch_size = int(diffusion_model.patch_size)
    image_pad_multiple = getattr(diffusion_model, "pad_tokens_multiple", None)
    latent_h, latent_w = latent.shape[-2:]
    try:
        for target, target_adapters in grouped.items():
            def output_hook(module, inputs, output, *, adapters_here=target_adapters):
                if not isinstance(output, torch.Tensor) or not inputs or not isinstance(inputs[0], torch.Tensor):
                    raise RuntimeError("Formato inesperado no forward da camada Z-Image")
                x = inputs[0]
                if x.ndim != 3 or output.ndim != 3:
                    raise RuntimeError("LoRA regional exige tensor [batch,tokens,features]")
                result = output
                for region_idx, adapter, strength in adapters_here:
                    key_mask = (region_idx, x.shape[0], x.shape[1], x.device, x.dtype)
                    gate = cache.get(("gate", key_mask))
                    if gate is None:
                        gate = build_gate_from_latent(
                            active[region_idx].mask, x.shape[0], x.shape[1], latent_h, latent_w,
                            patch_size, x.device, x.dtype, image_pad_multiple
                        )
                        cache[("gate", key_mask)] = gate
                    device_key = (id(adapter), x.device, x.dtype)
                    weights = cache.get(("weights", device_key))
                    if weights is None:
                        weights = (adapter.down.to(x.device, dtype=x.dtype), adapter.up.to(x.device, dtype=x.dtype))
                        cache[("weights", device_key)] = weights
                    down, up = weights
                    delta = F.linear(F.linear(x, down), up) * (adapter.scale * strength)
                    delta = (delta * gate).to(result.dtype)
                    if adapter.slice_index is None:
                        result = result + delta
                    else:
                        # Q/K/V are equal contiguous thirds only for supported native QKV layouts.
                        span = result.shape[-1] // 3
                        start = span * adapter.slice_index
                        part = result[..., start:start + span] + delta
                        result = torch.cat((result[..., :start], part, result[..., start + span:]), dim=-1)
                return result
            handles.append(named[target].register_forward_hook(output_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
        cache.clear()


@contextmanager
def temporary_lora_hooks(diffusion_model, adapters, mask, latent, strength):
    """Backwards-compatible single-region entry point used in v0.2 tests."""
    region = SpatialRegion(tuple(adapters), mask, strength)
    with temporary_multi_lora_hooks(diffusion_model, (region,), latent):
        yield
