"""Model-specific backend registry. New architectures should implement this contract."""
from .zimage import ZImageBackend, SpatialModelWrapper

BACKENDS = {"zimage_nextdit": ZImageBackend()}


def resolve_backend(model, requested="auto"):
    if requested not in ("auto", *BACKENDS):
        raise ValueError(f"Unknown Spatial LoRA backend: {requested}")
    if requested == "auto":
        for backend in BACKENDS.values():
            if backend.supports(model):
                return backend
        raise TypeError(
            "Spatial LoRA: no compatible backend detected. "
            "v0.6 currently supports native Z-Image NextDiT models only "
            "(not GGUF, Nunchaku, FLUX, Qwen, or WAN)."
        )
    backend = BACKENDS[requested]
    if not backend.supports(model):
        raise TypeError(f"Spatial LoRA: selected backend {requested} does not support this MODEL.")
    return backend
