"""Model-specific backend registry. New architectures should implement this contract."""
from .zimage import ZImageBackend, SpatialModelWrapper
from .anima import AnimaBackend

BACKENDS = {"zimage_nextdit": ZImageBackend(), "anima_cosmos2": AnimaBackend()}


def resolve_backend(model, requested="auto"):
    if requested not in ("auto", *BACKENDS):
        raise ValueError(f"Unknown Spatial LoRA backend: {requested}")
    if requested == "auto":
        for backend in BACKENDS.values():
            if backend.supports(model):
                return backend
        raise TypeError(
            "Spatial LoRA: no compatible backend detected. "
            "this experimental build supports native Z-Image NextDiT and Anima Cosmos-Predict2 only "
            "(not GGUF, Nunchaku, FLUX, Qwen, or WAN)."
        )
    backend = BACKENDS[requested]
    if not backend.supports(model):
        raise TypeError(f"Spatial LoRA: selected backend {requested} does not support this MODEL.")
    return backend
