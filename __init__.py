"""ComfyUI-Spatial-LoRA v0.7: unified Z-Image spatial LoRA node.

The mask math and chained-node regressions remain in source for testing, but
only the user-facing unified node is registered in this release.
"""
from .multi_lora import SpatialMultiLoRA

NODE_CLASS_MAPPINGS = {"SpatialMultiLoRA": SpatialMultiLoRA}
NODE_DISPLAY_NAME_MAPPINGS = {"SpatialMultiLoRA": "Spatial LoRA | Multi LoRA"}
WEB_DIRECTORY = "./web"

# Optional in-memory thumbnail API for connected IMAGE previews (post-execution).
from .reference_cache import register_reference_route
register_reference_route()
