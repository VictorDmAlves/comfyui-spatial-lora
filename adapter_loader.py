"""LoRA I/O shared by the legacy and the unified node; no model-specific hooks."""
import os

from .spatial import extract_adapters
from .diagnostics import log


class AdapterCache:
    def __init__(self):
        self._entries = {}

    def get(self, lora_name):
        import folder_paths
        import comfy.utils
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        mtime = os.stat(path).st_mtime_ns
        cached = self._entries.get(path)
        if cached is not None and cached[0] == mtime:
            log.debug("Adapter cache hit: %s", lora_name)
            return cached[1]
        data = comfy.utils.load_torch_file(path, safe_load=True)
        adapters, ignored = extract_adapters(data)
        if ignored:
            log.warning("Ignored %d unsupported/non-spatial LoRA keys in %s (examples: %s)",
                        len(ignored), lora_name, ignored[:3])
        value = tuple(adapters)
        self._entries[path] = (mtime, value)
        log.info("Loaded LoRA: %s (%d supported projections)", lora_name, len(value))
        return value
