"""Small, local-only terminal diagnostics for Spatial LoRA.

INFO is intentionally concise. DEBUG can be enabled through the environment;
no prompt, personal image or absolute model path is logged.
"""
from __future__ import annotations

import logging
import os
import sys

LOGGER_NAME = "ComfyUI-Spatial-LoRA"
log = logging.getLogger(LOGGER_NAME)
if not log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("[Spatial LoRA] %(levelname)s | %(message)s"))
    log.addHandler(_handler)
log.propagate = False
_setting = os.getenv("SPATIAL_LORA_LOG_LEVEL", "INFO").upper()
log.setLevel(getattr(logging, _setting, logging.INFO) if _setting in {"DEBUG", "INFO", "WARNING", "ERROR"} else logging.INFO)


def report_regions(*, backend, width, height, canvas_size, configs, masks, names, strengths, active_slots):
    """Print a single per-configuration diagnostics report (no per-step spam).

    Coverage is mean mask weight over all image pixels, which includes feather.
    For nonzero feather, overlaps are weighted intersections, not merely
    rectangular intersections.
    """
    # INFO summaries can be disabled without suppressing WARNING overlap alerts.
    if not log.isEnabledFor(logging.WARNING) and not log.isEnabledFor(logging.INFO):
        return
    log.info("Backend: %s", backend)
    log.info("Resolution: %d x %d", width, height)
    if canvas_size and tuple(canvas_size) != (width, height):
        log.info("Editor canvas: %d x %d (scaled to runtime)", *canvas_size)
    log.info("Active regions: %d", len(active_slots))
    for slot in active_slots:
        i = slot - 1
        c = configs[i]
        m = masks[i]
        fraction = float(m.mean().item()) * 100.0
        log.info("Region %d | %s | LoRA: %s | strength: %.2f | feather: %d px | mask coverage: %.2f%%",
                 slot, c['name'], names[i], float(strengths[i]), c['feather'], fraction)
        log.debug("Region %d bbox: (%d, %d) %d x %d", slot, c['x'], c['y'], c['w'], c['h'])
    for idx, slot_a in enumerate(active_slots):
        for slot_b in active_slots[idx + 1:]:
            # The average weighted intersection is cheap on CPU and is measured
            # in the same output geometry as the actual sampler mask.
            weighted = float(masks[slot_a - 1].minimum(masks[slot_b - 1]).mean().item()) * 100.0
            if weighted > 0.001:
                log.warning("Regions %d/%d overlap (weighted area %.3f%% of canvas); identities may mix",
                            slot_a, slot_b, weighted)
    log.debug("Masks are diagnostics of direct LoRA gates; attention/context remains global")
