"""Z-Image native NextDiT backend: proven v0.3 multi-hook implementation.

The backend owns all architecture-specific operations. Other backends must
provide model detection and attach(model, regions); do NOT reuse Z-Image token
mapping for different transformers without validating their token layouts.
"""
from ..spatial import temporary_multi_lora_hooks
from ..diagnostics import log


class SpatialModelWrapper:
    """Accumulate all regional adapters in one temporary hook context."""

    MAX_REGIONS = 3

    def __init__(self, diffusion_model, regions, previous_wrapper=None):
        self.diffusion_model = diffusion_model
        self.regions = tuple(regions)
        self.previous_wrapper = previous_wrapper
        self._warned_geometry = False
        self._reported_hooks = False

    def __call__(self, apply_model, args):
        latent = args["input"]
        if latent.ndim == 4 and not self._warned_geometry:
            latent_ratio = latent.shape[-1] / latent.shape[-2]
            for idx, region in enumerate(self.regions, start=1):
                mask_ratio = region.mask.shape[-1] / region.mask.shape[-2]
                if abs(mask_ratio / latent_ratio - 1) > 0.04:
                    log.warning("BBox %d (%dx%d) aspect ratio differs from latent (%dx%d); "
                                "connect width/height to the same latent size source",
                                idx, region.mask.shape[-1], region.mask.shape[-2],
                                latent.shape[-1], latent.shape[-2])
            self._warned_geometry = True
        # The temporary hook context is entered once per forward/denoising
        # step. Log INFO only once per wrapper; DEBUG records every step.
        first = not self._reported_hooks
        entered = False
        try:
            with temporary_multi_lora_hooks(self.diffusion_model, self.regions, latent):
                entered = True  # all hooks have been installed
                if first:
                    log.info("Sampling hooks installed (temporary; Z-Image native)")
                log.debug("Sampling forward: latent=%s, regions=%d", tuple(latent.shape), len(self.regions))
                if self.previous_wrapper is not None:
                    output = self.previous_wrapper(apply_model, args)
                else:
                    output = apply_model(latent, args["timestep"], **args["c"])
            if first:
                log.info("Sampling hooks removed cleanly after forward")
            else:
                log.debug("Sampling hooks removed after forward")
            return output
        except Exception:
            if entered and first:
                log.warning("Sampling forward failed; temporary hooks cleaned during exception handling")
            raise
        finally:
            if entered:
                self._reported_hooks = True


class ZImageBackend:
    backend_id = "zimage_nextdit"
    display_name = "Z-Image Native NextDiT"

    def supports(self, model):
        diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
        return bool(diffusion is not None
                    and hasattr(diffusion, "layers")
                    and hasattr(diffusion, "patch_size")
                    and hasattr(diffusion, "x_embedder")
                    and int(diffusion.patch_size) == 2)

    def attach(self, model, regions):
        if not self.supports(model):
            raise TypeError("Spatial LoRA Z-Image backend requires native NextDiT (patch_size=2).")
        regions = tuple(regions)
        if not regions:
            return model
        out = model.clone()
        diffusion = out.model.diffusion_model
        previous = out.model_options.get("model_function_wrapper")
        earlier = ()
        if isinstance(previous, SpatialModelWrapper):
            if previous.diffusion_model is not diffusion:
                raise RuntimeError("Spatial LoRA: regions must share the same diffusion model")
            earlier = previous.regions
            previous = previous.previous_wrapper
        if len(earlier) + len(regions) > SpatialModelWrapper.MAX_REGIONS:
            raise ValueError("Limite experimental de 3 LoRAs regionais. Remova um node Masked LoRA.")
        out.set_model_unet_function_wrapper(SpatialModelWrapper(diffusion, earlier + regions, previous))
        return out
