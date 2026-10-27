# Example workflows

Two **editable examples** provided by the author for native Z-Image Turbo:

- `Z Image Turbo Spatial TXT2IMG.json`
- `Z Image Turbo Spatial IMG2IMG.json`

Drop a JSON file onto the ComfyUI canvas or load it from ComfyUI's workflow menu.

## Before running

1. Replace the diffusion model, CLIP and VAE selections with files installed on **your** machine. The author's workflow references Z-Image Turbo INT8 Convrot (native safetensors), a Qwen text encoder and the Z-Image VAE.
2. Open **Spatial LoRA | Multi LoRA**, select your own LoRA(s), and adjust their regions. LoRA selectors intentionally start at `None`.
3. Check the additional custom nodes referenced by these examples: `ZSamplerTurbo2`, `StylePromptEncoder2`, `ImpactWildcardProcessor`, `SDXLEmptyLatentSizePicker+`; the IMG2IMG example also uses `Fast Groups Bypasser (rgthree)`, `ImageResizeKJv2` and `ColorMatchV2`. These are separate projects and are **not included** in Spatial LoRA.
4. For IMG2IMG, copy `assets/example.png` into `ComfyUI/input/`, or choose another image in the workflow's `Load Image` node. The template expects `example.png` by default.
5. Verify that the spatial node receives the same width and height as the actual sampling resolution.

The original JSON templates are preserved as supplied. They are examples of a complete setup, not minimal examples of plugin dependencies.
