# Visual examples — Z-Image Turbo
The README image panels are based on eight illustrations supplied by me:

| Document case | Setup | Observation |
| --- | --- | --- |
| 01 | Conventional TXT2IMG, one character LoRA | Both subjects show some shared visual traits. |
| 02 | Conventional TXT2IMG, two LoRAs (0.8 and 1.0) | Both faces appear unusually similar in the supplied sample. |
| 03 | Conventional IMG2IMG, one LoRA (denoise 0.8) | Broad identity influence appears across both people. |
| 04 | Conventional IMG2IMG, two LoRAs (0.8/1.0, denoise 0.8) | The resulting faces are visually similar. |
| 05 | Spatial TXT2IMG, one selected region | The two faces look more distinct in this example. |
| 06 | Spatial TXT2IMG, two selected regions | Different character features are better differentiated. |
| 07 | Spatial IMG2IMG, one region | Character changes are directed toward the selected person. |
| 08 | Spatial IMG2IMG, two regions | Each LoRA is associated with a different subject. |

The images show examples, not guarantees or a controlled scientific study. Results depend on training captions/dataset composition, model, seeds, prompt, input image and strength/denoise settings.

## Practical takeaway
A well-trained LoRA with distinct trigger words and more varied single-person and group training photos can improve identity separation. Spatial LoRA helps limit *direct* changes outside a target region but cannot guarantee that global attention will never spread identity features between subjects.

## Images and reuse
Only rendered comparison panels are included in the public documentation package. The original source images remain in my supplied material; character LoRA weight files are not distributed. Check the permission to redistribute all imagery before publishing any derivative.