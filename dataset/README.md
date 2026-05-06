Example dataset layout (template only). Replace placeholders with real images.

Expected files per scene:
- images/scene_xxx/defocus_1.tif ... defocus_6.tif
- focusmap/scene_xxx/focusmap_1.png ... focusmap_6.png
- labels/scene_xxx/all_in_focus_1.tif

Optional flow supervision (Alignment-Net):
- flow/scene_xxx/flow_1.flo ... flow_5.flo

This directory only contains folders. Training or inference will fail until
valid image/flow files are placed here.
