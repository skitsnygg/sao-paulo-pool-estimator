Jardins pool segmentation workspace

This folder is for manual mask annotation.

Expected mask format:
- One PNG mask per image in `masks/`
- Same base filename as the image (e.g., image `r0000_c0000.png` -> mask `r0000_c0000.png`)
- Single channel, 8-bit PNG
- Pixel values: 0 = background, 255 = pool
