# predict_tiles.py Script Documentation

## Overview
The `predict_tiles.py` script runs segmentation inference on orthophoto tiles and saves the results in multiple formats for analysis and further processing.

## Expected Input Format
- **Directory of orthophoto tiles**: Any directory containing image files
- **Supported formats**: .jpg, .jpeg, .png, .tif, .tiff
- **Example**: `data/raw/tiles/19/` (contains tile images like `193948_297189.jpg`)

## Output Directory Structure
The script creates the following output structure:
```
output_directory/
├── images/              # Annotated images with bounding boxes
├── labels/              # YOLO format label files (.txt)
└── detections.csv       # CSV with pool detections (image_name, confidence, bbox_area)
```

## Example Command
```bash
python scripts/predict_tiles.py \
    --weights runs/segment/sao-paulo-pools-seg-v3/weights/best.pt \
    --source data/raw/tiles/19/ \
    --output data/processed/predictions/ \
    --batch-size 8 \
    --save-images \
    --save-labels \
    --save-csv
```

## Key Features
1. **Model Loading**: Loads trained YOLO segmentation model from specified path
2. **Batch Processing**: Processes images in configurable batches for efficiency
3. **Multi-format Support**: Handles various image formats
4. **Multiple Output Formats**: Generates annotated images, YOLO labels, and CSV detections
5. **Error Handling**: Graceful handling of missing files and processing errors

## Validation Results
The script has been validated to:
- Import correctly with all dependencies
- Have proper function signatures
- Support all required modules
- Follow project conventions

## Next Steps
To run the actual inference pipeline:
1. Ensure all dependencies are installed
2. Run the command from the project root directory
3. Monitor output for processing results
4. Inspect generated files for validation