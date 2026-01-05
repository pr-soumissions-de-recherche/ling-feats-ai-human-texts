#!/bin/bash

# Update these variables as needed
INPUT_FILE=""
OUTPUT_DIR="../data/features"
CREATE_COMBINED=false
NORMALIZE_METHOD="token"

PYTHON_SCRIPT="elfen_extractor.py"

echo "=== ELFEN Feature Extraction ==="
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Create combined file: $CREATE_COMBINED"
echo "Normalization method: $NORMALIZE_METHOD"
echo ""

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' does not exist."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

PYTHON_CMD="python $PYTHON_SCRIPT --input \"$INPUT_FILE\" --output \"$OUTPUT_DIR\" --normalize $NORMALIZE_METHOD"

if [ "$CREATE_COMBINED" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --combined"
fi

echo "Executing: $PYTHON_CMD"
echo ""

eval $PYTHON_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "Feature extraction completed successfully!"
else
    echo ""
    echo "Error: Feature extraction failed."
    exit 1
fi