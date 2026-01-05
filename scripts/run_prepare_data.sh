#!/bin/bash

# Update these variables as needed
MODE="all"                                    # Options: check, prepare, verify, all
DATA_DIR="../data/features"                   # For check/verify modes
MAGE_DIR="../data/mage_features_raw"          # Raw MAGE features
CMV_DIR="../data/cmv_features_raw"            # Raw CMV features
OUTPUT_DIR="../data/features"                 # Output for prepared features
SAVE_REPORT=false                             # Save error reports to JSON?

PYTHON_SCRIPT="prepare_data.py"

echo "=== Data Preparation Tool ==="
echo "Mode: $MODE"
echo ""

case $MODE in
    check)
        echo "Checking for NaN and Inf values in: $DATA_DIR"
        PYTHON_CMD="python $PYTHON_SCRIPT --mode check --data-dir \"$DATA_DIR\""
        ;;
    prepare)
        echo "Preparing features:"
        echo "  MAGE: $MAGE_DIR"
        echo "  CMV: $CMV_DIR"
        echo "  Output: $OUTPUT_DIR"
        PYTHON_CMD="python $PYTHON_SCRIPT --mode prepare --mage-dir \"$MAGE_DIR\" --cmv-dir \"$CMV_DIR\" --output-dir \"$OUTPUT_DIR\""
        ;;
    verify)
        echo "Verifying feature consistency in: $DATA_DIR"
        PYTHON_CMD="python $PYTHON_SCRIPT --mode verify --data-dir \"$DATA_DIR\""
        ;;
    all)
        echo "Running all preparation steps"
        PYTHON_CMD="python $PYTHON_SCRIPT --mode all --mage-dir \"$MAGE_DIR\" --cmv-dir \"$CMV_DIR\" --output-dir \"$OUTPUT_DIR\" --data-dir \"$DATA_DIR\""
        ;;
    *)
        echo "Error: Invalid mode. Choose: check, prepare, verify, or all"
        exit 1
        ;;
esac

if [ "$SAVE_REPORT" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --save-report"
fi

echo ""
echo "Executing: $PYTHON_CMD"
echo ""

eval $PYTHON_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "Data preparation completed successfully!"
else
    echo ""
    echo "Error: Data preparation failed."
    exit 1
fi