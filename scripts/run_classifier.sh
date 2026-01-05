#!/bin/bash

# Update these variables as needed
TESTBED=4
FEATURE_GROUP="combined"
DATA_PATH="../data/mage"
CMV_PATH="../data/cmv"
OUTPUT_DIR="../results"
LOG_DIR="../logs"
RUN_CMV=false

PYTHON_SCRIPT="main_classifier.py"

echo "=== Running AI Text Detection Classifier ==="
echo "Testbed: $TESTBED"
echo "Feature group: $FEATURE_GROUP"
echo "Data path: $DATA_PATH"
echo "Run CMV: $RUN_CMV"
echo ""

PYTHON_CMD="python $PYTHON_SCRIPT --testbed $TESTBED --feature-group $FEATURE_GROUP"
PYTHON_CMD="$PYTHON_CMD --data-path \"$DATA_PATH\" --cmv-path \"$CMV_PATH\""
PYTHON_CMD="$PYTHON_CMD --output-dir \"$OUTPUT_DIR\" --log-dir \"$LOG_DIR\""

if [ "$RUN_CMV" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --run-cmv"
fi

echo "Executing: $PYTHON_CMD"
echo ""

eval $PYTHON_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "Classifier experiment completed successfully!"
else
    echo ""
    echo "Error: Classifier experiment failed."
    exit 1
fi