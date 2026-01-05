#!/bin/bash

# Update these variables as needed
TESTBED=4 
FEATURE_GROUP="combined"
FEATURE_JSON="../data/feature_consistency_report.json" # file containing complete set of features after preprocessing
DATA_PATH="../data/mage"
CMV_PATH="../data/cmv"
OUTPUT_DIR="../results/ablation"
LOG_DIR="../logs/ablation"
RUN_CMV=false

PYTHON_SCRIPT="ablation_study.py"

echo "=== Running Ablation Study ==="
echo "Testbed: $TESTBED"
echo "Feature group: $FEATURE_GROUP"
echo "Run CMV: $RUN_CMV"
echo ""

PYTHON_CMD="python $PYTHON_SCRIPT --testbed $TESTBED --feature-group $FEATURE_GROUP"
PYTHON_CMD="$PYTHON_CMD --feature-json \"$FEATURE_JSON\""
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
    echo "Ablation study completed successfully!"
else
    echo ""
    echo "Error: Ablation study failed."
    exit 1
fi