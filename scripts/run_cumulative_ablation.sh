#!/bin/bash

# Update these variables as needed
TESTBED=7
ABLATION_ORDER="morphological,psycholinguistic,information,dependency,surface,pos,emotion,entities,semantic,readability,lexical_richness"
FEATURE_JSON="../data/feature_consistency_report.json"
DATA_PATH="../data/mage"
CMV_PATH="../data/cmv"
OUTPUT_DIR="../results/cumulative_ablation"
LOG_DIR="../logs/cumulative_ablation"
RUN_CMV=false

PYTHON_SCRIPT="cumulative_ablation.py"

echo "=== Running Cumulative Ablation Study ==="
echo "Testbed: $TESTBED"
echo "Ablation order: $ABLATION_ORDER"
echo "Run CMV: $RUN_CMV"
echo ""

PYTHON_CMD="python $PYTHON_SCRIPT --testbed $TESTBED --ablation-order \"$ABLATION_ORDER\""
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
    echo "Cumulative ablation study completed successfully!"
else
    echo ""
    echo "Error: Cumulative ablation study failed."
    exit 1
fi