#!/bin/bash
# =============================================================================
# RoboTech ML Training Pipeline
# =============================================================================
# Automated flow: Match Logs → Training Data → Neural Network → Deploy
#
# Usage:
#   ./train_from_matches.sh <logs_dir>
#
# Example:
#   ./train_from_matches.sh /home/robotechtecnl/game_logs/
#
# Requirements:
#   - Python 3 with tensorflow/keras installed
#   - RCG log files from played matches
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEAM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CPPDNN_DIR="$TEAM_DIR/../../tools/CppDNN"
OUTPUT_DIR="$SCRIPT_DIR/output"
DATA_DIR="$SCRIPT_DIR/data"

if [ -z "$1" ]; then
    echo "Usage: $0 <logs_directory>"
    echo ""
    echo "The logs directory should contain .rcg or .rcg.gz files from matches."
    echo ""
    echo "Example:"
    echo "  $0 /home/robotechtecnl/game_logs/"
    echo ""
    echo "Full pipeline:"
    echo "  1. Extracts features from match logs → CSV"
    echo "  2. Trains neural network (Keras) → .h5"
    echo "  3. Converts to C++ format → .txt"
    echo "  4. Deploys to build directory"
    exit 1
fi

LOGS_DIR="$1"

echo "============================================="
echo " RoboTech ML Training Pipeline"
echo "============================================="
echo " Team dir:  $TEAM_DIR"
echo " Logs dir:  $LOGS_DIR"
echo " Output:    $OUTPUT_DIR"
echo "============================================="

# Step 1: Extract data from logs
echo ""
echo "[Step 1/4] Extracting training data from match logs..."
mkdir -p "$DATA_DIR"
python3 "$SCRIPT_DIR/extract_from_logs.py" "$LOGS_DIR" "$DATA_DIR"

# Check if data was generated
if [ ! -f "$DATA_DIR/field_eval_data.csv" ]; then
    echo "ERROR: No training data generated. Check log files."
    exit 1
fi

SAMPLE_COUNT=$(wc -l < "$DATA_DIR/field_eval_data.csv")
echo "  → $((SAMPLE_COUNT - 1)) training samples extracted"

if [ "$SAMPLE_COUNT" -lt 100 ]; then
    echo "WARNING: Very few samples. Need more match logs for good training."
    echo "  Recommendation: at least 10 full matches (~5000+ samples)"
fi

# Step 2: Train the network
echo ""
echo "[Step 2/4] Training neural network..."
mkdir -p "$OUTPUT_DIR"
python3 "$SCRIPT_DIR/trainer.py" "$DATA_DIR" "$OUTPUT_DIR"

# Step 3: Convert to C++ format
echo ""
echo "[Step 3/4] Converting model to C++ format..."
if [ -f "$OUTPUT_DIR/best_field_eval.h5" ]; then
    MODEL_FILE="$OUTPUT_DIR/best_field_eval.h5"
else
    MODEL_FILE="$OUTPUT_DIR/field_eval_model.h5"
fi

cd "$OUTPUT_DIR"
python3 "$CPPDNN_DIR/script/DecodeKerasModel.py" "$MODEL_FILE"

# The decoder outputs a .txt file with same name
MODEL_TXT="${MODEL_FILE%.h5}.txt"
if [ ! -f "$MODEL_TXT" ]; then
    # Try common output patterns
    MODEL_TXT=$(ls -t "$OUTPUT_DIR"/*.txt 2>/dev/null | head -1)
fi

if [ -z "$MODEL_TXT" ] || [ ! -f "$MODEL_TXT" ]; then
    echo "ERROR: Model conversion failed. Check DecodeKerasModel.py output."
    exit 1
fi

# Step 4: Deploy to build
echo ""
echo "[Step 4/4] Deploying to team build directory..."
cp "$MODEL_TXT" "$TEAM_DIR/src/field_eval_weights.txt"
cp "$MODEL_TXT" "$TEAM_DIR/build/bin/field_eval_weights.txt" 2>/dev/null || true

echo ""
echo "============================================="
echo " Training Complete!"
echo "============================================="
echo " Model:   $MODEL_TXT"
echo " Weights: $TEAM_DIR/src/field_eval_weights.txt"
echo ""
echo " To use in the team, integrate FieldEvalDNN"
echo " into sample_field_evaluator.cpp (see code comments)"
echo "============================================="
