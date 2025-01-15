#!/bin/bash

# Fixed paths for input and output files
INPUT_JSON="data/public_test.csv"    # Đường dẫn đến file JSON đầu vào
OUTPUT_DIR="/results"                # Thư mục đầu ra
LABEL_MAPPING="data/label_mappings.json"  # Đường dẫn đến file mapping

# Model name and other arguments
MODEL_NAME="microsoft/deberta-v3-large"
BATCH_SIZE=2  # Batch size nếu cần
WEIGHTS=(1.0) # Cân nặng nếu dùng ensemble (chỉnh sửa nếu cần)
cd ..
# Run the Python script
python3 predict_multitask.py \
    --model_name "$MODEL_NAME" \
    --input_json "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --label_mapping "$LABEL_MAPPING" \
    --weights "${WEIGHTS[@]}"

# Change directory to the folder outside

