INPUT_JSON="data/test_data/public_test_512.json"    
OUTPUT_DIR="results"                
LABEL_MAPPING="data/label_mappings.json"  

MODEL_NAME="Quintu/deberta-multitask-v0"
BATCH_SIZE=2  

cd ..

python3 predict_multitask.py \
    --model_name "$MODEL_NAME" \
    --input_json "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --label_mapping "$LABEL_MAPPING" 
