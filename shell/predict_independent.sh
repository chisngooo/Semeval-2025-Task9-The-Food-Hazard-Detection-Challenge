#!/bin/bash

# Cấu hình các đường dẫn đầu vào và đầu ra
hazard_model_path="Quintu/roberta-large-768-hazard"
product_model_path="Quintu/roberta-large-768-product"
input_json_path="data/private_test_512.json"
output_csv_path="submission.csv"
output_zip_path="submission.zip"
output_hazard_json_path="results/private/hazard/hazard_predictions_roberta-large-768.json"
output_product_json_path="results/private/product/product_predictions_roberta-large-768.json"
cd ..
# Chạy script Python với các biến đã cấu hình
python3 predict_independent.py \
    --hazard_model "$hazard_model_path" \
    --product_model "$product_model_path" \
    --input_json "$input_json_path" \
    --output_csv "$output_csv_path" \
    --output_zip "$output_zip_path" \
    --output_hazard_json "$output_hazard_json_path" \
    --output_product_json "$output_product_json_path"
