hazard_model_path="checkpoint/hazard/deberta-512"
product_model_path="checkpoint/product/deberta-512"
input_json_path="data/public_test_512.json"
output_csv_path="submission.csv"
output_zip_path="submission.zip"
output_hazard_json_path="new_result/public/hazard/hazard_predictions_roberta-large-512.json"
output_product_json_path="new_resul/public/product/product_predictions_roberta-large-512.json"

cd ..

python3 predict_independent.py \
    --hazard_model "$hazard_model_path" \
    --product_model "$product_model_path" \
    --input_json "$input_json_path" \
    --output_csv "$output_csv_path" \
    --output_zip "$output_zip_path" \
    --output_hazard_json "$output_hazard_json_path" \
    --output_product_json "$output_product_json_path"
