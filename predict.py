import argparse
import json
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import csv
import zipfile
from tqdm import tqdm

# Define function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Hazard and product prediction script")
    parser.add_argument("--hazard_model", type=str, required=True, help="Path to the hazard model checkpoint on Hugging Face")
    parser.add_argument("--product_model", type=str, required=True, help="Path to the product model checkpoint on Hugging Face")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file")
    parser.add_argument("--output_zip", type=str, required=True, help="Path to the output ZIP file")
    parser.add_argument("--output_hazard_json", type=str, required=True, help="Path to the output hazard predictions JSON")
    parser.add_argument("--output_product_json", type=str, required=True, help="Path to the output product predictions JSON")
    return parser.parse_args()

# Main function for prediction
def main():
    # Parse command-line arguments
    args = parse_args()

    MAX_LENGTH = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models and tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(args.hazard_model)
    hazard_model = AutoModelForSequenceClassification.from_pretrained(args.hazard_model)
    product_model = AutoModelForSequenceClassification.from_pretrained(args.product_model)

    hazard_model.to(device)
    product_model.to(device)

    hazard_model.eval()
    product_model.eval()

    hazard_label_mapping = hazard_model.config.id2label
    product_label_mapping = product_model.config.id2label
    product_label_mapping = {str(k): v for k, v in product_label_mapping.items()}
    hazard_label_mapping = {str(k): v for k, v in hazard_label_mapping.items()}

    def predict(model, tokenizer, text):
        inputs = tokenizer(text, truncation=True, return_tensors="pt", padding=True, max_length=MAX_LENGTH)
        inputs = {key: value.to(device) for key, value in inputs.items()}  

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()  
        return probs

    def aggregate_by_stt(data):
        grouped_results = defaultdict(lambda: {"product_probs": defaultdict(float), "hazard_probs": defaultdict(float)})
        for row in data:
            stt = row["stt"]
            for label, prob in row["product_prediction"].items():
                grouped_results[stt]["product_probs"][label] += prob
            for label, prob in row["hazard_prediction"].items():
                grouped_results[stt]["hazard_probs"][label] += prob
        return grouped_results

    with open(args.input_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    detailed_predictions = []

    for item in tqdm(raw_data, desc="Predicting", unit="sample"):
        text = item["text"]
        stt = item["stt"]
        product_probs = predict(product_model, tokenizer, text)
        hazard_probs = predict(hazard_model, tokenizer, text)

        product_prediction = {product_label_mapping[str(idx)]: float(prob) for idx, prob in enumerate(product_probs)}
        hazard_prediction = {hazard_label_mapping[str(idx)]: float(prob) for idx, prob in enumerate(hazard_probs)}

        detailed_predictions.append({
            "stt": stt,
            "product_prediction": product_prediction,
            "hazard_prediction": hazard_prediction
        })

    aggregated_results = aggregate_by_stt(detailed_predictions)

    final_predictions = []
    product_output = []
    hazard_output = []

    for stt, results in aggregated_results.items():
        product_label = max(results["product_probs"], key=results["product_probs"].get)
        hazard_label = max(results["hazard_probs"], key=results["hazard_probs"].get)

        product_probs = {label: results["product_probs"][label] for label in results["product_probs"]}
        hazard_probs = {label: results["hazard_probs"][label] for label in results["hazard_probs"]}

        final_predictions.append({
            "stt": stt,
            "product_category": product_label,
            "hazard_category": hazard_label
        })

        product_output.append({
            "stt": stt,
            "product_probabilities": product_probs
        })

        hazard_output.append({
            "stt": stt,
            "hazard_probabilities": hazard_probs
        })

    with open(args.output_product_json, "w", encoding="utf-8") as jsonf:
        json.dump(product_output, jsonf, indent=4, ensure_ascii=False)

    with open(args.output_hazard_json, "w", encoding="utf-8") as jsonf:
        json.dump(hazard_output, jsonf, indent=4, ensure_ascii=False)

    with open(args.output_csv, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["stt", "product-category", "hazard-category"])
        for pred in final_predictions:
            writer.writerow([pred["stt"], pred["product_category"], pred["hazard_category"]])

    with zipfile.ZipFile(args.output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(args.output_csv, arcname="submission.csv")

    print(f"Dự đoán hoàn tất! Kết quả được lưu trong '{args.output_csv}', '{args.output_product_json}', '{args.output_hazard_json}', và '{args.output_zip}'.")

if __name__ == "__main__":
    main()
