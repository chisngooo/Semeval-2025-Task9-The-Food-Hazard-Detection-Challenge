import json
from collections import defaultdict

# Load label mapping
label_mapping_file = "data/label_mappings.json"
with open(label_mapping_file, "r", encoding="utf-8") as f:
    label_mapping = json.load(f)

product_label_to_id = {str(v): k for k, v in label_mapping["product_label_to_id"].items()}
hazard_label_to_id = {str(v): k for k, v in label_mapping["hazard_label_to_id"].items()}

# Hàm gộp theo STT từ một file
def aggregate_by_stt(data):
    grouped_results = defaultdict(lambda: {"product_probs": defaultdict(float), "hazard_probs": defaultdict(float)})
    for row in data:
        stt = row["stt"]
        for label, prob in row["product_prediction"].items():
            grouped_results[stt]["product_probs"][label] += prob
        for label, prob in row["hazard_prediction"].items():
            grouped_results[stt]["hazard_probs"][label] += prob
    return grouped_results

# Đọc file JSON
files = [
"predictions_with_stt_3370.json"
]

weights = [1]
grouped_results_list = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        grouped_results_list.append(aggregate_by_stt(data))

# Gộp kết quả từ các file với trọng số
combined_results = defaultdict(lambda: {"product_probs": defaultdict(float), "hazard_probs": defaultdict(float)})

for grouped_results, weight in zip(grouped_results_list, weights):
    for stt, probs in grouped_results.items():
        for label, prob in probs["product_probs"].items():
            combined_results[stt]["product_probs"][label] += prob * weight
        for label, prob in probs["hazard_probs"].items():
            combined_results[stt]["hazard_probs"][label] += prob * weight

# Tạo file kết quả đầu ra
final_results_product = []
final_results_hazard = []

for stt, probs in combined_results.items():
    # Kết quả cho product_probabilities (convert từ ID -> label)
    product_probabilities = {product_label_to_id[label]: prob for label, prob in probs["product_probs"].items()}
    final_results_product.append({
        "stt": int(stt),
        "product_probabilities": product_probabilities,
    })
    
    # Kết quả cho hazard_probabilities (convert từ ID -> label)
    hazard_probabilities = {hazard_label_to_id[label]: prob for label, prob in probs["hazard_probs"].items()}
    final_results_hazard.append({
        "stt": int(stt),
        "hazard_probabilities": hazard_probabilities,
    })

# Xuất ra file JSON
output_product_file = "results/public/product/product_probabilities_3370.json"
output_hazard_file = "results/public/hazard/hazard_probabilities_3370.json"

with open(output_product_file, "w", encoding="utf-8") as f:
    json.dump(final_results_product, f, ensure_ascii=False, indent=4)

with open(output_hazard_file, "w", encoding="utf-8") as f:
    json.dump(final_results_hazard, f, ensure_ascii=False, indent=4)

print(f"Kết quả đã được lưu vào '{output_product_file}' và '{output_hazard_file}'.")
