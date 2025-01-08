import json
import csv
from collections import defaultdict
import zipfile

# Đường dẫn đến các tệp JSON đầu vào
HAZARD_FILES = [
    "results/hazard/hazard_predictions_l_1024.json",
    "results/hazard/hazard_predictions_large-512.json",
    "results/hazard/hazard_predictions_base-512.json",
    "results/hazard/hazard_probabilities_ens.json"
    
]
PRODUCT_FILES = [
    "results/product/product_predictions_l_512.json",
    "results/product/product_predictions_large-1024.json",
    "results/product/product_predictions_base-512.json",
    "results/product/product_probabilities_ens.json"
    
]
OUTPUT_HAZARD_ENSEMBLED = "hazard_predictions_ensembled.json"
OUTPUT_PRODUCT_ENSEMBLED = "product_predictions_ensembled.json"
OUTPUT_CSV = "submission.csv"

# best-weight
# HAZARD_WEIGHTS = [0.5, 0.1, 0.15, 0.25]  # Thêm trọng số vào nếu có thêm tệp hazard
# PRODUCT_WEIGHTS = [0.55, 0.1, 0.15, 0.2]   

# Trọng số mặc định cho mỗi tệp
HAZARD_WEIGHTS = [0.55, 0.1, 0.15, 0.2]  # Thêm trọng số vào nếu có thêm tệp hazard
PRODUCT_WEIGHTS = [0.55, 0.1, 0.15, 0.2]   # Thêm trọng số vào nếu có thêm tệp product

 
def load_json(file_path):
    """Tải dữ liệu từ tệp JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_probabilities(data):
    """Chuẩn hóa xác suất để tổng các giá trị bằng 1."""
    for entry in data:
        probabilities = entry[next(iter(entry.keys() - {"stt"}))]
        total = sum(probabilities.values())
        for label in probabilities:
            probabilities[label] /= total
    return data

def ensemble_probabilities(data_list, weights):
    """Ensemble xác suất của nhiều tập dữ liệu với trọng số tương ứng."""
    ensembled_data = []

    for entries in zip(*data_list):  # Sử dụng zip để kết hợp các entry từ các tập dữ liệu
        stt = entries[0]["stt"]

        # Ensemble xác suất cho mỗi tệp
        combined_probs = defaultdict(float)
        for entry, weight in zip(entries, weights):
            probs = entry[next(iter(entry.keys() - {"stt"}))]
            for label in probs:
                combined_probs[label] += probs[label] * weight

        # Chuẩn hóa xác suất
        total = sum(combined_probs.values())
        for label in combined_probs:
            combined_probs[label] /= total

        # Lưu kết quả đã ensemble
        ensembled_data.append({
            "stt": stt,
            next(iter(entries[0].keys() - {"stt"})): dict(combined_probs)
        })

    return ensembled_data

def save_json(file_path, data):
    """Lưu dữ liệu vào tệp JSON."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def write_csv(file_path, hazard_data, product_data):
    """Lưu kết quả ensemble vào file CSV."""
    with open(file_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["stt", "hazard-category", "product-category"])

        for h_entry, p_entry in zip(hazard_data, product_data):
            stt = h_entry["stt"]
            hazard_probs = h_entry["hazard_probabilities"]
            product_probs = p_entry["product_probabilities"]

            # Lấy nhãn có xác suất cao nhất
            hazard_label = max(hazard_probs, key=hazard_probs.get)
            product_label = max(product_probs, key=product_probs.get)

            writer.writerow([stt, hazard_label, product_label])

# Tải và chuẩn hóa dữ liệu hazard
hazard_data_list = [normalize_probabilities(load_json(file)) for file in HAZARD_FILES]
hazard_ensembled = ensemble_probabilities(hazard_data_list, HAZARD_WEIGHTS)
save_json(OUTPUT_HAZARD_ENSEMBLED, hazard_ensembled)

# Tải và chuẩn hóa dữ liệu product
product_data_list = [normalize_probabilities(load_json(file)) for file in PRODUCT_FILES]
product_ensembled = ensemble_probabilities(product_data_list, PRODUCT_WEIGHTS)
save_json(OUTPUT_PRODUCT_ENSEMBLED, product_ensembled)

# Lưu kết quả cuối cùng vào CSV
write_csv(OUTPUT_CSV, hazard_ensembled, product_ensembled)

# Nén file CSV vào một tệp ZIP
with zipfile.ZipFile("submission.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(OUTPUT_CSV, arcname="submission.csv")

print(f"Ensemble hoàn tất! Kết quả được lưu trong '{OUTPUT_HAZARD_ENSEMBLED}', '{OUTPUT_PRODUCT_ENSEMBLED}', và '{OUTPUT_CSV}'")
