import json
import csv
from collections import defaultdict
import zipfile

HAZARD_FILES = [
            "results/private/hazard/aug1_deberta_512.json",                 
            "results/private/hazard/aug1_deberta_768.json",                 
            "results/private/hazard/aug1_roberta_1280.json",                
            "results/private/hazard/aug1_roberta_512.json",         
            "results/private/hazard/aug2_deberta_512.json",
        ]
PRODUCT_FILES = [
            "results/private/product/aug1_deberta_512.json",              
            "results/private/product/aug1_deberta_768.json",               
            "results/private/product/original_deberta_multitask.json",                  
            "results/private/product/aug1_roberta_512.json",       
            "results/private/product/aug2_deberta_512.json",
        ]

# relative_weight
# HAZARD_WEIGHTS = [0.1, 0.4, 0.2, 0.2, 0.1]  
# PRODUCT_WEIGHTS = [0.1, 0.1, 0.3, 0.1, 0.1, 0.2, 0.1]

# # best each moedl
# HAZARD_WEIGHTS = [0, 0, 1, 0 , 0, 0, 0]  
# PRODUCT_WEIGHTS = [0, 0, 0, 0, 1, 0, 0]

# #grid_search weight- 1+2
# HAZARD_WEIGHTS = [0.0400, 0.0400, 0.4000, 0.4000, 0.0400, 0.0400, 0.0400]  
# PRODUCT_WEIGHTS = [0.2258, 0.0323, 0.0323, 0.2258, 0.3226, 0.1290, 0.0323]

# #grid_search weight- 2
# HAZARD_WEIGHTS = [0.2381, 0.3333, 0.0476, 0.1429, 0.0476, 0.1429, 0.0476]  
# PRODUCT_WEIGHTS = [0.1628, 0.2326, 0.0930, 0.0930, 0.1628, 0.0233, 0.2326]

# #grid_search weight- 0.8112
# HAZARD_WEIGHTS = [0.3571, 0.3571, 0.2143, 0.0714]  
# PRODUCT_WEIGHTS = [0.1250, 0.2083, 0.3750, 0.2917]

# #grid_search weight- 0.8152
# HAZARD_WEIGHTS = [0.3571, 0.3571, 0.2143, 0.0714]  
# PRODUCT_WEIGHTS = [0.2059, 0.2647, 0.2647, 0.0294, 0.2059, 0.0294]

# #grid_search weight- 0.8223
HAZARD_WEIGHTS = [0.3500, 0.3500, 0.2000, 0.0500, 0.0500]  
PRODUCT_WEIGHTS = [0.1842, 0.2632, 0.1053, 0.1842, 0.2632]


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

    for entries in zip(*data_list):  
        stt = entries[0]["stt"]

        combined_probs = defaultdict(float)
        for entry, weight in zip(entries, weights):
            probs = entry[next(iter(entry.keys() - {"stt"}))]
            for label in probs:
                combined_probs[label] += probs[label] * weight

        total = sum(combined_probs.values())
        for label in combined_probs:
            combined_probs[label] /= total

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

            hazard_label = max(hazard_probs, key=hazard_probs.get)
            product_label = max(product_probs, key=product_probs.get)

            writer.writerow([stt, hazard_label, product_label])

hazard_data_list = [normalize_probabilities(load_json(file)) for file in HAZARD_FILES]
hazard_ensembled = ensemble_probabilities(hazard_data_list, HAZARD_WEIGHTS)

product_data_list = [normalize_probabilities(load_json(file)) for file in PRODUCT_FILES]
product_ensembled = ensemble_probabilities(product_data_list, PRODUCT_WEIGHTS)

write_csv("submission.csv", hazard_ensembled, product_ensembled)

with zipfile.ZipFile("submission.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("submission.csv", arcname="submission.csv")

print("Ensemble hoàn tất! Kết quả được lưu trong submission.csv")