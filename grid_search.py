import itertools
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import csv
import json

def load_ground_truth(csv_file):
    """Load ground truth từ file CSV."""
    hazard_cats, product_cats = [], []
    with open(csv_file, "r", encoding="utf-8") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            hazard_cats.append(row['hazard-category'])
            product_cats.append(row['product-category'])
    return np.array(hazard_cats), np.array(product_cats)

def compute_macro_f1_score(true_labels, pred_labels):
    """Tính macro F1-score."""
    return f1_score(true_labels, pred_labels, average='macro')

def load_json(file_path):
    """Load JSON file containing model predictions."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        raise

def normalize_probabilities(predictions):
    """Chuẩn hóa probabilities cho mỗi mẫu."""
    normalized = []
    for pred in predictions:
        if "hazard_probabilities" in pred:
            hazard_sum = sum(pred["hazard_probabilities"].values())
            hazard_probs = {k: v/hazard_sum for k, v in pred["hazard_probabilities"].items()}
            pred["hazard_probabilities"] = hazard_probs
        if "product_probabilities" in pred:
            product_sum = sum(pred["product_probabilities"].values())
            product_probs = {k: v/product_sum for k, v in pred["product_probabilities"].items()}
            pred["product_probabilities"] = product_probs
        normalized.append(pred)
    return normalized

def ensemble_probabilities(data_list, weights):
    """Kết hợp predictions từ nhiều models với weights."""
    ensembled = []
    for idx in range(len(data_list[0])):
        entry = {}
        # Ensemble hazard probabilities
        if "hazard_probabilities" in data_list[0][idx]:
            hazard_probs = {}
            for label in data_list[0][idx]["hazard_probabilities"].keys():
                weighted_sum = sum(
                    weights[i] * data[idx]["hazard_probabilities"].get(label, 0)
                    for i, data in enumerate(data_list)
                )
                hazard_probs[label] = weighted_sum
            entry["hazard_probabilities"] = hazard_probs
        # Ensemble product probabilities
        if "product_probabilities" in data_list[0][idx]:
            product_probs = {}
            for label in data_list[0][idx]["product_probabilities"].keys():
                weighted_sum = sum(
                    weights[i] * data[idx]["product_probabilities"].get(label, 0)
                    for i, data in enumerate(data_list)
                )
                product_probs[label] = weighted_sum
            entry["product_probabilities"] = product_probs
        ensembled.append(entry)
    return ensembled

def grid_search_hazard(hazard_files, ground_truth_csv, weight_range):
    """Grid search để tìm best weight cho hazard."""
    hazard_cats_true, _ = load_ground_truth(ground_truth_csv)
    hazard_data_list = [normalize_probabilities(load_json(file)) for file in hazard_files]
    
    best_score = -1
    best_weights = None
    
    total_combinations = len(list(itertools.product(weight_range, repeat=len(hazard_files))))
    
    for hazard_weights in tqdm(
        itertools.product(weight_range, repeat=len(hazard_files)),
        desc="Searching best hazard weights",
        total=total_combinations
    ):
        hazard_weights = np.array(hazard_weights) / sum(hazard_weights)
        hazard_ensembled = ensemble_probabilities(hazard_data_list, hazard_weights)
        hazards_pred = [
            max(entry["hazard_probabilities"], key=entry["hazard_probabilities"].get)
            for entry in hazard_ensembled
        ]
        score = compute_macro_f1_score(hazard_cats_true, hazards_pred)
        if score > best_score:
            best_score = score
            best_weights = hazard_weights
    return best_score, best_weights

def grid_search_product(product_files, ground_truth_csv, weight_range):
    """Grid search để tìm best weight cho product."""
    _, product_cats_true = load_ground_truth(ground_truth_csv)
    product_data_list = [normalize_probabilities(load_json(file)) for file in product_files]
    
    best_score = -1
    best_weights = None
    
    total_combinations = len(list(itertools.product(weight_range, repeat=len(product_files))))
    
    for product_weights in tqdm(
        itertools.product(weight_range, repeat=len(product_files)),
        desc="Searching best product weights",
        total=total_combinations
    ):
        product_weights = np.array(product_weights) / sum(product_weights)
        product_ensembled = ensemble_probabilities(product_data_list, product_weights)
        products_pred = [
            max(entry["product_probabilities"], key=entry["product_probabilities"].get)
            for entry in product_ensembled
        ]
        score = compute_macro_f1_score(product_cats_true, products_pred)
        if score > best_score:
            best_score = score
            best_weights = product_weights
    return best_score, best_weights

if __name__ == "__main__":
    # Define files and parameters
    HAZARD_FILES = [
            "results/hazard/hazard_predictions_large-512-v2.json",                 #data aug ver1
            "results/hazard/hazard_predictions_large-768.json",                 #data aug ver2
            "results/hazard/hazard_predictions_large-1024.json",                #data aug ver1
            "results/hazard/hazard_predictions_large-1280.json",                #data aug ver2
            "results/hazard/hazard_probabilities_3145.json",                    #multitask under and over sample 
            "results/hazard/hazard_predictions_roberta-large-512.json",
            "results/hazard/hazard_predictions_roberta-large-1024.json",
            "results/hazard/hazard_predict_LLM-1.json"
        ]
    PRODUCT_FILES = [
            "results/product/product_predictions_large-512-v2.json",               #data aug ver 1
            "results/product/product_predictions_large-768.json",               #data aug ver2
            "results/product/product_predictions_large-1024.json",              #data aug ver 1
            "results/product/product_predictions_large-1280.json",              #data aug ver2
            "results/product/product_probabilities_3145.json",                  #multitask under and over sample 
            "results/product/product_predictions_robert-large-512.json",
            "results/product/product_predictions_roberta-large-1024.json",
            "results/product/product_predict_LLM-3.json"
        ]
        # Define ground truth file
    GROUND_TRUTH_FILE = "incidents_valid.csv"
    
    # Define weight range (0.1 to 1.0 với step 0.1)
    WEIGHT_RANGE = np.arange(0.1, 1.1, 0.4)

    print("\nStarting hazard category grid search...")
    try:
        hazard_best_score, hazard_best_weights = grid_search_hazard(
            hazard_files=HAZARD_FILES,
            ground_truth_csv=GROUND_TRUTH_FILE,
            weight_range=WEIGHT_RANGE
        )
        print("\nBest hazard category macro F1 score:", hazard_best_score)
        print("\nHazard weights to model mapping:")
        for weight, file in zip(hazard_best_weights, HAZARD_FILES):
            print(f"  {file}: {weight:.4f}")
        print("\nBest hazard weights list:")
        print(", ".join([f"{weight:.4f}" for weight in hazard_best_weights]))
    except Exception as e:
        print(f"Error in hazard grid search: {str(e)}")

    print("\nStarting product category grid search...")
    try:
        product_best_score, product_best_weights = grid_search_product(
            product_files=PRODUCT_FILES,
            ground_truth_csv=GROUND_TRUTH_FILE,
            weight_range=WEIGHT_RANGE
        )
        print("\nBest product category macro F1 score:", product_best_score)
        print("\nProduct weights to model mapping:")
        for weight, file in zip(product_best_weights, PRODUCT_FILES):
            print(f"  {file}: {weight:.4f}")
        print("\nBest product weights list:")
        print(", ".join([f"{weight:.4f}" for weight in product_best_weights]))
    except Exception as e:
        print(f"Error in product grid search: {str(e)}")