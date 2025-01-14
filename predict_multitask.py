import torch
from transformers import AutoTokenizer, AutoModel 
import json
from torch.nn import functional as F
from torch import nn
from safetensors.torch import load_file
from tqdm import tqdm
import json
from collections import defaultdict

class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name, product_num_labels, hazard_num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size
        
        self.product_classifier = nn.Linear(hidden_size, product_num_labels)
        self.hazard_classifier = nn.Linear(hidden_size, hazard_num_labels)

    def forward(self, input_ids, attention_mask, product_labels=None, hazard_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)

        product_logits = self.product_classifier(pooled_output)
        hazard_logits = self.hazard_classifier(pooled_output)

        return {
            'loss': None,
            'product_logits': product_logits,
            'hazard_logits': hazard_logits
        }

def load_model_and_tokenizer(checkpoint_folder):
    tokenizer_path = f"{checkpoint_folder}"  # Make sure the tokenizer exists at this path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Load training arguments and label mappings
    training_args = torch.load(f"{checkpoint_folder}/training_args.bin")
    try:
        with open("data/label_mappings.json", 'r') as f:
            mappings = json.load(f)
    except FileNotFoundError:
        mappings = {
            'product_label_to_id': {}, 
            'hazard_label_to_id': {}   
        }
    
    product_id_to_label = {v: k for k, v in mappings['product_label_to_id'].items()}
    hazard_id_to_label = {v: k for k, v in mappings['hazard_label_to_id'].items()}
    
    model = MultiTaskClassifier(
        "microsoft/deberta-v3-large", 
        product_num_labels=len(mappings['product_label_to_id']),
        hazard_num_labels=len(mappings['hazard_label_to_id'])
    )
    
    state_dict = load_file(f"{checkpoint_folder}/model.safetensors")
    model.load_state_dict(state_dict)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model, tokenizer


def process_batch(texts, model, tokenizer):
    """
    Process a batch of texts and return probabilities
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
        return_token_type_ids=False 
    )
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        product_logits = outputs['product_logits']
        hazard_logits = outputs['hazard_logits']
        
        product_probs = F.softmax(product_logits, dim=-1)
        hazard_probs = F.softmax(hazard_logits, dim=-1)
        
    return product_probs.cpu(), hazard_probs.cpu()

def process_json_file(input_json_path, output_json_path, checkpoint_folder, batch_size=8):
    """
    Process the input JSON file, perform predictions and save to output JSON file
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint_folder)
    
    # Read input JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(data), batch_size)):
        batch_texts = [item['text'] for item in data[i:i + batch_size]]
        
        # Get predictions for the batch
        product_probs, hazard_probs = process_batch(batch_texts, model, tokenizer)
        
        # Convert predictions to labels
        for j, item in enumerate(data[i:i + batch_size]):
            product_pred = {str(idx): float(prob) for idx, prob in enumerate(product_probs[j])}
            hazard_pred = {str(idx): float(prob) for idx, prob in enumerate(hazard_probs[j])}
            
            item['product_prediction'] = product_pred
            item['hazard_prediction'] = hazard_pred
            
            results.append(item)
    
    # Save results to output JSON
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def aggregate_by_stt(data):
    grouped_results = defaultdict(lambda: {"product_probs": defaultdict(float), "hazard_probs": defaultdict(float)})
    for row in data:
        stt = row["stt"]
        for label, prob in row["product_prediction"].items():
            grouped_results[stt]["product_probs"][label] += prob
        for label, prob in row["hazard_prediction"].items():
            grouped_results[stt]["hazard_probs"][label] += prob
    return grouped_results


if __name__ == "__main__":
    checkpoint_folder = "checkpoint-multitask/checkpoint-3370"  
    input_json_path = "data/public_test_512.json"  
    output_json_path = "predictions_with_stt3370.json"  
    batch_size = 2 
    
    processed_data = process_json_file(
        input_json_path, 
        output_json_path, 
        checkpoint_folder,
        batch_size
    )
    

    label_mapping_file = "data/label_mappings.json"
    with open(label_mapping_file, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)

    product_label_to_id = {str(v): k for k, v in label_mapping["product_label_to_id"].items()}
    hazard_label_to_id = {str(v): k for k, v in label_mapping["hazard_label_to_id"].items()}


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
