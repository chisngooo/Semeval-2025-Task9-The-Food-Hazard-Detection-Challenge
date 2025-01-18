import torch
from transformers import AutoTokenizer, AutoModel 
import json
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm
import json
from collections import defaultdict
from safetensors.torch import load_file  
from huggingface_hub import hf_hub_download
import argparse
import os

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

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)    
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
        "microsoft/deberta-v3-large" , 
        product_num_labels=len(mappings['product_label_to_id']),
        hazard_num_labels=len(mappings['hazard_label_to_id'])
    )
    try:
        safetensors_file = hf_hub_download(
            repo_id=model_name, 
            filename="model.safetensors"
        )
        state_dict = load_file(safetensors_file)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Lỗi khi tải trọng số từ safetensors: {e}")
    model.eval()  
    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer

def process_batch(texts, model, tokenizer):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
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

def process_json_file(input_json_path, output_json_path, model_name, batch_size=8):
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    
    for i in tqdm(range(0, len(data), batch_size)):
        batch_texts = [item['text'] for item in data[i:i + batch_size]]
        product_probs, hazard_probs = process_batch(batch_texts, model, tokenizer)
        
        for j, item in enumerate(data[i:i + batch_size]):
            product_pred = {str(idx): float(prob) for idx, prob in enumerate(product_probs[j])}
            hazard_pred = {str(idx): float(prob) for idx, prob in enumerate(hazard_probs[j])}
            
            item['product_prediction'] = product_pred
            item['hazard_prediction'] = hazard_pred
            
            results.append(item)
    
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

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-task Classification for Products and Hazards')
    
    parser.add_argument('--model_name', type=str, default="Quintu/deberta-multitask-v0",
                      help='HuggingFace model name or path (default: Quintu/deberta-multitask-v0)')
    parser.add_argument('--input_json', type=str, required=True,
                      help='Path to input JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save output files')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for processing (default: 2)')
    parser.add_argument('--label_mapping', type=str, default="data/label_mappings.json",
                      help='Path to label mapping file (default: data/label_mappings.json)')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    intermediate_output = f"{args.output_dir}/predictions_intermediate.json"
    processed_data = process_json_file(
        args.input_json,
        intermediate_output,
        args.model_name,
        args.batch_size
    )
    
    with open(args.label_mapping, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)

    product_label_to_id = {str(v): k for k, v in label_mapping["product_label_to_id"].items()}
    hazard_label_to_id = {str(v): k for k, v in label_mapping["hazard_label_to_id"].items()}

    grouped_results = aggregate_by_stt(processed_data)
    
    final_results_product = []
    final_results_hazard = []

    for stt, probs in grouped_results.items():
        product_probabilities = {product_label_to_id[label]: prob for label, prob in probs["product_probs"].items()}
        final_results_product.append({
            "stt": int(stt),
            "product_probabilities": product_probabilities,
        })
        
        hazard_probabilities = {hazard_label_to_id[label]: prob for label, prob in probs["hazard_probs"].items()}
        final_results_hazard.append({
            "stt": int(stt),
            "hazard_probabilities": hazard_probabilities,
        })

    # Save results
    output_product_file = f"{args.output_dir}/product/product_probabilities.json"
    output_hazard_file = f"{args.output_dir}/hazard/hazard_probabilities.json"

    # Ensure directories exist
    os.makedirs(os.path.dirname(output_product_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_hazard_file), exist_ok=True)

    with open(output_product_file, "w", encoding="utf-8") as f:
        json.dump(final_results_product, f, ensure_ascii=False, indent=4)

    with open(output_hazard_file, "w", encoding="utf-8") as f:
        json.dump(final_results_hazard, f, ensure_ascii=False, indent=4)

    print(f"Results saved to '{output_product_file}' and '{output_hazard_file}'")