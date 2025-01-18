import os
import torch
import gdown
from transformers import AutoTokenizer, AutoModel
import json
from torch.nn import functional as F
from torch import nn
from safetensors.torch import load_file
from tqdm import tqdm
from collections import defaultdict
import argparse

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

def download_file_from_drive(url, output_path):
    """
    Download file from Google Drive
    """
    gdown.download(url, output_path, quiet=False)

def load_model_and_tokenizer(checkpoint_folder):
    tokenizer_path = f"{checkpoint_folder}/tokenizer"  # Tokenizer path
    
    # Check if the tokenizer exists at the provided checkpoint path
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Load training arguments and label mappings
    training_args = torch.load(f"{checkpoint_folder}/training_args.bin")
    try:
        with open(f"{checkpoint_folder}/label_mappings.json", 'r') as f:
            mappings = json.load(f)
    except FileNotFoundError:
        mappings = {
            'product_label_to_id': {}, 
            'hazard_label_to_id': {}   
        }
    
    product_id_to_label = {v: k for k, v in mappings['product_label_to_id'].items()}
    hazard_id_to_label = {v: k for k, v in mappings['hazard_label_to_id'].items()}
    
    model = MultiTaskClassifier(
        "microsoft/deberta-v3-base", 
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

def aggregate_by_stt(data):
    """
    Aggregate predictions by 'stt' and sum the probabilities for each label.
    """
    grouped_results = defaultdict(lambda: {"product_probs": defaultdict(float), "hazard_probs": defaultdict(float)})
    
    for row in data:
        stt = row["stt"]
        for label, prob in row["product_prediction"].items():
            grouped_results[stt]["product_probs"][label] += prob
        for label, prob in row["hazard_prediction"].items():
            grouped_results[stt]["hazard_probs"][label] += prob
            
    return grouped_results

def process_json_file(input_json_path, output_json_folder, checkpoint_folder, batch_size=8):
    """
    Process the input JSON file, perform predictions and save to separate JSON files for product and hazard probabilities
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint_folder)
    
    # Read input JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    product_results = defaultdict(list)  # Store product probabilities
    hazard_results = defaultdict(list)   # Store hazard probabilities
    
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
            # Store for separate files
            for label, prob in product_pred.items():
                product_results[label].append(prob)
            for label, prob in hazard_pred.items():
                hazard_results[label].append(prob)
    
    # Save results as two separate JSON files
    with open(f"{output_json_folder}/product_probs.json", 'w') as f:
        json.dump(product_results, f, indent=4)
    
    with open(f"{output_json_folder}/hazard_probs.json", 'w') as f:
        json.dump(hazard_results, f, indent=4)
    
    # Aggregate results by 'stt' and save separately
    aggregated_results = aggregate_by_stt(results)
    
    with open(f"{output_json_folder}/aggregated_results.json", 'w') as f:
        json.dump(aggregated_results, f, indent=4)
    
    return aggregated_results

def main(args):
    # Download tokenizer and model from Google Drive
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)

    print("Downloading model files...")
    download_file_from_drive(args.tokenizer_url, f"{args.checkpoint_folder}/tokenizer")
    download_file_from_drive(args.model_url, f"{args.checkpoint_folder}/model.safetensors")
    
    # Process input JSON and save predictions
    processed_data = process_json_file(
        args.input_json_path, 
        args.output_json_folder, 
        args.checkpoint_folder,
        args.batch_size
    )
    
    print("\nExample aggregated result for first 'stt':")
    first_stt = list(processed_data.keys())[0]
    print(f"STT: {first_stt}")
    print("Product predictions:", dict(list(processed_data[first_stt]["product_probs"].items())[:3]), "...")
    print("Hazard predictions:", dict(list(processed_data[first_stt]["hazard_probs"].items())[:3]), "...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multitask Classifier Prediction")
    parser.add_argument("--checkpoint_folder", type=str, required=True, help="Folder to store model and tokenizer")
    parser.add_argument("--input_json_path", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_json_folder", type=str, required=True, help="Folder to save output JSON files")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for predictions")
    parser.add_argument("--tokenizer_url", type=str, required=True, help="Google Drive URL to download tokenizer")
    parser.add_argument("--model_url", type=str, required=True, help="Google Drive URL to download model")

    args = parser.parse_args()
    main(args)
