import torch
from transformers import AutoTokenizer, AutoModel 
import json
from torch.nn import functional as F
from torch import nn
from safetensors.torch import load_file
from tqdm import tqdm

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
        with open("archive/label_mappings.json", 'r') as f:
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
    
    print("\nExample prediction for first record:")
    print(f"Text: {processed_data[0]['text'][:100]}...")
    print("Product predictions:", dict(list(processed_data[0]['product_prediction'].items())[:3]), "...")
    print("Hazard predictions:", dict(list(processed_data[0]['hazard_prediction'].items())[:3]), "...")