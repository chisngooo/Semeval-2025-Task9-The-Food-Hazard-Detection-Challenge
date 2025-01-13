import json
import pandas as pd
import numpy as np
import argparse
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding
)
import evaluate
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = FocalLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics_classification(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, _, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    f1 = (1 + 5*5) * recall * precision / (5*5*precision + recall)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def load_and_preprocess_data(data_path, max_length):
    with open(data_path, "r") as f:
        data_json = json.load(f)
    data = pd.DataFrame(data_json)
    
    data['full_text'] = data['text']
    X = data['full_text']
    y_hazard = data["hazard_category"]
    y_product = data["product_category"]
    
    X_train, X_val, y_train_hazard, y_val_hazard, y_train_product, y_val_product = train_test_split(
        X, y_hazard, y_product, test_size=0.2, random_state=42
    )
    
    dataset_dict = {
        'train': pd.DataFrame({
            'text': X_train,
            'hazard': y_train_hazard,
            'product': y_train_product
        }),
        'validation': pd.DataFrame({
            'text': X_val,
            'hazard': y_val_hazard,
            'product': y_val_product
        })
    }
    
    datasets = DatasetDict({
        split: Dataset.from_pandas(df) for split, df in dataset_dict.items()
    })
    
    hazard_labels = sorted(list(data["hazard_category"].unique()))
    hazard_label2id = {label: idx for idx, label in enumerate(hazard_labels)}
    hazard_id2label = {idx: label for label, idx in hazard_label2id.items()}
    
    return datasets, hazard_labels, hazard_label2id, hazard_id2label

def main():
    parser = argparse.ArgumentParser(description='Train a classification model')
    parser.add_argument('--model_path', type=str, default="microsoft/deberta-v3-large",
                        help='Path to the pretrained model')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default="output_classification",
                        help='Output directory for model checkpoints')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data JSON file')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    datasets, hazard_labels, hazard_label2id, hazard_id2label = load_and_preprocess_data(
        args.data_path, args.max_length
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length
        )
    
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    
    def encode_hazard_labels(examples):
        label = examples['hazard']
        if label not in hazard_label2id:
            return {'labels': hazard_label2id.get('other hazard', 0)}
        return {'labels': hazard_label2id[label]}
    
    hazard_datasets = tokenized_datasets.map(encode_hazard_labels, batched=False)
    
    # Initialize model
    hazard_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=len(hazard_labels),
        id2label=hazard_id2label,
        label2id=hazard_label2id
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        run_name=f"{args.model_path}_{args.max_length}_hazard_data_aug_ver2",
        output_dir=f"{args.output_dir}_{args.max_length}/hazard",
        fp16=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        report_to="wandb",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        do_eval=True,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        learning_rate=args.learning_rate,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}_{args.max_length}/hazard/logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    
    trainer = CustomTrainer(
        model=hazard_model,
        args=training_args,
        train_dataset=hazard_datasets['train'],
        eval_dataset=hazard_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_classification,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()