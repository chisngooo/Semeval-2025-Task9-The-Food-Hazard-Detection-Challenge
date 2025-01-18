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

def load_and_preprocess_data(data_path, max_length, task):
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
    
    if task == 'hazard':
        labels = sorted(list(data["hazard_category"].unique()))
    else:  
        labels = sorted(list(data["product_category"].unique()))
        
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    return datasets, labels, label2id, id2label

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
    parser.add_argument('--task', type=str, choices=['hazard', 'product'], required=True,
                        help='Classification task: hazard-category or product-category')
    
    args = parser.parse_args()
    
    datasets, labels, label2id, id2label = load_and_preprocess_data(
        args.data_path, args.max_length, args.task
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length
        )
    
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    
    def encode_labels(examples):
        label = examples[args.task]  
        if label not in label2id:
            return {'labels': label2id.get(f'other {args.task}', 0)}
        return {'labels': label2id[label]}
    
    labeled_datasets = tokenized_datasets.map(encode_labels, batched=False)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        run_name=f"{args.model_path}_{args.max_length}_{args.task}_data_aug_ver2",
        output_dir=f"{args.output_dir}_{args.max_length}/{args.task}",
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
        logging_dir=f"{args.output_dir}_{args.max_length}/{args.task}/logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=labeled_datasets['train'],
        eval_dataset=labeled_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_classification,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()