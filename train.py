import pandas as pd
import json
import argparse
from sklearn.utils import resample
from datasets import Dataset, Features, Value, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
import os
import shutil
from transformers import AutoModel
from torch import nn
from collections import Counter
import numpy as np
from torch.nn import functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Task Classification Training')
    parser.add_argument('--input_file', type=str, default="/kaggle/input/chain-chunk-512/train_chunk.json",
                      help='Path to input JSON file')
    parser.add_argument('--output_dir', type=str, default="./results",
                      help='Directory for saving training results')
    parser.add_argument('--model_output_dir', type=str, default="./result",
                      help='Directory for saving the final model')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=4,
                      help='Training batch size per device')
    parser.add_argument('--eval_batch_size', type=int, default=2,
                      help='Evaluation batch size per device')
    parser.add_argument('--oversample_count', type=int, default=50,
                      help='Number of samples to add for oversampling')
    parser.add_argument('--undersample_count', type=int, default=500,
                      help='Number of samples to remove for undersampling')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    return parser.parse_args()

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

class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name, product_num_labels, hazard_num_labels, 
                 product_class_weights=None, hazard_class_weights=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size
        
        self.product_classifier = nn.Linear(hidden_size, product_num_labels)
        self.hazard_classifier = nn.Linear(hidden_size, hazard_num_labels)
        
        self.product_class_weights = product_class_weights
        self.hazard_class_weights = hazard_class_weights
        
        self.focal_loss = FocalLoss(gamma=2)

    def forward(self, input_ids, attention_mask, product_labels=None, hazard_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)

        product_logits = self.product_classifier(pooled_output)
        hazard_logits = self.hazard_classifier(pooled_output)

        loss = None
        if product_labels is not None and hazard_labels is not None:
            if self.product_class_weights is not None:
                product_loss = CrossEntropyLoss(weight=self.product_class_weights)(
                    product_logits.view(-1, product_logits.size(-1)),
                    product_labels.view(-1)
                )
            else:
                product_loss = self.focal_loss(product_logits, product_labels)

            if self.hazard_class_weights is not None:
                hazard_loss = CrossEntropyLoss(weight=self.hazard_class_weights)(
                    hazard_logits.view(-1, hazard_logits.size(-1)),
                    hazard_labels.view(-1)
                )
            else:
                hazard_loss = self.focal_loss(hazard_logits, hazard_labels)

            loss = 0.5 * product_loss + 0.5 * hazard_loss

        return {
            'loss': loss if loss is not None else torch.tensor(0.0).to(input_ids.device),
            'product_logits': product_logits,
            'hazard_logits': hazard_logits
        }

def calculate_class_weights(labels):
    count = Counter(labels)
    total = sum(count.values())
    weights = {label: total / (count[label] * len(count)) for label in count}
    return weights

def process_categories(df, category_column, oversample_count, undersample_count, oversample_n_last, undersample_n_first):
    label_counts = df[category_column].value_counts().sort_values(ascending=False)
    sorted_labels = label_counts.index.tolist()

    oversample_labels = sorted_labels[-oversample_n_last:]  
    undersample_labels = sorted_labels[:undersample_n_first]  

    dfs = []
    for label in oversample_labels:
        df_label = df[df[category_column] == label]
        oversampled = resample(
            df_label,
            replace=True,
            n_samples=len(df_label) + oversample_count,
            random_state=42
        )
        dfs.append(oversampled)
    
    for label in undersample_labels:
        df_label = df[df[category_column] == label]
        undersampled = resample(
            df_label,
            replace=False,
            n_samples=max(0, len(df_label) - undersample_count),
            random_state=42
        )
        dfs.append(undersampled)
    
    remaining_labels = set(df[category_column].unique()) - set(oversample_labels + undersample_labels)
    for label in remaining_labels:
        dfs.append(df[df[category_column] == label])
    balanced_df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df

def compute_metrics(pred):
    product_preds, hazard_preds = pred.predictions
    product_labels, hazard_labels = pred.label_ids
    
    product_preds = torch.tensor(product_preds).argmax(-1)
    hazard_preds = torch.tensor(hazard_preds).argmax(-1)
    
    product_precision, product_recall, product_f1, _ = precision_recall_fscore_support(
        product_labels, product_preds, average='weighted'
    )
    product_acc = accuracy_score(product_labels, product_preds)
    
    hazard_precision, hazard_recall, hazard_f1, _ = precision_recall_fscore_support(
        hazard_labels, hazard_preds, average='weighted'
    )
    hazard_acc = accuracy_score(hazard_labels, hazard_preds)
    
    product_macro_f1 = precision_recall_fscore_support(
        product_labels, product_preds, average='macro'
    )[2]
    hazard_macro_f1 = precision_recall_fscore_support(
        hazard_labels, hazard_preds, average='macro'
    )[2]
    
    avg_f1 = (product_f1 + hazard_f1) / 2
    avg_macro_f1 = (product_macro_f1 + hazard_macro_f1) / 2
    
    return {
        "product_accuracy": product_acc,
        "product_f1": product_f1,
        "product_macro_f1": product_macro_f1,
        "product_precision": product_precision,
        "product_recall": product_recall,
        "hazard_accuracy": hazard_acc,
        "hazard_f1": hazard_f1,
        "hazard_macro_f1": hazard_macro_f1,
        "hazard_precision": hazard_precision,
        "hazard_recall": hazard_recall,
        "avg_f1": avg_f1,
        "avg_macro_f1": avg_macro_f1
    }

def main():
    args = parse_args()

    with open(args.input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    df = pd.DataFrame(data)

    print("Initial Product Category Distribution:")
    print(df['product_category'].value_counts())
    print("\nInitial Hazard Category Distribution:")
    print(df['hazard_category'].value_counts())

    product_weights = calculate_class_weights(df['product_category'])
    hazard_weights = calculate_class_weights(df['hazard_category'])

    df_balanced = df.copy()
    df_balanced = process_categories(
        df_balanced,
        category_column='product_category',
        oversample_count=args.oversample_count,
        undersample_count=args.undersample_count,
        oversample_n_last=9,
        undersample_n_first=1
    )

    df_balanced = process_categories(
        df_balanced,
        category_column='hazard_category',
        oversample_count=args.oversample_count,
        undersample_count=args.undersample_count,
        oversample_n_last=4,
        undersample_n_first=2
    )

    df_balanced = df_balanced.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    product_labels = sorted(df_balanced['product_category'].unique().tolist())
    hazard_labels = sorted(df_balanced['hazard_category'].unique().tolist())

    product_label_to_id = {label: idx for idx, label in enumerate(product_labels)}
    hazard_label_to_id = {label: idx for idx, label in enumerate(hazard_labels)}

    df_balanced['product_labels'] = df_balanced['product_category'].map(product_label_to_id)
    df_balanced['hazard_labels'] = df_balanced['hazard_category'].map(hazard_label_to_id)

    product_weight_tensor = torch.tensor(
        [product_weights[label] for label in product_labels],
        dtype=torch.float
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    hazard_weight_tensor = torch.tensor(
        [hazard_weights[label] for label in hazard_labels],
        dtype=torch.float
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    features = Features({
        'stt': Value('int64'),
        'chunk_id': Value('string'),
        'text': Value('string'),
        'product_category': Value('string'),
        'hazard_category': Value('string'),
        'product_labels': Value('int64'),
        'hazard_labels': Value('int64')
    })

    dataset = Dataset.from_pandas(df_balanced, features=features)

    model_name = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=args.seed)

    model = MultiTaskClassifier(
        model_name,
        product_num_labels=len(product_labels),
        hazard_num_labels=len(hazard_labels)
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="avg_macro_f1",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test_split["train"],
        eval_dataset=train_test_split["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    evaluation_results = trainer.evaluate()
    print("Evaluation Results:", evaluation_results)

    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)

    trainer.save_model(args.model_output_dir)
    print(f"Model saved to {args.model_output_dir}")

    mappings = {
        'product_label_to_id': product_label_to_id,
        'hazard_label_to_id': hazard_label_to_id
    }
    with open(os.path.join(args.model_output_dir, 'label_mappings.json'), 'w') as f:
        json.dump(mappings, f)

    zip_file_path = f"{args.model_output_dir}.zip"
    shutil.make_archive(zip_file_path.replace('.zip', ''), 'zip', args.model_output_dir)
    print(f"Model and results are zipped as {zip_file_path}")

if __name__ == "__main__":
    main()