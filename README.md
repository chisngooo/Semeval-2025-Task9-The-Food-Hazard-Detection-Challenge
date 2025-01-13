# SemEval 2025 Task 9: The Food Hazard Detection Challenge

## Overview
This repository contains our implementation for **SemEval 2025 Task 9: The Food Hazard Detection Challenge**. The challenge focuses on **explainable classification systems** for food-incident report titles collected from the web. The goal is to develop automated systems that identify and extract food-related hazards with high transparency and explainability.

### Subtasks:
1. **ST1: Text classification for food hazard prediction**
   - Predict the type of hazard and product.
2. **ST2: Food hazard and product “vector” detection**
   - Predict the exact hazard and product for explainability.

### Scoring:
The task is evaluated using a **macro F1 score**, with a strong emphasis on hazard label accuracy.

---

## Our System

Our system focuses on **Subtask 1 (ST1)**: Text classification for food hazard prediction.

### Approach:
1. **Data Augmentation**:
   - Augmented **100 samples** for the **9 lowest product categories** and **4 lowest hazard categories** to address class imbalance.

2. **Ensemble of 14 Models**:
   - The ensemble consists of **7 models for `hazard-category`** and **7 models for `product-category`**.
   - All 14 models are variations based on two main architectures:
     - **`deberta-v3-large`**
     - **`roberta-large`**
   - The variations are achieved by applying different token chunking strategies during preprocessing.  

### Results:
- Our system achieved **Top 2 on the Public Leaderboard** during the **Evaluation Phase**, showcasing the effectiveness of our ensemble and preprocessing strategies.

---

## Data and Code

### Data:
The dataset provided for the task includes:
- **Training Data**: 5,082 labeled samples.
- **Validation Data**: 565 unlabeled samples (Conception Phase) and labeled samples (Evaluation Phase).
- **Test Data**: 997 unlabeled samples (Conception Phase) and labeled samples (Paper Phase).

All necessary data can be downloaded from the [SemEval 2025 Task 9 Landing Page](https://semeval2025.org/task9).

### Code:
- Training and prediction scripts for both `hazard-category` and `product-category` models are available in the `/src` directory.
- Ensemble strategies and final prediction aggregation scripts are also included.

---

## Timeline

1. **Trial Phase** (before September 2, 2024):
   - Access labeled trial/training data.
2. **Conception Phase** (September 2, 2024 - January 10, 2025):
   - Unlabeled validation and test data provided.
   - Codalab accepts submissions for both ST1 and ST2.
3. **Evaluation Phase** (January 10, 2025 - January 17, 2025):
   - Submit final predictions for test data.
4. **Paper Phase** (January 17, 2025 - February 28, 2025):
   - Participants submit system descriptions in scientific papers.

---

## Running the Code

### Requirements:
- Python >= 3.8
- PyTorch >= 1.11.0
- Transformers (Hugging Face) >= 4.22.0

### Setup:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/semeval2025-task9.git](https://github.com/Zhennor/Semeval-Task9-The-Food-Hazard-Detection-Challenge-2025
   cd Semeval-Task9-The-Food-Hazard-Detection-Challenge-2025
2. Train model:
   ## For Multitask Training (Recommended for combined training of both hazard-category and product-category models):
   ```bash
   python3 train_multitask.py \
     --input_file /path/to/your/train_chunk.json \
     --output_dir ./results \
     --model_output_dir ./result \
     --learning_rate 2e-5 \
     --num_epochs 10 \
     --train_batch_size 4 \
     --eval_batch_size 2 \
     --gradient_accumulation_steps 4 \
     --oversample_count 50 \
     --undersample_count 500 \
     --seed 42
   
   ## For Independent Training (When you want to train each model separately):
   ```bash
   python3 train_independent.py \
   --data_path /path/to/data.json \
   --model_path microsoft/deberta-v3-large \
   --max_length 1280 \
   --output_dir output_classification \
   --batch_size 1 \
   --learning_rate 1e-5 \
   --num_epochs 15

3. Predict:
   ```bash
   python predict.py \
      --hazard_model "huggingface_hazard_model_path" \
      --product_model "huggingface_product_model_path" \
      --input_json "private_test_512.json" \
      --output_csv "submission.csv" \
      --output_zip "submission.zip" \
      -output_hazard_json "hazard_predictions.json" \
      --output_product_json "product_predictions.json"
   
