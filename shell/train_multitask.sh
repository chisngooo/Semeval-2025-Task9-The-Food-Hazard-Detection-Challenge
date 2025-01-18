input_file="./data/train.json"
output_dir="./results"
model_output_dir="./model"
learning_rate=2e-5
num_epochs=10
train_batch_size=8
eval_batch_size=4
gradient_accumulation_steps=2
oversample_count=50
undersample_count=500
seed=42

python3 train_multitask.py \
  --input_file "$input_file" \
  --output_dir "$output_dir" \
  --model_output_dir "$model_output_dir" \
  --learning_rate "$learning_rate" \
  --num_epochs "$num_epochs" \
  --train_batch_size "$train_batch_size" \
  --eval_batch_size "$eval_batch_size" \
  --gradient_accumulation_steps "$gradient_accumulation_steps" \
  --oversample_count "$oversample_count" \
  --undersample_count "$undersample_count" \
  --seed "$seed"