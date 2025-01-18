
model_path="microsoft/deberta-v3-large"
data_path="data/train.json"
output_dir="output/checkpoints"
task="hazard"
max_length="512"
batch_size="8"
learning_rate="2e-5"
num_epochs="15"

cd ..

python3 train_independent.py \
    --model_path "$model_path" \
    --data_path "$data_path" \
    --output_dir "$output_dir" \
    --task "$task" \
    --max_length "$max_length" \
    --batch_size "$batch_size" \
    --learning_rate "$learning_rate" \
    --num_epochs "$num_epochs"
