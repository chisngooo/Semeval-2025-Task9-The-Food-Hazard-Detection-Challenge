import pandas as pd
import numpy as np

# Đường dẫn tới file CSV
input_file = "data.csv"

# Đọc dữ liệu
df = pd.read_csv(input_file)

# Đặt tỷ lệ tách dữ liệu
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Trộn dữ liệu ngẫu nhiên
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Tính toán số mẫu cho mỗi tập
n_total = len(df)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)

# Chia dữ liệu
train_df = df[:n_train]
val_df = df[n_train:n_train + n_val]
test_df = df[n_train + n_val:]

# Lưu các file kết quả
train_file = "train.csv"
val_file = "val.csv"
test_file = "test.csv"

train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"Dữ liệu đã được tách:\n- Train: {len(train_df)} mẫu (lưu ở '{train_file}')\n- Val: {len(val_df)} mẫu (lưu ở '{val_file}')\n- Test: {len(test_df)} mẫu (lưu ở '{test_file}')")
