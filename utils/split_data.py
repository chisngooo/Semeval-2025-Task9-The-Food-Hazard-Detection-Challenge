import pandas as pd
from sklearn.model_selection import train_test_split

input_file = "../data/data_preprocessed.csv"
df = pd.read_csv(input_file)

train_ratio = 0.8
val_ratio = 0.2

train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=42)

train_file = "train.csv"
val_file = "val.csv"

train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)

print(f"Dữ liệu đã được tách:\n- Train: {len(train_df)} mẫu (lưu ở '{train_file}')\n- Val: {len(val_df)} mẫu (lưu ở '{val_file}')")
