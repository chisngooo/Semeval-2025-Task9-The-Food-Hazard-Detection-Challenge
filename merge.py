import pandas as pd
import zipfile

# Đọc hai file CSV
file1 = "submission.csv"
file2 = "sample_submit.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Kiểm tra cột `stt` để thay thế các cột `hazard-category` và `product-category`
df2 = df2.drop(columns=["hazard-category", "product-category"], errors="ignore")  # Xóa 2 cột cũ (nếu có)
df2 = df2.merge(df1[["stt", "hazard-category", "product-category"]], on="stt", how="left")

# Lưu lại file CSV đã thay thế
output_file = "submission.csv"
df2.to_csv(output_file, index=False)

# Tạo file zip và thêm file CSV vào trong đó
zip_filename = "submission.zip"
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(output_file)

print(f"File đã được cập nhật và nén vào: {zip_filename}")
