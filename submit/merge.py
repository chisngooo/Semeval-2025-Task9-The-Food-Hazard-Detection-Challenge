import pandas as pd
import zipfile

# Đọc file CSV đầu tiên
file1 = "submission.csv"  # Thay bằng đường dẫn tới file CSV thứ nhất
df1 = pd.read_csv(file1)

# Đọc file CSV thứ hai
file2 = "sample_submit.csv"  # Thay bằng đường dẫn tới file CSV thứ hai
df2 = pd.read_csv(file2)

# Lấy cột `product` và `hazard` từ file thứ hai
columns_to_add = df2[['product', 'hazard']]

# Thêm các cột vào file thứ nhất
df1 = pd.concat([df1, columns_to_add], axis=1)

# Lưu file kết quả
output_file = "submission.csv"  # Đường dẫn file CSV kết quả
df1.to_csv(output_file, index=False)

# Nén file kết quả thành file ZIP
zip_file = "submission.zip"  # Tên file ZIP
with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(output_file)

print(f"Đã lưu file kết quả tại: {output_file}")
print(f"Đã nén file thành: {zip_file}")
