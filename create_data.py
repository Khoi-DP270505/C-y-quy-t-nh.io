import pandas as pd
import random

# Tạo dữ liệu mẫu với 100 dòng bổ sung cho file animal_data.csv
data = {
    'Legs': [random.choice([0, 2, 4, 6, 8]) for _ in range(100)],
    'Habitat': [random.choice(['Đất', 'Không khí', 'Nước', 'Cây']) for _ in range(100)],
    'Cover_Type': [random.choice(['Lông', 'Vảy', 'Da trơn']) for _ in range(100)],
    'Reproduction': [random.choice(['Đẻ trứng', 'Đẻ con']) for _ in range(100)],
    'Mammal': [random.choice(['Có', 'Không']) for _ in range(100)],
    'Animal_Type': [random.choice(['Chim', 'Thú', 'Bò sát', 'Cá', 'Côn trùng', 'Lưỡng cư']) for _ in range(100)]
}

# Tạo DataFrame và lưu vào file CSV
df = pd.DataFrame(data)
df.to_csv('animal_data_extended.csv', index=False)

print("File animal_data_extended.csv đã được tạo thành công.")
