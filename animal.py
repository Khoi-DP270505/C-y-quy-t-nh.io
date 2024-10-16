import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset (giả sử dữ liệu động vật đã được chuẩn bị dưới dạng DataFrame)
df = pd.read_csv('animal_data_extended.csv')

# Sử dụng LabelEncoder để chuyển đổi các cột dạng chuỗi thành số
label_encoder = LabelEncoder()
df['Habitat'] = label_encoder.fit_transform(df['Habitat'])
df['Cover_Type'] = label_encoder.fit_transform(df['Cover_Type'])
df['Reproduction'] = label_encoder.fit_transform(df['Reproduction'])
df['Mammal'] = label_encoder.fit_transform(df['Mammal'])
df['Animal_Type'] = label_encoder.fit_transform(df['Animal_Type'])

# Chọn đặc trưng và biến mục tiêu
features = ['Legs', 'Habitat', 'Cover_Type', 'Reproduction', 'Mammal']
target = 'Animal_Type'
X = df[features]
y = df[target]

# Chia bộ dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tạo mô hình cây quyết định sử dụng Gini Index
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Huấn luyện mô hình sử dụng Gini Index
clf_gini.fit(X_train, y_train)

# Tính toán độ chính xác cho Gini Index
accuracy_gini = clf_gini.score(X_test, y_test)
print(f"Accuracy using Gini Index: {accuracy_gini * 100:.2f}%")

# Tạo mô hình cây quyết định sử dụng Entropy (Information Gain)
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)

# Huấn luyện mô hình sử dụng Entropy
clf_entropy.fit(X_train, y_train)

# Tính toán độ chính xác cho Entropy
accuracy_entropy = clf_entropy.score(X_test, y_test)
print(f"Accuracy using Information Gain: {accuracy_entropy * 100:.2f}%")

# Tầm quan trọng của đặc trưng
feature_importances_gini = clf_gini.feature_importances_
feature_importances_entropy = clf_entropy.feature_importances_

print("Feature importances using Gini Index:")
for feature, importance in zip(features, feature_importances_gini):
    print(f"{feature}: {importance:.4f}")

print("\nFeature importances using Information Gain:")
for feature, importance in zip(features, feature_importances_entropy):
    print(f"{feature}: {importance:.4f}")

# Trực quan hóa cây quyết định (Gini Index)
plt.figure(figsize=(12, 8))
tree.plot_tree(clf_gini, feature_names=features, class_names=['Chim', 'Thú', 'Bò sát', 'Cá', 'Côn trùng', 'Lưỡng cư'], filled=True)
plt.title("Decision Tree using Gini Index")
plt.show()

# Trực quan hóa cây quyết định (Entropy)
plt.figure(figsize=(12, 8))
tree.plot_tree(clf_entropy, feature_names=features, class_names=['Chim', 'Thú', 'Bò sát', 'Cá', 'Côn trùng', 'Lưỡng cư'], filled=True)
plt.title("Decision Tree using Information Gain")
plt.show()
