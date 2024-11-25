import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 读取数据
file_path = r'D:\桌面\数据集\earthquakes.csv'
data = pd.read_csv(file_path)

# 打印数据集的基本信息
print("数据集的基本信息：")
print(data.info())

# 检查缺失值
print("缺失值统计：")
print(data.isnull().sum())

# 填充缺失值
# 对数值型数据使用均值填充
num_cols = data.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
data[num_cols] = imputer.fit_transform(data[num_cols])

# 对分类数据使用众数填充
cat_cols = data.select_dtypes(include=[object]).columns
imputer = SimpleImputer(strategy='most_frequent')
data[cat_cols] = imputer.fit_transform(data[cat_cols])

# 检查缺失值填充后的结果
print("缺失值填充后的统计：")
print(data.isnull().sum())

# 编码分类变量
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 标准化数值型数据
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# 打印预处理后的数据集信息
print("预处理后的数据集信息：")
print(data.info())

# 保存预处理后的数据集
output_file_path = r'D:\桌面\数据集\preprocessed_earthquakes.csv'
data.to_csv(output_file_path, index=False)

print(f"预处理后的数据集已保存到 {output_file_path}")