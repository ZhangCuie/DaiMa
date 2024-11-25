import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存可视化图像的目录
output_dir = r'D:\桌面\数据集\visualizations'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
file_path = r'D:\桌面\数据集\earthquakes.csv'
data = pd.read_csv(file_path)

# 数据描述性统计
desc_stats = data.describe()
print("数据描述性统计：")
print(desc_stats)

# 保存数据描述性统计为 Excel 文件
desc_stats.to_excel(r'D:\桌面\数据集\earthquakes_descriptive_statistics.xlsx')

# 可视化展示描述性统计
plt.figure(figsize=(12, 8))
sns.heatmap(desc_stats.T, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('描述性统计')
plt.savefig(os.path.join(output_dir, 'descriptive_statistics.png'))
plt.show()

# 绘制直方图和箱线图
focus_columns = ['latitude', 'longitude', 'magnitude']

# 绘制直方图在一个大图上
plt.figure(figsize=(18, 12))
for i, column in enumerate(focus_columns):
    plt.subplot(3, 1, i + 1)
    sns.histplot(data[column].dropna(), kde=True)
    plt.title(f'{column} 分布图')
    plt.xlabel(column)
    plt.ylabel('频数')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'histograms.png'))
plt.show()

# 绘制箱线图在一个大图上
plt.figure(figsize=(18, 12))
for i, column in enumerate(focus_columns):
    plt.subplot(3, 1, i + 1)
    sns.boxplot(x=data[column].dropna())
    plt.title(f'{column} 箱线图')
    plt.xlabel(column)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'boxplots.png'))
plt.show()

# 绘制总的相关性热力图
plt.figure(figsize=(12, 10))
corr_matrix = data.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('总的相关性热力图')
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.show()

# 定义相关性分析函数
def corr_with_target(df, target): 
    corr_matrix = df.corr(numeric_only=True)
    target_corr = corr_matrix[target].sort_values(ascending=False)
    plt.figure(figsize=(10, 10))
    sns.heatmap(target_corr.to_frame(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'{target} 与其他特征的相关性热力图')
    plt.savefig(os.path.join(output_dir, f'{target}_correlation_heatmap.png'))
    plt.show()
    return target_corr

# 分别绘制 latitude、longitude 和 magnitude 列与其他特征之间的相关性热力图
focus_columns = ['latitude', 'longitude', 'magnitude']
for column in focus_columns:
    if column in data.columns:
        corr_with_target(data, column)

# 在全球地图上展示地震发生分布
plt.figure(figsize=(15, 10))
m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary()

# 提取经纬度数据
if 'longitude' in data.columns and 'latitude' in data.columns:
    lons = data['longitude'].values
    lats = data['latitude'].values
    m.scatter(lons, lats, latlon=True, c='red', marker='o', alpha=0.5)
    plt.title('全球地震分布图')
    plt.savefig(os.path.join(output_dir, 'global_earthquake_distribution.png'))
    plt.show()
else:
    print("数据集中没有 'longitude' 或 'latitude' 列")