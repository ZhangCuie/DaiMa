import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存目录
output_dir = r'D:\桌面\数据集\模型'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
file_path = r'D:\桌面\数据集\earthquakes.csv'
data = pd.read_csv(file_path)

# 特征选择
magnitude_features_final_selection = ['depth', 'latitude', 'longitude', 'gap', 'rms', 'felt', 'sig', 'alert']
X_mag = data[magnitude_features_final_selection]
y_mag = data['magnitude']

# 检查并转换数据类型
X_mag = X_mag.apply(pd.to_numeric, errors='coerce', downcast='float')
y_mag = pd.to_numeric(y_mag, errors='coerce')


# 分割数据
X_trainM, X_testM, y_trainM, y_testM = train_test_split(X_mag, y_mag, test_size=0.2, random_state=42)

# 定义预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['depth', 'latitude', 'longitude', 'gap', 'rms', 'felt', 'sig']),
        ('cat', OneHotEncoder(), ['alert'])
    ])

# 初始化模型
models = {
    'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())]),
    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=42))]),
    'XGBoost': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', XGBRegressor(random_state=42))])
}

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储性能指标
results = {name: {'mae': [], 'rmse': [], 'mape': []} for name in models.keys()}

# 运行3次实验
for i in range(3):
    for name, model in models.items():
        mae_fold_scores = []
        rmse_fold_scores = []
        mape_fold_scores = []
        
        for train_index, test_index in kf.split(X_trainM):
            X_train, X_val = X_trainM.iloc[train_index], X_trainM.iloc[test_index]
            y_train, y_val = y_trainM.iloc[train_index], y_trainM.iloc[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mape = mean_absolute_percentage_error(y_val, y_pred)
            
            mae_fold_scores.append(mae)
            rmse_fold_scores.append(rmse)
            mape_fold_scores.append(mape)
        
        results[name]['mae'].append(np.mean(mae_fold_scores))
        results[name]['rmse'].append(np.mean(rmse_fold_scores))
        results[name]['mape'].append(np.mean(mape_fold_scores))

# 计算平均值和标准差
metrics = ['mae', 'rmse', 'mape']
summary = {name: {metric: (np.mean(scores), np.std(scores)) for metric, scores in result.items()} for name, result in results.items()}

# 打印性能指标
for name, result in summary.items():
    print(f"{name}:")
    for metric, (mean, std) in result.items():
        print(f"  {metric.upper()}: {mean:.4f}")

# 保存性能指标到 Excel
summary_df = pd.DataFrame({name: {metric: f'{mean:.4f}' for metric, (mean, std) in result.items()} for name, result in summary.items()})
summary_df.to_excel(os.path.join(output_dir, 'model_performance_summary.xlsx'))

# 定义绘图函数
def plot_prediction(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 6), dpi=300)
    plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'{model_name} 真实值与预测值对比')
    plt.savefig(os.path.join(output_dir, f'{model_name}_prediction_comparison.png'))
    plt.show()

# 绘制三个模型的真实值与预测值对比图
for name, model in models.items():
    model.fit(X_trainM, y_trainM)
    plot_prediction(model, X_testM, y_testM, name)

# 可视化性能指标
plt.figure(figsize=(12, 8), dpi=300)
for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i + 1)
    means = [summary[name][metric][0] for name in models.keys()]
    sns.barplot(x=list(models.keys()), y=means, palette='Set2')
    plt.title(f'{metric.upper()}')
    plt.xlabel('模型')
    plt.ylabel('值')
    for j in range(len(models)):
        plt.text(j, means[j], f'{means[j]:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'))
plt.show()