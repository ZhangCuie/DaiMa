import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
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
alert_features_final_selection = ['cdi', 'longitude', 'magnitude', 'mmi', 'sig', 'tsunami']
X_alr = data[alert_features_final_selection]
y_alr = data['alert']

# 检查并转换数据类型
X_alr = X_alr.apply(pd.to_numeric, errors='coerce', downcast='float')
y_alr = y_alr.astype(str)  # 将目标变量转换为字符串类型

# 删除包含缺失值的行
X_alr = X_alr.dropna()
y_alr = y_alr[X_alr.index]

# 使用 LabelEncoder 将目标变量转换为数值类型
label_encoder = LabelEncoder()
y_alr_encoded = label_encoder.fit_transform(y_alr)

# 分割数据
X_trainA, X_testA, y_trainA, y_testA = train_test_split(X_alr, y_alr_encoded, test_size=0.2, random_state=42)

# 定义预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), alert_features_final_selection)
    ])

# 初始化模型
models = {
    'Logistic Regression': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=5000))]),
    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))]),
    'XGBoost': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42))])
}

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储性能指标
results = {name: {'accuracy': [], 'mae': [], 'rmse': [], 'mape': [], 'report': [], 'confusion_matrix': []} for name in models.keys()}

# 运行3次实验
for i in range(3):
    for name, model in models.items():
        accuracy_fold_scores = []
        mae_fold_scores = []
        rmse_fold_scores = []
        mape_fold_scores = []
        
        for train_index, test_index in kf.split(X_trainA):
            X_train, X_val = X_trainA.iloc[train_index], X_trainA.iloc[test_index]
            y_train, y_val = y_trainA[train_index], y_trainA[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            accuracy = accuracy_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mape = mean_absolute_percentage_error(y_val, y_pred)
            
            accuracy_fold_scores.append(accuracy)
            mae_fold_scores.append(mae)
            rmse_fold_scores.append(rmse)
            mape_fold_scores.append(mape)
        
        results[name]['accuracy'].append(np.mean(accuracy_fold_scores))
        results[name]['mae'].append(np.mean(mae_fold_scores))
        results[name]['rmse'].append(np.mean(rmse_fold_scores))
        results[name]['mape'].append(np.mean(mape_fold_scores))
        results[name]['report'].append(classification_report(y_testA, model.predict(X_testA), output_dict=True))
        results[name]['confusion_matrix'].append(confusion_matrix(y_testA, model.predict(X_testA)))

# 计算平均值和标准差
metrics = ['accuracy', 'mae', 'rmse', 'mape']
summary = {name: {metric: (np.mean(scores), np.std(scores)) for metric, scores in result.items() if metric in metrics} for name, result in results.items()}

# 打印性能指标
for name, result in summary.items():
    print(f"{name}:")
    for metric, (mean, std) in result.items():
        print(f"  {metric.upper()}: {mean:.4f}")

# 保存性能指标到 Excel
summary_df = pd.DataFrame({name: {metric: f'{mean:.4f}' for metric, (mean, std) in result.items()} for name, result in summary.items()})
summary_df.to_excel(os.path.join(output_dir, 'model_performance_summary.xlsx'))

# 定义类别标签
class_labels = label_encoder.classes_

# 定义评估指标打印函数
def print_metrics(y_true, y_pred, model_name, target_type, class_labels):
    print(f"模型: {model_name}")
    print(f"目标类型: {target_type}")
    print(f"分类报告:\n{classification_report(y_true, y_pred, target_names=class_labels, zero_division=0)}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"混淆矩阵:\n{cm}")
    return cm

# 计算并打印评估指标
y_prLR_A = models['Logistic Regression'].predict(X_testA)
y_prRF_A = models['Random Forest'].predict(X_testA)
y_prXG_A = models['XGBoost'].predict(X_testA)

confu_mtrx_LR = print_metrics(y_testA, y_prLR_A, 'Logistic Regression', 'classification', class_labels)
confu_mtrx_RF = print_metrics(y_testA, y_prRF_A, 'Random Forest', 'classification', class_labels)
confu_mtrx_XG = print_metrics(y_testA, y_prXG_A, 'XGBoost', 'classification', class_labels)

# 绘制混淆矩阵
model_in = [confu_mtrx_LR, confu_mtrx_RF, confu_mtrx_XG]
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']

for cm, name in zip(model_in, model_names):
    if cm is not None:
        plt.figure(figsize=(10, 6), dpi=300)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('预测值')
        plt.ylabel('真实值')
        plt.title(f'{name} 混淆矩阵')
        plt.savefig(os.path.join(output_dir, f'{name}_confusion_matrix.png'))
        plt.show()

# 可视化性能指标
plt.figure(figsize=(18, 12), dpi=300)
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i + 1)
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