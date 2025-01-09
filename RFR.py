#%% 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import rcParams
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置支持中文的字体（假设你的系统安装了 SimHei 字体）
font_path = "C:/Windows/Fonts/simhei.ttf"  # Windows 系统字体路径
font_prop = font_manager.FontProperties(fname=font_path)
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（SimHei）
rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 加载波士顿房价数据集
boston = pd.read_excel("boston.xlsx")

# 特征值与目标值
X = boston.drop(columns=["MEDV"])
y = boston["MEDV"]

# 对偏态特征应用对数变换
X_skewed = ['CRIM', 'AGE', 'DIS', 'TAX', 'LSTAT']
X[X_skewed] = X[X_skewed].apply(lambda x: np.log1p(x))

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 设置随机森林参数
max_trees = 500  # 最大树的数量
rf = RandomForestRegressor(
    n_estimators=max_trees,  # 森林中决策树的数量，默认为 100
    criterion='squared_error',  # 分裂时的评价准则，默认为平方误差
    max_depth=10,  # 决策树的最大深度，None 表示不限深度
    min_samples_split=5,  # 分裂一个节点所需的最小样本数，默认为 2
    random_state=122,  # 随机种子，确保结果可复现
    warm_start=True,  # 是否在已有模型基础上增量训练，默认为 False
)

# 记录性能指标
train_mse_list = []
test_mse_list = []
train_r2_list = []
test_r2_list = []
tree_numbers = []

# 逐步增加决策树数量
for n_trees in range(1, max_trees + 1):
    print(n_trees)
    rf.set_params(n_estimators=n_trees)
    rf.fit(X_train, y_train)

    # 预测结果
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    # 记录训练集和测试集的 MSE 和 R²
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)
    tree_numbers.append(n_trees)

#%% 绘制图像
plt.figure(figsize=(14, 6))

# 子图1：MSE变化图
plt.subplot(1, 2, 1)
plt.plot(tree_numbers, train_mse_list, label="训练集 MSE", linestyle="-")
plt.plot(tree_numbers, test_mse_list, label="测试集 MSE", linestyle="--")
plt.xlabel("决策树数量")
plt.ylabel("损失 (MSE)")
plt.title("不同决策树数量的 MSE 变化曲线")
plt.legend()

# 子图2：R²变化图
plt.subplot(1, 2, 2)
plt.plot(tree_numbers, train_r2_list, label="训练集 R^2", linestyle="-")
plt.plot(tree_numbers, test_r2_list, label="测试集 R^2", linestyle="--")
plt.xlabel("决策树数量")
plt.ylabel("R^2 值")
plt.title("不同决策树数量的 R^2 变化曲线")
plt.legend()

# 设置中文字体
plt.tight_layout()
plt.savefig("随机森林性能变化.png",dpi=600)
plt.close()
#%%
# 输出目标方程
feature_importance = "目标方程："
for i, feature in enumerate(X.columns):
    feature_importance = feature_importance + f"{rf.feature_importances_[i]} * {feature} +"
print(feature_importance)

# 绘制特征重要性条形图
feature_importance = rf.feature_importances_
feature_names = X.columns.tolist()
sorted_idx = feature_importance.argsort()
#避免中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.barh(range(len(feature_importance)), feature_importance[sorted_idx])
plt.yticks(range(len(feature_importance)), [feature_names[i] for i in sorted_idx],fontsize=5)
plt.xlabel('特征重要性')
plt.ylabel('特征名称')
plt.title('随机森林回归特征重要性')
plt.savefig('随机森林回归特征重要性',dpi=600)
plt.close()

#%%
print(f"训练集最小MSE{min(train_mse_list)}，最小RMSE{min(train_mse_list)**0.5}，最大R2{max(train_r2_list)}\n测试集集最小MSE{min(test_mse_list)}，最小RMSE{min(test_mse_list)**0.5}，最大R2{max(test_r2_list)}\n")