import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 加载波士顿房价数据集
boston = pd.read_excel("boston.xlsx")

# 导出特征值和目标值
X = boston.drop(columns=["MEDV"])   # 特征值
y = boston["MEDV"]      # 目标值

# 对偏态特征进行对数变换
X_skewed = ['CRIM', 'AGE', 'DIS', 'TAX', 'LSTAT']
X[X_skewed] = X[X_skewed].apply(lambda x: np.log1p(x))

# 标准化所有特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用 Ridge 回归模型进行训练
class RidgeRegressionCustom:
    def __init__(self, learning_rate=0.01, n_iterations=500, method="gd", alpha=1.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.alpha = alpha  # 正则化参数
        self.coefficients_ = None
        self.losses_train = []
        self.losses_test = []
        self.r2_train = []
        self.r2_test = []

    def fit(self, X, y, X_test=None, y_test=None):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 加入偏置项
        self.coefficients_ = np.zeros(X_b.shape[1])


        if self.method == "gd":
            for iteration in range(self.n_iterations):
                y_pred = X_b.dot(self.coefficients_)
                error = y_pred - y
                # 计算损失，包括正则化项（不对偏置项进行正则化）
                loss = np.mean(error ** 2) + self.alpha * np.sum(self.coefficients_[1:] ** 2)
                self.losses_train.append(loss)
                self.r2_train.append(r2_score(y, y_pred))

                if X_test is not None and y_test is not None:
                    y_pred_test = np.c_[np.ones((X_test.shape[0], 1)), X_test].dot(self.coefficients_)
                    error_test = y_pred_test - y_test
                    loss_test = np.mean(error_test ** 2) + self.alpha * np.sum(self.coefficients_[1:] ** 2)
                    self.losses_test.append(loss_test)
                    self.r2_test.append(r2_score(y_test, y_pred_test))

                # 计算梯度，加入正则化项
                # 这里的正则化项不对偏置项进行更新
                gradients = 2 / X_b.shape[0] * X_b.T.dot(error)  # 基础梯度
                gradients[1:] += 2 * self.alpha * self.coefficients_[1:]  # 正则化部分（不对偏置项进行正则化）
                self.coefficients_ -= self.learning_rate * gradients  # 更新参数

        elif self.method == "sgd":
            for iteration in range(self.n_iterations):
                for i in range(X.shape[0]):
                    xi = X_b[i:i + 1]  # 取一个样本
                    yi = y[i:i + 1]  # 取该样本的目标值
                    y_pred_i = xi.dot(self.coefficients_)  # 预测值
                    error_i = y_pred_i - yi  # 误差
                    gradients = 2 * xi.T.dot(error_i)  # 基础梯度
                    gradients[1:] += 2 * self.alpha * self.coefficients_[1:]  # 正则化部分（不对偏置项进行正则化）
                    self.coefficients_ -= self.learning_rate * gradients  # 更新参数

                # 计算训练集的预测和损失
                y_pred = X_b.dot(self.coefficients_)
                error = y_pred - y
                loss = np.mean(error ** 2) + self.alpha * np.sum(self.coefficients_[1:] ** 2)  # 加入正则化项
                self.losses_train.append(loss)
                self.r2_train.append(r2_score(y, y_pred))  # 添加训练集R²

                # 如果有测试集，计算测试集的预测和损失
                if X_test is not None and y_test is not None:
                    y_pred_test = np.c_[np.ones((X_test.shape[0], 1)), X_test].dot(self.coefficients_)
                    error_test = y_pred_test - y_test
                    loss_test = np.mean(error_test ** 2) + self.alpha * np.sum(self.coefficients_[1:] ** 2)  # 加入正则化项
                    self.losses_test.append(loss_test)
                    self.r2_test.append(r2_score(y_test, y_pred_test))  # 添加测试集R²

        elif self.method == "bfgs":
                X_b_test = None
                if X_test is not None:
                    X_b_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

                # 初始化海森矩阵为单位矩阵
                H_k = np.eye(X_b.shape[1])
                for iteration in range(self.n_iterations):

                    # 预测值和误差
                    y_pred = X_b.dot(self.coefficients_)
                    gradient = 2 / X_b.shape[0] * X_b.T.dot(y_pred - y)  # 基础梯度
                    gradient[1:] += 2 * self.alpha * self.coefficients_[1:]  # 正则化部分（不对偏置项进行正则化）
                    loss = np.mean((y_pred - y) ** 2) + self.alpha * np.sum(self.coefficients_[1:] ** 2)  # 加入正则化项
                    self.losses_train.append(loss)

                    # 如果有测试集，记录测试集损失
                    if X_test is not None and y_test is not None:
                        y_pred_test = X_b_test.dot(self.coefficients_)
                        loss_test = np.mean((y_pred_test - y_test) ** 2) + self.alpha * np.sum(self.coefficients_[1:] ** 2)  # 加入正则化项
                        self.losses_test.append(loss_test)

                    # 计算迭代方向
                    delta_theta = -H_k.dot(gradient)

                    # 更新参数
                    self.coefficients_ += self.learning_rate * delta_theta

                    # 计算新的梯度
                    y_pred_new = X_b.dot(self.coefficients_)
                    gradient_new = 2 / X_b.shape[0] * X_b.T.dot(y_pred_new - y)  # 基础梯度
                    gradient_new[1:] += 2 * self.alpha * self.coefficients_[1:]  # 正则化部分（不对偏置项进行正则化）

                    # 差分向量，确保是列向量
                    s_k = self.learning_rate * delta_theta.reshape(-1, 1)  # 将 delta_theta 转为列向量
                    y_k = (gradient_new - gradient).reshape(-1, 1)  # 同样转换为列向量

                    # 更新海森矩阵 (BFGS 公式)
                    yk_T_sk = y_k.T.dot(s_k)  # 计算 y_k^T * s_k，确保是标量
                    if yk_T_sk > 1e-10:  # 避免数值不稳定
                        H_k = (
                                H_k
                                + (1 + y_k.T.dot(H_k).dot(y_k) / yk_T_sk) * (s_k.dot(s_k.T)) / yk_T_sk
                                - (H_k.dot(y_k).dot(s_k.T) + s_k.dot(y_k.T).dot(H_k)) / yk_T_sk
                        )
                    # 检查收敛条件
                    if np.linalg.norm(gradient_new) < 1e-6:
                        break
                # 计算 R²
                self.r2_train.append(r2_score(y, y_pred))
                if X_test is not None and y_test is not None:
                            self.r2_test.append(r2_score(y_test, y_pred_test))

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.coefficients_)


    def r2_score(self, y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)


# 初始化岭回归模型，分别使用不同的训练方法
ridge_model_gd = RidgeRegressionCustom(alpha=1.0, method="gd")
ridge_model_sgd = RidgeRegressionCustom(alpha=1.0, method="sgd")
ridge_model_bfgs = RidgeRegressionCustom(alpha=1.0, method="bfgs")

# 训练并预测
ridge_model_gd.fit(X_train, y_train, X_test, y_test)
ridge_model_sgd.fit(X_train, y_train, X_test, y_test)
ridge_model_bfgs.fit(X_train, y_train, X_test, y_test)

# 计算结果并对比
def print_results(model_name, model):
    print(
        f"\n{model_name} - 训练集 MSE: {mean_squared_error(y_train, model.predict(X_train)):.2f}, 测试集 MSE: {mean_squared_error(y_test, model.predict(X_test)):.2f}")
    print(
        f"{model_name} - 训练集 RMSE: {np.sqrt(mean_squared_error(y_train, model.predict(X_train))):.2f}, 测试集 RMSE: {np.sqrt(mean_squared_error(y_test, model.predict(X_test))):.2f}")
    print(
        f"{model_name} - 训练集 R²: {r2_score(y_train, model.predict(X_train)):.2f}, 测试集 R²: {r2_score(y_test, model.predict(X_test)):.2f}")

# 输出对比结果
print_results("Ridge Regression (GD)", ridge_model_gd)
print_results("Ridge Regression (SGD)", ridge_model_sgd)
print_results("Ridge Regression (BFGS)", ridge_model_bfgs)

# 设置支持中文的字体（假设你的系统安装了 SimHei 字体）
font_path = "C:/Windows/Fonts/simhei.ttf"  # Windows 系统字体路径
font_prop = font_manager.FontProperties(fname=font_path)
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（SimHei）
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 绘制损失曲线和 R² 曲线对比
plt.figure(figsize=(14, 6))

# 子图 1：损失变化
plt.subplot(1, 2, 1)
plt.plot(ridge_model_gd.losses_train, label="Ridge GD 训练集损失")
plt.plot(ridge_model_sgd.losses_train, label="Ridge SGD 训练集损失")
plt.plot(ridge_model_bfgs.losses_train, label="Ridge BFGS 训练集损失")
print(ridge_model_gd.losses_train)
print(ridge_model_sgd.losses_train)
print(ridge_model_bfgs.losses_train)
plt.xlabel("迭代次数")
plt.ylabel("损失 (MSE)")
plt.title("训练过程中的损失变化")
plt.legend()

# 子图 2：R² 值变化
plt.subplot(1, 2, 2)
plt.plot(ridge_model_gd.r2_train, label="Ridge GD 训练集 R^2 值", linestyle="-")
plt.plot(ridge_model_sgd.r2_train, label="Ridge SGD 训练集 R^2 值", linestyle="-")
plt.plot(ridge_model_bfgs.r2_train, label="Ridge BFGS 训练集 R^2 值", linestyle="-")

# 绘制测试集 R² 曲线
plt.plot(ridge_model_gd.r2_test, label="GD 测试集 R^2", linestyle="--")
plt.plot(ridge_model_sgd.r2_test, label="SGD 测试集 R^2", linestyle="--")
plt.plot(ridge_model_bfgs.r2_test, label="BFGS 测试集 R^2", linestyle="--")

# 设置图例、标题等
plt.xlabel("迭代次数")
plt.ylabel("R^2 值")
plt.title("不同训练算法的 R^2 变化曲线")
plt.legend()

plt.tight_layout()
plt.show()