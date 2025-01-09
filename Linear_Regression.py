import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import rcParams

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

# 自定义线性回归模型
class LinearRegressionCustom:
    def __init__(self, learning_rate=0.01, n_iterations=500, method="gd"):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.coefficients_ = None
        self.losses_train = []
        self.losses_test = []
        self.r2_train = []
        self.r2_test = []

    def fit(self, X, y, X_test=None, y_test=None):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.coefficients_ = np.zeros(X_b.shape[1])

        for iteration in range(self.n_iterations):
            if self.method == "gd":
                y_pred = X_b.dot(self.coefficients_)
                error = y_pred - y
                self.losses_train.append(np.mean(error ** 2))
                self.r2_train.append(r2_score(y, y_pred))
                if X_test is not None and y_test is not None:
                    y_pred_test = np.c_[np.ones((X_test.shape[0], 1)), X_test].dot(self.coefficients_)
                    error_test = y_pred_test - y_test
                    self.losses_test.append(np.mean(error_test ** 2))
                    self.r2_test.append(r2_score(y_test, y_pred_test))

                gradients = 2 / X_b.shape[0] * X_b.T.dot(error)
                self.coefficients_ -= self.learning_rate * gradients

            elif self.method == "sgd":
                for i in range(X.shape[0]):
                    xi = X_b[i:i + 1]  # 取一个样本
                    yi = y[i:i + 1]  # 取该样本的目标值
                    y_pred_i = xi.dot(self.coefficients_)  # 预测值
                    error_i = y_pred_i - yi  # 误差
                    gradients = 2 * xi.T.dot(error_i)  # 计算梯度
                    self.coefficients_ -= self.learning_rate * gradients  # 更新参数

                # 计算训练集的预测和损失
                y_pred = X_b.dot(self.coefficients_)
                error = y_pred - y
                self.losses_train.append(np.mean(error ** 2))  # 添加训练集损失
                self.r2_train.append(r2_score(y, y_pred))  # 添加训练集R²

                # 如果有测试集，计算测试集的预测和损失
                if X_test is not None and y_test is not None:
                    y_pred_test = np.c_[np.ones((X_test.shape[0], 1)), X_test].dot(self.coefficients_)
                    error_test = y_pred_test - y_test
                    self.losses_test.append(np.mean(error_test ** 2))  # 添加测试集损失
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
                    gradient = 2 / X_b.shape[0] * X_b.T.dot(y_pred - y)  # 梯度
                    loss = np.mean((y_pred - y) ** 2)  # 损失函数值
                    self.losses_train.append(loss)

                    # 如果有测试集，记录测试集损失
                    if X_test is not None and y_test is not None:
                        y_pred_test = X_b_test.dot(self.coefficients_)
                        loss_test = np.mean((y_pred_test - y_test) ** 2)
                        self.losses_test.append(loss_test)

                    # 计算迭代方向
                    delta_theta = -H_k.dot(gradient)

                    # 更新参数
                    self.coefficients_ += self.learning_rate * delta_theta

                    # 计算新的梯度
                    y_pred_new = X_b.dot(self.coefficients_)
                    gradient_new = 2 / X_b.shape[0] * X_b.T.dot(y_pred_new - y)

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

                    # 计算 R²
                    self.r2_train.append(r2_score(y, y_pred))
                    if X_test is not None and y_test is not None:
                        self.r2_test.append(r2_score(y_test, y_pred_test))
                    # 检查收敛条件
                    if np.linalg.norm(gradient_new) < 1e-6:
                        break

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.coefficients_)


# 初始化模型
gd_model = LinearRegressionCustom(method="gd")
sgd_model = LinearRegressionCustom(method="sgd")
bfgs_model = LinearRegressionCustom(learning_rate=0.01, n_iterations=20, method="bfgs")

# 模型训练
gd_model.fit(X_train, y_train, X_test, y_test)
sgd_model.fit(X_train, y_train, X_test, y_test)
bfgs_model.fit(X_train, y_train, X_test, y_test)

# 计算结果并对比
def print_results(model_name, model):
    print(
        f"\n{model_name} - 训练集 MSE: {mean_squared_error(y_train, model.predict(X_train)):.2f}, 测试集 MSE: {mean_squared_error(y_test, model.predict(X_test)):.2f}")
    print(
        f"{model_name} - 训练集 RMSE: {np.sqrt(mean_squared_error(y_train, model.predict(X_train))):.2f}, 测试集 RMSE: {np.sqrt(mean_squared_error(y_test, model.predict(X_test))):.2f}")
    print(
        f"{model_name} - 训练集 R^2: {r2_score(y_train, model.predict(X_train)):.2f}, 测试集 R²: {r2_score(y_test, model.predict(X_test)):.2f}")


# 输出对比结果
print_results("Gradient Descent", gd_model)
print_results("Stochastic Gradient Descent", sgd_model)
print_results("BFGS (拟牛顿法)", bfgs_model)

# 设置支持中文的字体（假设你的系统安装了 SimHei 字体）
font_path = "C:/Windows/Fonts/simhei.ttf"  # Windows 系统字体路径
font_prop = font_manager.FontProperties(fname=font_path)
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（SimHei）
rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 绘制损失曲线与R²曲线
plt.figure(figsize=(14, 6))

# 子图1：损失曲线
plt.subplot(1, 2, 1)
plt.plot(gd_model.losses_train, label="GD 训练集损失")
plt.plot(sgd_model.losses_train, label="SGD 训练集损失")
plt.plot(bfgs_model.losses_train, label="BFGS 训练集损失")
print(gd_model.losses_train)
print(sgd_model.losses_train)
print(bfgs_model.losses_train)
plt.xlabel("迭代次数")
plt.ylabel("损失 (MSE)")
plt.title("不同训练算法的损失变化曲线")
plt.legend()

# 子图2：R²曲线，增加测试集的 R² 曲线
plt.subplot(1, 2, 2)
# 绘制训练集 R² 曲线
plt.plot(gd_model.r2_train, label="GD 训练集 R^2", linestyle="-")
plt.plot(sgd_model.r2_train, label="SGD 训练集 R^2", linestyle="-")
plt.plot(bfgs_model.r2_train, label="BFGS 训练集 R^2", linestyle="-")

# 绘制测试集 R² 曲线
plt.plot(gd_model.r2_test, label="GD 测试集 R^2", linestyle="--")
plt.plot(sgd_model.r2_test, label="SGD 测试集 R^2", linestyle="--")
plt.plot(bfgs_model.r2_test, label="BFGS 测试集 R^2", linestyle="--")

# 设置图例、标题等
plt.xlabel("迭代次数")
plt.ylabel("R^2 值")
plt.title("不同训练算法的 R^2 变化曲线")
plt.legend()

plt.tight_layout()
plt.show()

