# 波士顿房价预测项目

## 项目简介

本项目是 2024 年《模式识别与机器学习》课程的结课作业，选题为波士顿房价预测。通过对经典的波士顿房价数据集进行预处理与建模，探索传统回归模型（线性回归、岭回归）与现代机器学习模型（随机森林回归）在房价预测任务中的表现。通过对比不同算法的性能，分析各模型的优缺点与适用场景。

## 项目文件结构
├── Linear_Regression.py # 基于梯度下降、自定义BFGS等优化方法的线性回归实现 

├── RFR.py # 随机森林回归模型代码 

├── Ridge.py # 岭回归模型代码 

├── boston.mat # 波士顿房价数据集（MATLAB 格式） 

├── boston.xlsx # 波士顿房价数据集（Excel 格式） 

├── data_processing.ipynb # 数据处理和可视化的 Jupyter Notebook 

├── requirements.txt # 项目依赖的 Python 库列表 

└── README.md # 项目说明文档


## 文件说明

- **`Linear_Regression.py`**  
  实现了多种优化方法的线性回归，包括：
  - 梯度下降（Gradient Descent, GD）
  - 随机梯度下降（Stochastic Gradient Descent, SGD）
  - 拟牛顿法（BFGS）  
  支持训练集与测试集的损失（MSE）与 R² 曲线绘制，并输出模型性能对比。

- **`RFR.py`**  
  实现了随机森林回归模型，基于 `sklearn.ensemble.RandomForestRegressor`，并对模型进行了超参数调优和性能评估。

- **`Ridge.py`**  
  实现了岭回归模型，基于 `sklearn.linear_model.Ridge`，并对正则化参数进行了调优。
  比较了岭回归与普通线性回归在处理多重共线性问题上的差异。

- **`boston.mat` / `boston.xlsx`**  
  波士顿房价数据集的不同格式版本，包含 506 个样本，13 个特征列，用于预测房价中位数（`MEDV`）。

- **`data_processing.ipynb`**  
  数据预处理代码，包括：
  - 数据分布可视化
  - 对偏态特征的对数变换
  - 特征标准化处理  
  为各模型训练做好数据准备工作。

- **`requirements.txt`**  
  列出了项目所需的 Python 库及其版本，便于快速搭建运行环境。

## 环境配置

请确保已安装 Python 3.8 或以上版本，按照以下步骤配置运行环境：

1. 克隆项目到本地：
   ```bash
   git clone https://github.com/your-repo/boston-house-prediction.git
   cd boston-house-prediction```
2. 创建虚拟环境并激活（可选）：
```python -m venv boston_env
source boston_env/bin/activate  # Linux/macOS
boston_env\Scripts\activate     # Windows```
3. 安装依赖：
``pip install -r requirements.txt``
4. 确保必要的文件（如 boston.xlsx）已放置于项目根目录下。
## 使用说明

### 运行各模型代码

####数据预处理
运行 `data_processing.ipynb`，可视化并预处理数据，为各模型训练做好准备。

#### 线性回归
运行 `Linear_Regression.py`：
```bash
python Linear_Regression.py```
输出梯度下降（GD）、随机梯度下降（SGD）、拟牛顿法（BFGS）等优化方法的性能对比，并展示损失曲线与 R² 曲线。

#### 随机森林回归
运行 `RFR.py`：
```bash
python RFR.py```
输出随机森林模型的训练集和测试集性能，并展示模型对波士顿房价的预测结果。

#### 岭回归
运行 `Ridge.py`：
```bash
python Ridge.py```
输出岭回归模型的性能，并与普通线性回归进行对比。
