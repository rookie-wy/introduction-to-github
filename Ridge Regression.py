import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 设置随机数种子以保证结果可复现
np.random.seed(42)

# 生成 1000 条样本，每个样本有 5 个特征
n_samples = 1000
n_features = 5
X = np.random.randn(n_samples, n_features)

# 真实的系数
true_coef = np.array([2, -1, 3, -0.5, 1])

# 生成目标值 y，加入一些噪声
noise = np.random.normal(0, 0.5, n_samples)
y = np.dot(X, true_coef) + noise

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建岭回归模型
ridge = Ridge(alpha=1.0)

# 训练模型
ridge.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = ridge.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差: {mse}")

# 输出模型系数
print("模型系数:", ridge.coef_)

# 可视化预测值和真实值的分布
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('岭回归模型：真实值 vs 预测值')
plt.show()

# 可视化模型系数的分布
plt.figure(figsize=(10, 6))
plt.bar(range(len(ridge.coef_)), ridge.coef_)
plt.xlabel('特征索引')
plt.ylabel('系数值')
plt.title('岭回归模型系数分布')
plt.show()
    
