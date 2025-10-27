import numpy as np

# 设置随机种子以便结果可复现
np.random.seed(42)

# 数据（添加噪声）
areas = np.array([50, 80, 100, 120, 150])
prices = np.array([200, 290, 350, 410, 500]) + np.random.normal(0, 15, 5)  # 添加标准差为15的高斯噪声

# 构造 X（添加偏置项）
X = np.vstack([np.ones(len(areas)), areas])  # 2×5
y = prices.reshape(-1, 1)  # 5×1

print("=" * 50)
print("矩阵 X (特征矩阵，包含偏置项):")
print(X)
print(f"形状: {X.shape}")

print("\n" + "=" * 50)
print("矩阵 y (目标值):")
print(y)
print(f"形状: {y.shape}")

# 求解 w* = (XX^T)^(-1) Xy
w_star = np.linalg.inv(X @ X.T) @ X @ y

print("\n" + "=" * 50)
print("权重矩阵 w* (最优参数):")
print(w_star)
print(f"形状: {w_star.shape}")
print(f"\nw0 (偏置) = {w_star[0, 0]:.4f}")
print(f"w1 (斜率) = {w_star[1, 0]:.4f}")

# 计算预测值
y_pred = X.T @ w_star

print("\n" + "=" * 50)
print("预测值 vs 实际值:")
print(f"{'面积':<8} {'实际价格':<12} {'预测价格':<12} {'误差':<10}")
for i in range(len(areas)):
    error = y[i, 0] - y_pred[i, 0]
    print(f"{areas[i]:<8} {y[i, 0]:<12.2f} {y_pred[i, 0]:<12.2f} {error:<10.2f}")

# 经验风险就是训练集上的平均损失 (MSE)
empirical_risk = np.mean((y - y_pred) ** 2)

print(f"\n使用平方损失函数:")
print(f"L(y, ŷ) = (y - ŷ)²")

print(f"\n逐样本损失:")
for i in range(len(y)):
    loss_i = (y[i, 0] - y_pred[i, 0]) ** 2
    print(f"样本 {i+1}: L(y_{i+1}, ŷ_{i+1}) = ({y[i,0]:.2f} - {y_pred[i,0]:.2f})² = {loss_i:.4f}")

print(f"\n经验风险 (训练集上的平均损失):")
print(f"R_emp(w*) = (1/{len(y)}) × Σ L(y_i, ŷ_i)")
print(f"          = (1/{len(y)}) × {np.sum((y - y_pred) ** 2):.4f}")
print(f"          = {empirical_risk:.4f}")

print("\n" + "=" * 50)
print(f"最终模型: 房价 = {w_star[0, 0]:.4f} + {w_star[1, 0]:.4f} × 面积")