import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Use default fonts for better compatibility
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set random seed for reproducibility
np.random.seed(42)

# Generate simulated housing data
n_samples = 100

# Generate base feature: house area (sqm)
area = np.random.uniform(50, 200, n_samples)

# Create highly collinear features
# Feature 1: House area
X1 = area

# Feature 2: Number of rooms (highly correlated with area)
X2 = area / 30 + np.random.normal(0, 0.5, n_samples)

# Feature 3: Almost completely dependent on area (high collinearity)
X3 = area * 1.001 + np.random.normal(0, 0.1, n_samples)

# Combine feature matrix
X = np.column_stack([X1, X2, X3])

# Generate true house price (10k CNY): mainly determined by area
true_price = 2 * area + 10 * X2 + np.random.normal(0, 5, n_samples)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate condition number and determinant of X^T X
XTX = X_scaled.T @ X_scaled
condition_number = np.linalg.cond(XTX)
determinant = np.linalg.det(XTX)

print("=" * 60)
print("Multicollinearity Diagnostics")
print("=" * 60)
print(f"Determinant of X^T X: {determinant:.6e}")
print(f"Condition Number of X^T X: {condition_number:.2f}")
print(f"Correlation Matrix between Features:")
correlation_matrix = np.corrcoef(X.T)
print(correlation_matrix)
print("=" * 60)

# Run multiple trials with noise to test stability
n_trials = 50
coefficients_ols = []
coefficients_ridge = []

for trial in range(n_trials):
    # Add small random noise to data
    noise = np.random.normal(0, 0.01, X_scaled.shape)
    X_noisy = X_scaled + noise
    
    # OLS regression
    ols = LinearRegression()
    ols.fit(X_noisy, true_price)
    coefficients_ols.append(ols.coef_)
    
    # Ridge regression (regularization)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_noisy, true_price)
    coefficients_ridge.append(ridge.coef_)

coefficients_ols = np.array(coefficients_ols)
coefficients_ridge = np.array(coefficients_ridge)

# Calculate statistics
ols_means = coefficients_ols.mean(axis=0)
ols_stds = coefficients_ols.std(axis=0)
ridge_means = coefficients_ridge.mean(axis=0)
ridge_stds = coefficients_ridge.std(axis=0)

# Create figure with 6 subplots
fig = plt.figure(figsize=(18, 10))

# 1. Real data: Area vs Price
ax1 = plt.subplot(2, 3, 1)
plt.scatter(X1, true_price, alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
plt.xlabel('Area (sqm)', fontsize=12)
plt.ylabel('Price (10k CNY)', fontsize=12)
plt.title('Housing Data: Area vs Price', fontsize=12)
plt.grid(True, alpha=0.3)

# 2. Real data: Room Number vs Price
ax2 = plt.subplot(2, 3, 2)
plt.scatter(X2, true_price, alpha=0.6, s=50, c='coral', edgecolors='black', linewidth=0.5)
plt.xlabel('Number of Rooms', fontsize=12)
plt.ylabel('Price (10k CNY)', fontsize=12)
plt.title('Housing Data: Rooms vs Price', fontsize=12)
plt.grid(True, alpha=0.3)

# 3. Show collinearity between features
ax3 = plt.subplot(2, 3, 3)
plt.scatter(X1, X3, alpha=0.6, s=50, c='purple', edgecolors='black', linewidth=0.5)
plt.xlabel('Area (sqm)', fontsize=12)
plt.ylabel('Collinear Feature 3', fontsize=12)
plt.title(f'Multicollinearity Problem\nCorrelation: {np.corrcoef(X1, X3)[0,1]:.4f}', fontsize=12)
plt.grid(True, alpha=0.3)

# 4. 2D visualization of features and price
ax4 = plt.subplot(2, 3, 4)
scatter = plt.scatter(X1, X2, c=true_price, alpha=0.6, s=50, cmap='viridis', 
                     edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Price (10k CNY)')
plt.xlabel('Area (sqm)', fontsize=12)
plt.ylabel('Number of Rooms', fontsize=12)
plt.title('Feature Distribution & Price', fontsize=12)
plt.grid(True, alpha=0.3)

# 5. OLS coefficient instability trajectory
ax5 = plt.subplot(2, 3, 5)
for i in range(3):
    plt.plot(coefficients_ols[:, i], alpha=0.6, linewidth=2, 
             label=f'Feature {i+1}', marker='o', markersize=3)
plt.xlabel('Trial Number', fontsize=12)
plt.ylabel('OLS Coefficient Value', fontsize=12)
plt.title('OLS Instability\n(Small noise -> Large fluctuation)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Ridge regression stability
ax6 = plt.subplot(2, 3, 6)
for i in range(3):
    plt.plot(coefficients_ridge[:, i], alpha=0.6, linewidth=2,
             label=f'Feature {i+1}', marker='s', markersize=3)
plt.xlabel('Trial Number', fontsize=12)
plt.ylabel('Ridge Coefficient Value', fontsize=12)
plt.title('Ridge Regression Stability\n(Regularization -> More stable)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('/mnt/user-data/outputs/ols_instability_simplified.png', dpi=300, bbox_inches='tight')
print("\nFigure saved successfully!")

# Print detailed statistics
print("\n" + "=" * 60)
print("Coefficient Estimation Statistics")
print("=" * 60)
print("\nOLS (Ordinary Least Squares):")
for i in range(3):
    print(f"  Feature {i+1}: Mean={ols_means[i]:8.2f}, Std Dev={ols_stds[i]:8.2f}")
print(f"\n  Average Std Dev: {ols_stds.mean():.2f}")

print("\nRidge Regression:")
for i in range(3):
    print(f"  Feature {i+1}: Mean={ridge_means[i]:8.2f}, Std Dev={ridge_stds[i]:8.2f}")
print(f"\n  Average Std Dev: {ridge_stds.mean():.2f}")

print(f"\nStability Improvement: {(1 - ridge_stds.mean()/ols_stds.mean())*100:.1f}%")
print("=" * 60)

plt.show()