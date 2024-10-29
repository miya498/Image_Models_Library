import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 目的関数の定義 (Z = 3X^2 + 5Y^2 - 6XY)
def objective_function(x, y):
    return 3 * x ** 2 + 5 * y ** 2 - 6 * x * y

# 最適化手法のクラス
class OptimizerComparison:
    def __init__(self, optimizer, params, name):
        self.params = params
        self.optimizer = optimizer(self.params, lr=0.1)
        self.name = name
        self.history = []

    def step(self):
        self.optimizer.zero_grad()
        loss = objective_function(self.params[0], self.params[1])
        loss.backward()
        self.optimizer.step()
        self.history.append((self.params[0].item(), self.params[1].item(), loss.item()))

# 初期パラメータ
x = torch.tensor([10.0], requires_grad=True)
y = torch.tensor([10.0], requires_grad=True)

# 各最適化手法の設定
optimizers = [
    OptimizerComparison(optim.SGD, [x.clone().detach().requires_grad_(True), y.clone().detach().requires_grad_(True)], "GD"),
    OptimizerComparison(optim.SGD, [x.clone().detach().requires_grad_(True), y.clone().detach().requires_grad_(True)], "Momentum"),
    OptimizerComparison(optim.Adagrad, [x.clone().detach().requires_grad_(True), y.clone().detach().requires_grad_(True)], "AdaGrad"),
    OptimizerComparison(optim.Adam, [x.clone().detach().requires_grad_(True), y.clone().detach().requires_grad_(True)], "Adam")
]

# トレーニングループ
epochs = 50
for epoch in range(epochs):
    for optimizer in optimizers:
        optimizer.step()

# 各最適化手法の収束経路を個別に3Dグラフで描画
x_vals = np.linspace(-10, 10, 100)
y_vals = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = 3 * X ** 2 + 5 * Y ** 2 - 6 * X * Y

fig = plt.figure(figsize=(16, 12))

for i, optimizer in enumerate(optimizers, 1):
    ax = fig.add_subplot(2, 2, i, projection='3d')
    
    # 目的関数のサーフェスプロット
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.3, cmap='viridis')
    
    # 底面に等高線プロット
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z) - 50, cmap='viridis')
    
    # 各手法の収束経路のプロット
    x_history = [pos[0] for pos in optimizer.history]
    y_history = [pos[1] for pos in optimizer.history]
    z_history = [pos[2] for pos in optimizer.history]
    ax.plot(x_history, y_history, z_history, marker='o', color='red', label=optimizer.name)
    
    # ラベルとタイトル
    ax.set_xlabel("X value")
    ax.set_ylabel("Y value")
    ax.set_zlabel("Z value (Objective Function)")
    ax.set_title(f"Optimization Path for {optimizer.name}")
    ax.legend()

plt.tight_layout()
plt.show()
