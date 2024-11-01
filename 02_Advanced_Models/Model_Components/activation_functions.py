import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 活性化関数の定義
activation_functions = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(0.1),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'ELU': nn.ELU()
}

# 入力範囲を生成
x = torch.linspace(-5, 5, 100)

# 各活性化関数の出力を計算し、プロット
plt.figure(figsize=(12, 8))
for i, (name, func) in enumerate(activation_functions.items(), 1):
    y = func(x)
    plt.subplot(2, 3, i)
    plt.plot(x.numpy(), y.detach().numpy(), label=name)
    plt.title(name)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()