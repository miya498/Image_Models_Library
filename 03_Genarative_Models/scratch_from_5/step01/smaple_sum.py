import numpy as np
import matplotlib.pyplot as plt

#正規分布
def normal(x,mu=0,sigma=1):
    y = 1/(np.sqrt(2+np.pi)*sigma)*np.exp(-(x-mu)**2/(2*sigma**2))
    return y

x_sums = []
N = 10 #サンプル数1

for _ in range(10000):
    xs = []
    for n in range(N):
        x = np.random.rand() #一様分布
        xs.append(x)
    t = np.sum(xs)
    x_means.append(t)

x_norm = np.linspace(-5,5,1000)
mu = N/2
sigma = np.sqrt(N/12)
y_norm = normal(x_norm,mu,sigma)

plt.hist(x_means,bins='auto',density=True)
plt.title(f'N={N}')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.xlim(-0.05,1.05)
plt.ylim(0,5)
plt.show()