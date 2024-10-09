import os 
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__),'old_faithful.txt')
xs = np.loadtxt(path)
print(xs.shape)

#パラメータ初期値
phis = np.array([0.5,0.5])
mus = np.array([[0.0,50.0],[0.0,100.0]])
covs = np.array([np.eye(2),np.eye(2)]) #np.eyeで単位行列生成

K = len(phis)
N = len(xs)
MAX_ITERS = 100
THESHOLD = 1e-4  #EMアルゴリズムの更新を止める閾値

def multivariate_normal(x,mu,cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1/np.sqrt((2*np.pi)**D*det)
    y = z*np.exp((x-mu).T@inv@(x-mu)/-2.0)
    return y

def gmm(x,phis,mus,covs):
    K = len(phis)
    y = 0
    for k in range(K):
        phi,mu,cov = phis[k],mus[k],covs[k]
        y += phi*multivariate_normal(x,mu,cov)
    return y

#対数尤度関数
def likelihold(xs,phis,mus,covs):
    eps = 1e-8
    L = 0
    N = len(xs)
    for x in xs:
        y = gmm(x,phis,mus,covs)
        L += np.log(y+eps)

    return L/N

current_likelihold = likelihold(xs,phis,mus,covs)

for iter in range(MAX_ITERS):
    #E-step ========================
    qs = np.zeros((N,K))
    for n in range(N):
        x = xs[n]
        for k in range(K):
            phi,mu,cov = phis[k],mus[k],covs[k]
            qs[n,k] = phi*multivariate_normal(x,mu,cov)
        qs[n] /= gmm(x,phis,mus,covs)

    #M-step ========================
    qs_sum = qs.sum(axis=0)
    for k in range(K):
        #1. phis
        phis[k] = qs_sum[k]/N

        #2. mus
        c = 0
        for n in range(N):
            c += qs[n,k] * xs[n]
        mus[k] = c/qs_sum[k]

        #3. covs
        c = 0
        for n in range(N):
            z = xs[n]-mus[k]
            z = z[:,np.newaxis] #列ベクトル変換
            c += qs[n,k] * z @ z.T
        covs[k] = c/qs_sum[k]

    #終了判定 ========================
    print(f'{current_likelihold:.3f}')

    next_lilelihold = likelihold(xs,phis,mus,covs)
    diff = np.abs(next_lilelihold - current_likelihold)
    if diff < THESHOLD:
        break
    current_likelihold = next_lilelihold

# 可視化
def plot_contour(w, mus, covs):
    x = np.arange(1, 6, 0.1)
    y = np.arange(40, 100, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])

            for k in range(len(mus)):
                mu, cov = mus[k], covs[k]
                Z[i, j] += w[k] * multivariate_normal(x, mu, cov)
    plt.contour(X, Y, Z)

plt.scatter(xs[:,0], xs[:,1])
plot_contour(phis, mus, covs)
plt.xlabel('Eruptions(Min)')
plt.ylabel('Waiting(Min)')
plt.show()