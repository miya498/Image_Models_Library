import numpy as np
import torch
from torch.nn import functional as F

def log_softmax(x, dim):
    """
    ログソフトマックス関数
    """
    return x - torch.logsumexp(x, dim=dim, keepdim=True)

def kl_divergence(p, p_logits, q):
    """
    KLダイバージェンスを計算
    """
    assert p.ndim == p_logits.ndim == 2
    assert q.ndim == 1
    return torch.sum(p * (log_softmax(p_logits, dim=1) - torch.log(q).unsqueeze(0)), dim=1)

def _symmetric_matrix_square_root(mat, eps=1e-10):
    """
    対称行列の平方根を計算
    """
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return (u @ torch.diag(si) @ v.t())

def trace_sqrt_product(sigma, sigma_v):
    """
    共分散行列の積の平方根のトレースを計算
    """
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = sqrt_sigma @ sigma_v @ sqrt_sigma
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

def classifier_score_from_logits(logits):
    """
    生成モデルの評価のための分類器スコアを計算
    """
    assert logits.ndim == 2

    # 最大精度で計算
    logits = logits.double() if logits.dtype != torch.float64 else logits

    p = F.softmax(logits, dim=1)
    q = torch.mean(p, dim=0)
    kl = kl_divergence(p, logits, q)
    log_score = torch.mean(kl)
    final_score = torch.exp(log_score)

    return final_score.item()

def frechet_classifier_distance_from_activations(real_activations, generated_activations):
    """
    Frechet Inception距離（FID）を計算
    """
    assert real_activations.ndim == generated_activations.ndim == 2

    # データ型を統一
    real_activations = real_activations.double() if real_activations.dtype != torch.float64 else real_activations
    generated_activations = generated_activations.double() if generated_activations.dtype != torch.float64 else generated_activations

    # 平均と共分散行列を計算
    m = torch.mean(real_activations, dim=0)
    m_w = torch.mean(generated_activations, dim=0)

    real_centered = real_activations - m
    sigma = real_centered.t() @ real_centered / (real_activations.size(0) - 1)

    gen_centered = generated_activations - m_w
    sigma_w = gen_centered.t() @ gen_centered / (generated_activations.size(0) - 1)

    # FIDのトレース成分を計算
    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

    # 距離の各成分を計算
    trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component
    mean = torch.sum((m - m_w) ** 2)
    fid = trace + mean

    return fid.item()

# === テスト関数 ===
def test_all():
    """
    TensorFlowとPyTorchの結果を比較してテスト
    """
    rand = np.random.RandomState(1234)
    logits = torch.tensor(rand.randn(64, 1008), dtype=torch.float32)
    real_activations = torch.tensor(rand.randn(64, 2048), dtype=torch.float32)
    generated_activations = torch.tensor(rand.randn(256, 2048), dtype=torch.float32)

    # 分類器スコアを計算
    score = classifier_score_from_logits(logits)
    print(f"Classifier Score: {score}")

    # FIDを計算
    fid = frechet_classifier_distance_from_activations(real_activations, generated_activations)
    print(f"FID: {fid}")


if __name__ == '__main__':
    test_all()