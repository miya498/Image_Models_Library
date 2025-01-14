import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ハイパーパラメータ
image_size = 28
channels = 1
timesteps = 1000
batch_size = 128
epochs = 20
learning_rate = 1e-4

# データセットの読み込み
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ベータスケジュール（線形増加）
beta = np.linspace(1e-4, 0.02, timesteps).astype(np.float32)
alpha = 1.0 - beta
alpha_hat = np.cumprod(alpha)

# モデルアーキテクチャ（簡易的なUNet）
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, t):
        # 時間埋め込みを加える（簡易バージョン）
        t_embed = t.view(-1, 1, 1, 1).expand_as(x)
        x = x + t_embed
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# モデル、オプティマイザ、損失関数
model = SimpleUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# トレーニングプロセス
os.makedirs("results", exist_ok=True)
alpha_hat = torch.tensor(alpha_hat, device=device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        t = torch.randint(0, timesteps, (images.size(0),), device=device).long()
        noise = torch.randn_like(images)
        noisy_images = (
            torch.sqrt(alpha_hat[t, None, None, None]) * images +
            torch.sqrt(1 - alpha_hat[t, None, None, None]) * noise
        )

        optimizer.zero_grad()
        pred_noise = model(noisy_images, t)
        loss = criterion(pred_noise, noise)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # サンプル生成
    model.eval()
    with torch.no_grad():
        samples = torch.randn((16, channels, image_size, image_size), device=device)
        for t in reversed(range(timesteps)):
            alpha_t = alpha[t]
            beta_t = beta[t]
            alpha_hat_t = alpha_hat[t]

            pred_noise = model(samples, torch.tensor([t], device=device).float())
            samples = (samples - beta_t / torch.sqrt(1 - alpha_hat_t) * pred_noise) / torch.sqrt(alpha_t)
        save_image(samples, f"results/sample_epoch_{epoch + 1}.png", normalize=True)

print("トレーニング完了！生成結果は 'results/' フォルダに保存されました。")