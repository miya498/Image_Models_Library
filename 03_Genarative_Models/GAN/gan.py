import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ハイパーパラメータ
latent_dim = 100
image_size = 28 * 28  # MNISTの画像サイズ
batch_size = 64
epochs = 50
learning_rate = 0.0002

# データセットの準備
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Generatorの定義
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminatorの定義
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# モデルのインスタンス化
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 損失関数と最適化
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# トレーニングループ
for epoch in range(epochs):
    for batch_idx, (real_images, _) in enumerate(dataloader):
        # ラベルを生成
        real_images = real_images.view(-1, image_size).to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # -----------------
        # Discriminatorの学習
        # -----------------
        # 本物の画像での損失
        outputs = discriminator(real_images)
        loss_real = criterion(outputs, real_labels)

        # 偽の画像での損失
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        loss_fake = criterion(outputs, fake_labels)

        # 総損失
        loss_D = loss_real + loss_fake
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # -----------------
        # Generatorの学習
        # -----------------
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        loss_G = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}]  Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f}")

# サンプル画像の生成と保存
import matplotlib.pyplot as plt

z = torch.randn(16, latent_dim).to(device)
fake_images = generator(z).view(-1, 1, 28, 28).cpu().detach()

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(fake_images[i].squeeze(0), cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()