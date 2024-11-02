import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2  # OpenCVを使用して画像サイズの補間を行う

class AttentionBranch(nn.Module):
    def __init__(self, in_channels, attention_size):
        super(AttentionBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, attention_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(attention_size)
        self.conv2 = nn.Conv2d(attention_size, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map = self.conv1(x)
        attention_map = F.relu(self.bn1(attention_map))
        attention_map = self.conv2(attention_map)
        attention_map = self.sigmoid(attention_map)
        return attention_map

class ABN(nn.Module):
    def __init__(self, base_model, attention_size=256, num_classes=10):
        super(ABN, self).__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-2])  # 最後の畳み込み層まで使用
        self.attention_branch = AttentionBranch(in_channels=512, attention_size=attention_size)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.features(x)  # 特徴マップを取得
        attention_map = self.attention_branch(features)  # Attentionマップを生成
        
        # Attentionを特徴マップに適用
        enhanced_features = features * attention_map

        # 平均プーリングと全結合層
        pooled_features = F.adaptive_avg_pool2d(enhanced_features, (1, 1)).view(x.size(0), -1)
        output = self.fc(pooled_features)

        return output, attention_map

# モデルの準備
base_model = models.resnet18(pretrained=True)
abn_model = ABN(base_model=base_model, attention_size=256, num_classes=10)

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
abn_model = abn_model.to(device)

# CIFAR-10データセットの前処理とデータローダー設定
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # モデルの入力サイズに合わせる
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNetの平均と標準偏差で正規化
])

dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 損失関数と最適化手法の設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(abn_model.parameters(), lr=0.001)

# 学習と検証ループ
num_epochs = 10
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    # トレーニングフェーズ
    abn_model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, attention_map = abn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 検証フェーズ
    abn_model.eval()
    running_val_loss = 0.0
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, attention_map = abn_model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_loss = running_val_loss / len(val_loader.dataset)
    val_accuracy = 100 * correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n')

# 学習損失と精度のグラフ描画
epochs_range = range(1, num_epochs + 1)
plt.figure(figsize=(14, 6))

# 損失のグラフ
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()

# 精度のグラフ
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Train and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Attentionマップの可視化
def show_attention_map(image, attention_map):
    image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    image_np = np.clip(image_np, 0, 1)

    attention_map_resized = attention_map.squeeze().detach().cpu().numpy()  # .detach()を追加
    attention_map_resized = cv2.resize(attention_map_resized, (224, 224))
    attention_map_resized = np.clip(attention_map_resized, 0, 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    plt.imshow(attention_map_resized, cmap='jet', alpha=0.5)
    plt.title('Attention Map')
    plt.axis('off')

    plt.show()

# データを1つ取得してAttentionマップを表示
data_iter = iter(val_loader)
image, label = next(data_iter)
image = image.to(device)
output, attention_map = abn_model(image)
show_attention_map(image[0], attention_map[0])