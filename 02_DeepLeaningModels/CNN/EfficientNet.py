import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = in_channels // reduction
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size, reduction=4):
        super(MBConvBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(hidden_dim)

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, groups=hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # Squeeze and Excitation
        self.se = SqueezeExcitation(hidden_dim, reduction)

        # Pointwise convolution
        self.pointwise_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = Swish()

    def forward(self, x):
        identity = x
        out = x

        # Expansion
        if hasattr(self, 'expand_conv'):
            out = self.activation(self.bn0(self.expand_conv(x)))

        # Depthwise Convolution
        out = self.activation(self.bn1(self.depthwise_conv(out)))

        # Squeeze and Excitation
        out = self.se(out)

        # Pointwise Convolution
        out = self.bn2(self.pointwise_conv(out))

        if self.use_residual:
            out += identity

        return out

class EfficientNet(nn.Module):
    def __init__(self, width_coefficient, depth_coefficient, num_classes=1000, dropout_rate=0.2):
        super(EfficientNet, self).__init__()
        base_channels = 32
        final_channels = 1280

        # Stage 1
        self.stem_conv = nn.Conv2d(3, base_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(base_channels)
        self.stem_activation = Swish()

        # Configurations for each stage: (in_channels, out_channels, expand_ratio, stride, kernel_size, num_blocks)
        self.blocks_config = [
            (base_channels, 16, 1, 1, 3, int(1 * depth_coefficient)),
            (16, 24, 6, 2, 3, int(2 * depth_coefficient)),
            (24, 40, 6, 2, 5, int(2 * depth_coefficient)),
            (40, 80, 6, 2, 3, int(3 * depth_coefficient)),
            (80, 112, 6, 1, 5, int(3 * depth_coefficient)),
            (112, 192, 6, 2, 5, int(4 * depth_coefficient)),
            (192, 320, 6, 1, 3, int(1 * depth_coefficient))
        ]

        # Build blocks
        self.blocks = self._make_blocks(width_coefficient)

        # Final stage
        self.final_conv = nn.Conv2d(int(320 * width_coefficient), final_channels, kernel_size=1, bias=False)
        self.final_bn = nn.BatchNorm2d(final_channels)
        self.final_activation = Swish()

        # Classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(final_channels, num_classes)

    def _make_blocks(self, width_coefficient):
        layers = []
        for in_channels, out_channels, expand_ratio, stride, kernel_size, num_blocks in self.blocks_config:
            out_channels = int(out_channels * width_coefficient)
            for i in range(num_blocks):
                stride = stride if i == 0 else 1
                layers.append(MBConvBlock(in_channels, out_channels, expand_ratio, stride, kernel_size))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem_activation(self.stem_bn(self.stem_conv(x)))
        x = self.blocks(x)
        x = self.final_activation(self.final_bn(self.final_conv(x)))
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# EfficientNet-B0のインスタンスを作成
efficientnet_b0 = EfficientNet(width_coefficient=1.0, depth_coefficient=1.0, num_classes=10)
efficientnet_b0 = efficientnet_b0.to(device)

# データセットの前処理とデータローダー設定
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # モデルに合わせたサイズ
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
optimizer = optim.Adam(efficientnet_b0.parameters(), lr=0.001)

# 学習ループ
num_epochs = 10
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    # トレーニングフェーズ
    efficientnet_b0.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = efficientnet_b0(images)
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
    efficientnet_b0.eval()
    running_val_loss = 0.0
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = efficientnet_b0(images)
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
