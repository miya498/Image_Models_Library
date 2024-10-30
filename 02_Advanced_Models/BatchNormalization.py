import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# MNISTデータセットの読み込みと前処理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Batch Normalizationありのモデル定義
class MLP_WithBatchNorm(nn.Module):
    def __init__(self):
        super(MLP_WithBatchNorm, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Batch Normを追加
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)  # Batch Normを追加
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)   # Batch Normを追加
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)   # Batch Normを追加
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # 画像データを1次元に変換
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

# Batch Normalizationなしのモデル定義
class MLP_NoBatchNorm(nn.Module):
    def __init__(self):
        super(MLP_NoBatchNorm, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

# トレーニングと評価関数の定義
def train_model(model, criterion, optimizer, num_epochs=10):
    train_losses, test_accuracies = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # テストデータで評価
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total * 100
        test_accuracies.append(accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return train_losses, test_accuracies

# モデルとハイパーパラメータの設定
num_epochs = 20
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()

# Batch Normなしモデル
model_no_bn = MLP_NoBatchNorm()
optimizer_no_bn = optim.Adam(model_no_bn.parameters(), lr=learning_rate)
print("Training model without Batch Normalization...")
train_losses_no_bn, test_accuracies_no_bn = train_model(model_no_bn, criterion, optimizer_no_bn, num_epochs)

# Batch Normありモデル
model_with_bn = MLP_WithBatchNorm()
optimizer_with_bn = optim.Adam(model_with_bn.parameters(), lr=learning_rate)
print("\nTraining model with Batch Normalization...")
train_losses_with_bn, test_accuracies_with_bn = train_model(model_with_bn, criterion, optimizer_with_bn, num_epochs)

# グラフ描画
epochs_range = range(1, num_epochs + 1)
plt.figure(figsize=(12, 5))

# 損失グラフ
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses_no_bn, label='No Batch Norm')
plt.plot(epochs_range, train_losses_with_bn, label='With Batch Norm')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss (Batch Norm vs No Batch Norm)')
plt.legend()

# 精度グラフ
plt.subplot(1, 2, 2)
plt.plot(epochs_range, test_accuracies_no_bn, label='No Batch Norm')
plt.plot(epochs_range, test_accuracies_with_bn, label='With Batch Norm')
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy (Batch Norm vs No Batch Norm)')
plt.legend()

plt.show()
