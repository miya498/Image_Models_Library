import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import cv2  # OpenCVを使用して画像サイズの補間を行う

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データセットの前処理とデータローダーの設定
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # モデルの入力サイズに合わせる
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNetの平均と標準偏差で正規化
])

dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# 学習済みのResNet18モデルをロードし、最終層をCIFAR-10用に調整
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10に対応する出力層
model = model.to(device)
model.eval()

# 最終の畳み込み層の出力を取得するためのフック関数
features = None

def hook_function(module, input, output):
    global features
    features = output

# モデルの最後の畳み込み層にフックを追加
layer_name = 'layer4'
layer = model._modules.get(layer_name)
layer.register_forward_hook(hook_function)

# クラススコアを取得するための関数
def get_cam_weights(class_idx, output):
    weights = model.fc.weight[class_idx].detach().cpu().numpy()
    return weights

# 入力画像とCAMの生成
def generate_cam(image, class_idx):
    global features
    with torch.no_grad():
        output = model(image.to(device))
        pred = F.softmax(output, dim=1)
        class_score = pred[:, class_idx]

    # 特徴マップと重みを取得
    weights = get_cam_weights(class_idx, output)
    cam = np.zeros(features.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * features[0, i].cpu().numpy()

    cam = np.maximum(cam, 0)  # ReLUを適用して負の値を排除
    cam = cam / np.max(cam)  # 正規化

    # 入力画像のサイズに補間
    cam_resized = cv2.resize(cam, (224, 224))
    return cam_resized

# CAMを可視化する関数
def show_cam_on_image(image, cam):
    image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    image_np = np.clip(image_np, 0, 1)

    plt.figure(figsize=(12, 6))
    
    # 元画像の表示
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    plt.axis('off')

    # CAMを重ねた画像の表示
    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    plt.imshow(cam, cmap='jet', alpha=0.5)  # CAMを半透明で重ねる
    plt.title('Class Activation Map (CAM)')
    plt.axis('off')

    plt.show()

# データを1つ取得してCAMを生成
data_iter = iter(data_loader)
image, label = next(data_iter)

image = image.to(device)
output = model(image)
_, pred_idx = torch.max(output, 1)

print(f"Predicted Class: {pred_idx.item()}, True Class: {label.item()}")

cam = generate_cam(image, pred_idx.item())
show_cam_on_image(image, cam)