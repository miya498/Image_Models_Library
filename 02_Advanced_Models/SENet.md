# SENet（Squeeze-and-Excitation Network）モデルの解説

## 概要
SENetは、チャネルごとの重み付けを行うことで、畳み込みニューラルネットワーク（CNN）の性能を向上させるアーキテクチャです。ResNetの基本構造にSqueeze-and-Excitation（SE）ブロックを追加することで、重要な特徴を強調し、ネットワークの識別能力を高めています。

## SENetの構造
SENetは、以下のような主要な構造から成り立っています。

### 1. SEブロック（Squeeze-and-Excitation Block）
SEブロックは、入力特徴マップに対してチャネルごとの重み付けを行うことで、各チャネルの重要度を動的に調整します。

- **Squeeze**: 入力特徴マップをグローバル平均プーリングし、チャネルごとに1つの値に縮約します。
- **Excitation**: 全結合層を使用してチャネルの関係を学習し、ReLU活性化関数とシグモイド関数で重みを生成します。
- **Recalibration**: 元の入力特徴マップに重みをかけて、重要なチャネルを強調します。

### 2. 残差ブロックにSEブロックを組み込む
SENetでは、ResNetの残差ブロックにSEブロックを組み込み、各ブロックで学習された特徴にチャネルごとの重み付けを行います。これにより、深層モデルでも重要な情報を見逃さずに学習が進みます。

## コードの重要な箇所
以下に、SEブロックとSENetモデルの重要な実装部分を示します。

### SEブロック
```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch, channels)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(batch, channels, 1, 1)
        return x * y  # 入力特徴マップに重みをかける
```

### SEBasicBlock
```python
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, reduction)  # SEブロックを追加
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)  # SEブロックを適用

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # スキップ接続
        out = self.relu(out)
        return out
```

### SENetモデル
```python
class SENet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(SENet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
```

## まとめ
SENetは、チャネルごとの重要度を学習し、畳み込み層の出力に重み付けを行うことで、識別精度を向上させることができます。ResNetの残差ブロックと組み合わせることで、従来のCNNアーキテクチャに対してさらに効果的な特徴抽出を実現しています。