## ResNetにおける残差ブロック（Residual Block）

### 概要
残差ブロック（Residual Block）は、ResNet（Residual Network）で提案された構造であり、深いネットワークで学習が困難になる問題、特に勾配消失問題を緩和します。深層ネットワークでは、層が増えるほど勾配が伝わりにくくなり、モデルの性能が悪化する可能性があります。残差ブロックは「スキップ接続（Skip Connection）」を導入することで、この問題を解決しています。

### 残差ブロックの構造
残差ブロックは、通常の畳み込み層とスキップ接続から構成されます。スキップ接続は、入力をそのまま出力に加える役割を持ち、以下のように表されます。

\[
\text{output} = F(x) + x
\]

ここで、\( F(x) \) は残差ブロック内部の畳み込み層と活性化関数によって学習された特徴量、\( x \) はスキップ接続を通じてブロックの出力に加えられる入力です。


### 実装のポイント
1. **畳み込み層**:
   - 通常の畳み込み層とバッチ正規化、ReLU活性化関数を用います。これにより、特徴量が適切に学習されます。
   
2. **スキップ接続**:
   - ストライドやチャンネル数が異なる場合は、スキップ接続のために1×1の畳み込みとバッチ正規化を用いて、入力の次元を変換する「ダウンサンプリング」を行います。
   
3. **残差結合**:
   - スキップ接続の結果と畳み込み層の出力を足し合わせた後、ReLU関数を適用して非線形性を導入します。

### 残差ブロックの実装例
以下は、PyTorchを用いた残差ブロックの実装例です。

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # スキップ接続（入力xを直接出力に加える）
        out = self.relu(out)
        return out
