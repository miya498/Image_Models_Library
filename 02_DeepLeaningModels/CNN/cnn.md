# CIFAR-10 Classification using CNN with PyTorch

このプロジェクトでは、PyTorchを使用して畳み込みニューラルネットワーク（CNN）を構築し、CIFAR-10データセットで学習・分類を行います。CIFAR-10は10クラス（猫、犬、車、飛行機など）からなるカラー画像データセットで、各クラスの画像を分類するタスクです。

## データセットと前処理
### CIFAR-10
- **画像サイズ**: 32×32ピクセル
- **チャンネル数**: RGBカラー画像（3チャンネル）
- **クラス数**: 10クラス


## モデル構造
以下のCNNモデル（`SimpleCNN`）は、3つの畳み込み層と3つの全結合層で構成されています。ドロップアウトも含め、過学習を防止しています。

### nn.Conv2dの引数について
`nn.Conv2d`は2D畳み込み層を定義するためのクラスで、以下の引数を取ります。

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

- **in_channels**: 入力チャネル数（例えば、カラー画像はRGBの3チャネル）。
- **out_channels**: 出力チャネル数。次の層への出力の数。
- **kernel_size**: 畳み込みカーネルのサイズ。整数（例: 3）またはタプル（例: (3, 3)）で指定します。
- **stride**: 畳み込み操作時のストライド（デフォルトは1）。
- **padding**: 畳み込み操作時のパディング数。画像のサイズを調整するために使用されます。

### モデル構成
- **畳み込み層1 (`conv1`)**: 
  - `nn.Conv2d(3, 32, kernel_size=3, padding=1)`：入力3チャンネル、出力32チャンネル、カーネルサイズ3×3、パディング1。
- **畳み込み層2 (`conv2`)**: 
  - `nn.Conv2d(32, 64, kernel_size=3, padding=1)`：入力32チャンネル、出力64チャンネル、カーネルサイズ3×3、パディング1。
- **畳み込み層3 (`conv3`)**: 
  - `nn.Conv2d(64, 128, kernel_size=3, padding=1)`：入力64チャンネル、出力128チャンネル、カーネルサイズ3×3、パディング1。
- **プーリング層 (`MaxPool2d`)**: 
  - 各畳み込み層の後に`MaxPool2d(2, 2)`を適用して、特徴マップのサイズを2×2のウィンドウで半分に縮小します。
- **全結合層 (`fc1`, `fc2`, `fc3`)**:
  - `fc1`: 128 * 4 * 4から256ユニットに接続
  - `fc2`: 256ユニットから128ユニットに接続
  - `fc3`: 128ユニットから10ユニット（10クラスに分類）に接続
- **ドロップアウト (`Dropout`)**:
  - `nn.Dropout(0.5)`で50%の確率でランダムにユニットを無効にし、過学習を防ぎます。

