# MNIST Classification using MLP with PyTorch

このプロジェクトでは、PyTorchを用いた多層パーセプトロン（MLP）によるMNISTデータセットの多クラス分類を行います。MNISTデータセットは手書き数字の画像で構成されており、0～9の数字を10クラスで分類します。

## データセット
使用するデータセットは`torchvision.datasets`に含まれる「MNIST」です。手書き数字の画像データが含まれ、機械学習や深層学習の基本的なタスクとして広く利用されています。

- **画像サイズ**: 28×28ピクセル
- **チャンネル数**: グレースケール（1チャンネル）
- **クラス数**: 10クラス（0～9の数字）

## データの分割と前処理
1. **データ分割**: データは学習データ（60,000枚）とテストデータ（10,000枚）に分かれています。
2. **前処理**:
   - `ToTensor()`で画像をテンソル形式に変換し、ニューラルネットワークで扱いやすいように標準化しています。
   - `Normalize((0.5,), (0.5,))`でピクセル値を正規化し、各ピクセル値が平均0、標準偏差1の分布になるように調整しています。

## ネットワークモデル
多層パーセプトロン（MLP）を使用して、画像を10クラスに分類します。モデルの構成は以下の通りです。

- **入力層**: 28×28 = 784ユニット（28×28の画像を1次元に変換）
- **隠れ層1**: 128ユニット、活性化関数はReLU
- **隠れ層2**: 64ユニット、活性化関数はReLU
- **出力層**: 10ユニット（0～9のクラスに対応）

出力層には活性化関数は使わず、ソフトマックスは損失関数内で自動的に適用されます。

## ミニバッチ学習
ミニバッチを用いることで、計算効率と収束の安定性を向上させています。バッチサイズは64に設定し、`DataLoader`を使用してミニバッチを生成しています。

## 損失関数と最適化手法
- **損失関数**: `CrossEntropyLoss`を使用し、多クラス分類における誤差を計算します。
- **最適化手法**: `Adam`オプティマイザを使用し、学習率は0.001に設定しています。

## IterationとEpochについて
- **Epoch**: データセット全体が1回モデルを通過する回数です。本プロジェクトではエポック数を5に設定しています。
- **Iteration**: 1バッチ（ミニバッチ）がモデルを通過する回数です。各エポックの中で、`train_loader`内のバッチ数に相当するイテレーション数が実行されます。

## トレーニング
以下のコードでトレーニングを行い、5エポックにわたってモデルを学習します。

```python
epochs = 5
for epoch in range(epochs):
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
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')
```

## テストと予測結果の表示
テストデータを使ってモデルを評価し、正解率（accuracy）を計算します。さらに、10枚の画像を表示し、それぞれの予測結果と実際のラベルを確認します。

```python
def show_predictions():
    model.eval()
    images, labels = next(iter(test_loader))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    fig = plt.figure(figsize=(12, 6))
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f"Label: {labels[i].item()}\nPred: {predicted[i].item()}")
        ax.axis('off')
    plt.show()

show_predictions()
```

上記のコードで、テストデータから10枚の画像を表示し、それぞれの画像の実際のラベルとモデルの予測結果が表示されます。これにより、モデルの予測が正しいかどうかを確認できます。