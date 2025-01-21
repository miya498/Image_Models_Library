# Breast Cancer Classification using MLP

このプロジェクトでは、PyTorchを用いた多層パーセプトロン（MLP）を使い、「Breast Cancer Wisconsin」データセットを使用した2値クラス分類を行います。乳がん診断における腫瘍が良性か悪性かを予測するモデルを構築します。

## データセット
使用するデータセットは、`sklearn.datasets`から提供されている「Breast Cancer Wisconsin」です。このデータセットには、腫瘍に関する30の特徴量（半径、テクスチャ、平滑度、コンパクト性など）が含まれており、569サンプルのラベル（良性:0または悪性:1）を使って2値分類を行います。

- **特徴量の数**: 30
- **ラベル**: 2クラス（良性：0、悪性：1）
- **サンプル数**: 569

## データの分割と正規化
データセットを学習用とテスト用に分割し、訓練データが80％、テストデータが20％の割合で構成されています。

1. **データ分割**: `train_test_split`で学習データとテストデータに分割しています。
2. **標準化**: データの特徴量を`StandardScaler`で標準化します。これにより、各特徴量が平均0、分散1のガウス分布に近づくようにスケーリングされ、学習の収束が速くなります。

## ネットワークモデル
2値分類タスクにはシンプルなMLP（多層パーセプトロン）を使用しています。以下はネットワークモデルの構成です。

- **入力層**: 30ユニット（特徴量の数）
- **隠れ層1**: 16ユニット、活性化関数はReLU
- **隠れ層2**: 16ユニット、活性化関数はReLU
- **出力層**: 1ユニット、活性化関数はSigmoid（2値分類をするため、出力を0から1の範囲に変換）

損失関数として`BCELoss`（Binary Cross Entropy Loss）を使用し、最適化アルゴリズムには`Adam`を採用しています。

## IterationとEpochについて
- **Epoch**: エポックは、全ての学習データがモデルに一度通過するまでの単位です。本プロジェクトでは、エポック数を100に設定しており、モデルは学習データ全体を100回繰り返し学習します。
- **Iteration**: イテレーションは、1バッチ（データの一部）を使ってパラメータが更新される単位です。このコードではデータ全体を使用しているため、1エポックにおけるイテレーション数は1となります。

## トレーニング
エポック数100で学習を行い、10エポックごとに損失を表示して学習の進捗を確認します。

```python
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
```

## モデルの評価
テストデータを使用してモデルを評価し、分類精度（accuracy）を算出します。予測結果が0.5以上の場合、悪性腫瘍として分類します。

```python
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred >= 0.5).float()
    accuracy = (y_pred_class == y_test).float().mean()
    print(f'Accuracy: {accuracy.item() * 100:.2f}%')
```

## 結果
最終的に、テストデータでの正解率が表示されます。この精度によって、モデルの乳がん診断における分類性能を確認できます。
