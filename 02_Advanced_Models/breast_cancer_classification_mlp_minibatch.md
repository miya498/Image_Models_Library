# Breast Cancer Classification using MLP with Mini-Batch Training

このプロジェクトでは、PyTorchを用いて「Breast Cancer Wisconsin」データセットで2値クラス分類を行います。多層パーセプトロン（MLP）を用い、ミニバッチ学習を取り入れて効率的な学習を実現しています。ミニバッチを用いることで、収束速度やメモリ効率が向上します。

## データセット
使用データセットは`sklearn.datasets`に含まれる「Breast Cancer Wisconsin」です。このデータセットは乳がん診断のデータで、腫瘍が良性か悪性かを分類するために用いられます。

- **特徴量の数**: 30
- **ラベル**: 2クラス（良性：0、悪性：1）
- **サンプル数**: 569

## データの分割と正規化
1. **データ分割**: `train_test_split`を使用して、データを80％の学習用と20％のテスト用に分割します。
2. **標準化**: `StandardScaler`でデータを標準化し、全ての特徴量が平均0、分散1になるようにスケーリングします。標準化により、学習の安定性と収束速度が向上します。

## ネットワークモデル
本プロジェクトでは、多層パーセプトロン（MLP）を構築して2値分類を行います。

- **入力層**: 30ユニット（特徴量数に基づく）
- **隠れ層1**: 16ユニット、活性化関数はReLU
- **隠れ層2**: 16ユニット、活性化関数はReLU
- **出力層**: 1ユニット、活性化関数はSigmoid（2値分類用に0から1への変換）

損失関数には`BCELoss`（Binary Cross Entropy Loss）を使用し、最適化アルゴリズムとして`Adam`を用いています。

## ミニバッチ学習
ミニバッチ学習では、データを小さなバッチに分けて処理を行います。本プロジェクトでは、`DataLoader`を用いて`batch_size=32`で設定しており、32サンプルごとにパラメータを更新します。

### ミニバッチ学習のメリット
- **計算効率の向上**: 全データを一度に処理するのではなく、小分けにして計算するためメモリ効率が高まります。
- **収束の安定性**: イテレーションごとに損失の更新が行われ、より安定してモデルが収束します。
- **学習時間の短縮**: 大規模データの場合、ミニバッチ学習によって計算時間が短縮されます。

## IterationとEpochについて
- **Epoch**: エポックは、データセット全体が1回モデルを通過する回数です。本プロジェクトではエポック数を100に設定し、データセット全体が100回モデルを通過するようにしています。
- **Iteration**: イテレーションは、各ミニバッチが1回モデルを通過する回数です。各エポックにおけるイテレーション数は、`train_loader`内のバッチ数に相当します。

## トレーニング
以下のコードで、100エポックにわたりモデルを学習し、10エポックごとに平均損失を表示しています。

```python
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')
```

## モデルの評価
テストデータもミニバッチで評価し、全体の分類精度（accuracy）を算出します。

```python
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        y_pred_class = (y_pred >= 0.5).float()
        correct += (y_pred_class == y_batch).sum().item()
        total += y_batch.size(0)
    accuracy = correct / total * 100
    print(f'Accuracy: {accuracy:.2f}%')
```

## 結果
最終的に、テストデータでの正解率（accuracy）が表示され、モデルの乳がん診断の分類性能を確認できます。