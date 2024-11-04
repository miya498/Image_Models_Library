import selectivesearch
import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

#物体候補領域の生成
def get_proposals(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, regions = selectivesearch.selective_search(img_rgb, scale=500, sigma=0.9, min_size=10)
    proposals = []
    for region in regions:
        if region['rect'] not in proposals:
            x, y, w, h = region['rect']
            if w > 20 and h > 20:  # 小さすぎる領域を除外
                proposals.append((x, y, w, h))
    return proposals

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VGG16を使用して特徴抽出
model = models.vgg16(pretrained=True).features.to(device)
model.eval()

# 前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


#CNNで候補領域の特徴量抽出
def extract_features(image, proposals):
    features = []
    for (x, y, w, h) in proposals:
        cropped_img = image[y:y+h, x:x+w]
        cropped_img = cv2.resize(cropped_img, (224, 224))
        cropped_img = Image.fromarray(cropped_img)
        img_tensor = transform(cropped_img).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(img_tensor).flatten().cpu().numpy()
        features.append(feature)
    return np.array(features)



# ラベルと特徴データの準備
X_train = ...  # 特徴ベクトルのリスト
y_train = ...  # 対応するラベル

# ラベルエンコーダでラベルを整数に変換
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# SVMモデルの学習
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train_encoded)

# 特徴に基づいて予測
predicted_classes = svm.predict(X_train)

# データセットの読み込みと準備
import json

# COCOやPASCAL VOCデータセットを使用して、各領域に対応する正解ラベルと座標を取得
# COCOアノテーションファイルのパスを指定して読み込む
with open('path/to/coco/annotations/instances_train2017.json') as f:
    annotations = json.load(f)

def prepare_training_data(image, proposals, annotations):
    X_train = []  # 特徴ベクトルのリスト
    y_train = []  # ラベル
    y_reg_train = []  # 回帰モデル用の正しい境界ボックス座標

    # ここでproposalsに基づいて正解ラベルを割り当てる処理を実装
    for (x, y, w, h) in proposals:
        # アノテーションデータに基づいて、各プロポーザルに最適なラベルを付与
        # 交差割合（IoU）を使用して正しいラベルを判定し、ラベルと座標を収集
        # 適切なIoUの閾値を使用して、positiveとnegativeのデータを決定
        pass

    return np.array(X_train), np.array(y_train), np.array(y_reg_train)

def iou(box1, box2):
    # box1, box2は[x1, y1, x2, y2]形式
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area != 0 else 0

# 予測結果の評価
def evaluate_model(predictions, ground_truths):
    iou_scores = []
    for pred_box, true_box in zip(predictions, ground_truths):
        iou_score = iou(pred_box, true_box)
        iou_scores.append(iou_score)

    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU: {mean_iou:.2f}")

import matplotlib.pyplot as plt

def visualize_predictions(image, proposals, predicted_classes, adjusted_boxes):
    for (x, y, w, h), label, adjusted_box in zip(proposals, predicted_classes, adjusted_boxes):
        # 描画用の調整後の境界ボックスを取得
        x1, y1, x2, y2 = map(int, adjusted_box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # matplotlibで画像を表示
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()