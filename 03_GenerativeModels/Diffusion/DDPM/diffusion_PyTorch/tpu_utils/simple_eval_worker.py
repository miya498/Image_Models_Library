import os
import pickle
import time

import numpy as np
import torch
from tqdm import trange

from .tpu_utils import Model, make_ema, normalize_data
from torch.utils.data import DataLoader


class SimpleEvalWorker:
    def __init__(self, model_constructor, dataset, batch_size, device="cuda"):
        self.device = device
        self.batch_size = batch_size
        self.dataset = dataset

        # DataLoaderの作成
        self.train_loader = DataLoader(dataset.train_input_fn(), batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(dataset.eval_input_fn(), batch_size=batch_size, shuffle=False)

        # モデル初期化
        self.model = model_constructor().to(self.device)
        assert isinstance(self.model, Model)

        # EMA設定
        self.ema = make_ema(self.model)

    def _make_progressive_sampling(self, num_samples):
        """
        Progressive Sampling用のデータ生成
        """
        self.model.eval()
        all_samples = []
        num_batches = int(np.ceil(num_samples / self.batch_size))
        for _ in trange(num_batches, desc="Generating samples"):
            noise = torch.randn((self.batch_size, *self.dataset.image_shape), device=self.device)
            samples = self.model.progressive_sample(noise)
            all_samples.append(samples.cpu().numpy())
        return np.concatenate(all_samples, axis=0)[:num_samples]

    def dump_progressive_samples(self, output_dir, num_samples=50000):
        """
        Progressive Samplesを保存
        """
        os.makedirs(output_dir, exist_ok=True)
        samples = self._make_progressive_sampling(num_samples)
        output_path = os.path.join(output_dir, "progressive_samples.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Progressive samples saved to {output_path}")

    def dump_bpd(self, loader, output_dir, train=True):
        """
        BPD（Bits Per Dimension）を計算して保存
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()
        all_bpd = []

        with torch.no_grad():
            for batch in trange(len(loader), desc="Calculating BPD"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                bpd = self.model.bpd(images, labels)
                all_bpd.append(bpd.cpu().numpy())

        all_bpd = np.concatenate(all_bpd, axis=0)
        output_path = os.path.join(output_dir, f"bpd_{'train' if train else 'eval'}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(all_bpd, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"BPD results saved to {output_path}")

    def run(self, mode, logdir, load_ckpt):
        """
        メイン処理の実行
        """
        # モデルチェックポイントのロード
        checkpoint = torch.load(load_ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ema.load_state_dict(checkpoint["ema_state_dict"])
        print(f"Checkpoint loaded from {load_ckpt}")

        # 出力ディレクトリの作成
        output_dir = os.path.join(logdir, "simple_eval")
        os.makedirs(output_dir, exist_ok=True)

        # 指定されたモードで処理を実行
        if mode == "bpd_train":
            self.dump_bpd(self.train_loader, output_dir, train=True)
        elif mode == "bpd_eval":
            self.dump_bpd(self.eval_loader, output_dir, train=False)
        elif mode == "progressive_samples":
            self.dump_progressive_samples(output_dir)
        else:
            raise ValueError(f"Unknown mode: {mode}")